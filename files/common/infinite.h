// "infinite v1": a two-tier far-context KV policy for llama.cpp.
//
//   TRUNK (GPU)  dense window of the last W tokens at their true positions, plus the <s> sink
//                and the first n_pin tokens, pinned just beyond the band.  Normal attention.
//   FAR   (HOST) every token that leaves the trunk keeps its REAL K/V in host memory, organised
//                into units (sentences, <= unit_max tokens) and chunks (chunk tokens).
//   INDEX        (a) an inverted index from rare token ids to units;
//                (b) per (layer, KV head) chunk-local k-means clusters of ~chunk/n_clu keys on the
//                    n_planes slowest RoPE planes (de-rotated -> position free), each summarised
//                    by a sentinel (n, mu, Sigma).
//   BAND  (GPU)  S cells, re-gathered every block of B tokens: the units triggered by the block's
//                rare words + the top-`topc` clusters per (layer, KV head) by sentinel score
//                against the block's mean query + the last n_roll rolled-off tokens.  Gathered
//                cells are placed at slot positions W + 1 + rank, exactly as in paper13.
//
// The attention kernel is untouched and no sentinel term enters the softmax: everything is
// expressed as (a) which cells exist and (b) what position each cell carries.
#pragma once

#include "llama.h"
#include "ggml-backend.h"
#include "infinite-freq.h"   // inf_freq: the online unigram rarity model

#include <array>
#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------------------------
// host-side far K/V store: slab-allocated so it never reallocates.
// layout per token: [layer][row] for K and, separately, for V, in the cache's own row format.
// ---------------------------------------------------------------------------------------------
struct inf_hostkv {
    size_t  row       = 0;     // bytes per cell per layer
    int32_t n_layer   = 0;
    size_t  tok_bytes = 0;     // row * n_layer
    size_t  slab_tok  = 0;     // tokens per slab
    std::vector<uint8_t *> kslab, vslab;
    int64_t n_tok = 0;

    ~inf_hostkv();
    void init(size_t row_, int32_t n_layer_, size_t slab_bytes = (size_t) 256 << 20);
    void ensure(int64_t t);
    uint8_t * k(int64_t t) { return kslab[t / slab_tok] + (t % slab_tok) * tok_bytes; }
    uint8_t * v(int64_t t) { return vslab[t / slab_tok] + (t % slab_tok) * tok_bytes; }
    size_t bytes() const { return (kslab.size() + vslab.size()) * slab_tok * tok_bytes; }
};

// ---------------------------------------------------------------------------------------------
struct inf_params {
    int32_t W     = 8192;      // dense trunk window
    int32_t S     = 16384;     // retrieval band, in cells
    int32_t B     = 128;       // block: one gather per B tokens
    int32_t n_pin = 128;       // tokens after the sink kept resident (the sink itself is extra)
    int32_t n_roll = 128;      // last rolled-off tokens always gathered

    int32_t unit_max = 96;     // max tokens in a unit
    // DESIGN_v2 SS9: fixed chunks are v2's DEFAULT unit rule.  A unit is [64i, 64i + 96), so
    // units are 96 tokens long and overlap by 32; the stride guarantees that no span of 64
    // tokens can be split across a boundary without one unit containing it whole, which is what
    // the sentence rule failed at (it split 160 of 160 decimal and abbreviation sites and made
    // units 5 tokens long on hard-wrapped prose).  A token belongs to two units, so the band
    // dedups by token position, not by unit.
    int32_t unit_len    = 96;  // fixed-chunk length
    int32_t unit_stride = 0;   // fixed-chunk stride; 0 = use the sentence rule instead.  09-25 v2bld26 DEFAULT 0: sentence units for every source (LOG 137; was 64); unitstride=64 = the chunk rule
    int32_t chunk    = 2816;   // tokens per clustering chunk (22 blocks of 128)
    int32_t n_clu    = 64;     // clusters per chunk per (layer, KV head)  -> ~44 keys each
    int32_t km_iter  = 6;      // k-means iterations
    int32_t n_planes = 15;     // slowest RoPE planes used for clustering and scoring
    int32_t topc     = 4;      // clusters gathered per (layer, KV head)
    int32_t qtok     = 16;     // tokens at the end of a block averaged into the mean query
    int32_t shortlist = 256;   // clusters per (layer, KV head) that get the full quadratic term
    double  rare_tail = 0.02;  // (reporting only) tail occurrence mass that counts as "rare"
    int32_t max_post  = 8192;  // query tokens in more units than this are not discriminative
    double  trig_frac = 1.0;   // at most this fraction of the band goes to triggered units (09-24 shipping: 1.0; was 0.5)

    bool    use_clusters = false;   // 09-24 shipping default (noclu); was true
    bool    use_triggers = true;
    bool    use_roll     = true;
    bool    syncall      = false;  // extra llama_synchronize() around the gather (race probe)
    bool    verify       = false;  // check the band against the cache after every placement
    bool    regather     = true;   // gather again at the prompt->decode hand-off (see LOG 15)

    // --- event-driven refresh (Mark's design).  The policy assembles whole words online; when a
    //     word turns out to be RARE the moment its last piece arrives, its posting-list units are
    //     pulled into the band immediately, in prefill and in decode, without waiting for the
    //     block boundary.  The index is keyed by whole words, not pieces.
    bool    evt       = true;     // enable the event path (09-24 shipping default; was false)
    bool    evt_block = false;    // also keep the block-boundary gather (false: pins/rank only; 09-24 shipping: false = evtnoblock)
    int32_t evt_cap   = 8;        // posting-list units pulled per event, most recent first
    double  evt_surp  = 11.0;     // summed -log p over the word's pieces, in nats, to count rare
    int32_t evt_max   = 2048;     // cells one event may write (a guard, not a policy)
    bool    evt_log   = false;    // print one EVTLOG line per event attempt (driver diagnosis)
    // TRUNK-SEEDED HAND-OFF.  The bridge sentence of a two-hop question is already in the trunk
    // when the hand-off gather runs, so its rare words can be added to the query set -- at a
    // discount, so the request's own words still rank first -- and the far units about things
    // recently discussed become resident BEFORE generation starts.  Many two-hop cases then become
    // one-hop at the hand-off, with events still covering words that appear only in the output.
    int32_t seed_span = 4096;     // tokens of trunk to seed query words from (0 = off).  09-24 FINAL (v2bld23): 4096 with seed_words 32 (coordinator; LOG 135c: hop2 21/32 vs sentence 22, off 13)
    // The lexical trigger needs only TOKENS, not a forward pass, so the request's rare words are
    // known before the request is encoded.  With prehand the whole hand-off - the request's words
    // and the seed - runs BEFORE the request's forward pass, so the far units the question names
    // are resident while the question itself is being encoded, not merely before generation.
    int32_t prehand   = 1;        // DEFAULT ordering; 0 is the A/B control only
    // tailq generalises prehand.  Prefill is affected too: when a large text is prefilled, the
    // LAST Q tokens are held back and encoded as if they were part of the query, after the gather.
    // The tail's K/V stay in the trunk and are what the question reads, so if the tail's tokens -
    // a bridge, a summary, the last paragraphs - need far units visible to bind to them, encoding
    // the tail without the band leaves that binding out of the trunk.
    //   tailq = 0  ->  prehand: only the request is held back
    //   tailq = T  ->  the old periodic-gather prefill
    // Cost: one gather, plus a slightly wider attention over Q tokens.
    // The derived default is min(4096, W/2); -1 asks for it.
    int32_t tailq     = 0;        // OPTION, not a default: see DESIGN_v2 7b-v/vi
    // pfdepth generalises tailq.  By induction the residency effect can touch everything in the
    // prefill, so what is wanted is a PREFILL GATHER DEPTH: window-only up to T-N, then the last N
    // tokens prefilled with the machinery in place - gather from each block's words and the seed
    // before that block is encoded - then the request as a query.
    //   N = 0   window-only prefill (v1's default here)
    //   N = W   the derived default: one trunk's worth of history encoded with the band visible
    //   N = T   v1's streamed prefill
    // The old +45-58% prefill cost came from 8K-token gathers and the sentinel scan; with word
    // triggers a gather is a few hundred tokens, so the cost curve is new information.
    int32_t pf_depth  = 0;        // -1 = derive W; tokens at the end of prefill that gather
    // pf_blk > 0: gather every pf_blk tokens.  pf_blk < 0: gather once per UNIT - prefill the
    // document chunk by chunk with the deep fill done before each chunk.  Units are ~96 tokens and
    // eval feeds in blocks of B, so a unit-triggered gather is realised at the next block
    // boundary; the report says so rather than claiming token-exact unit granularity.
    int32_t pf_blk    = 128;      // gather every pf_blk tokens; < 0 = once per unit
    // With evictafter the band the hand-off placed is PARKED once the request has been encoded and
    // before the first generated token: the cells are removed from the sequence, so the answer
    // cannot attend to them.  Anything the answer still gets right came through the residuals the
    // request's own encoding produced.  Events still fire during decode.
    int32_t evictafter = 0;       // MEASUREMENT ONLY - never a shipping default: the
                                  // answer's copy still reads the unit, so the band is NOT
                                  // released after the request is encoded
    double  seed_disc = 0.5;      // discount on a seed word's surprisal
    bool    seed_indexed = false; // seed candidates must have a posting list (NOTE 09-25: ANY posting; every trunk word has one, so this filters nothing -- LOG 137h)
    bool    seed_far = true;      // 09-25 v2bld30 SHIPPING DEFAULT 1 (LOG 137h rule: hop2 +3/+4/+13/+7, key3r 95/93): seed candidates must have a posting in the FAR region; seedfar=0 = v2bld26
    int32_t seed_words = 32;      // cap on seed words used, highest surprisal first (0 = all).  09-24 FINAL (v2bld23): 32, rarest first
    // oracle_seed: seed EXACTLY the handle word, supplied by the driver.  This is the clean
    // timing-only corner: the term is known to be the right one, so anything it fails to fetch
    // is a timing or placement failure and not a choice-of-term failure.
    bool    oracle_seed = false;
    // prerope: store the key BEFORE the rotation and rotate it exactly once, at its slot, when
    // the cell is placed.  The current path stores the post-RoPE key and rotates a second time
    // via the K-shift, which at 1M means two f32 rotations with a ~400K delta.  Pre-RoPE is also
    // what the compressed store needs: a key quantises better without the position baked in.
    // Cost: placement must REWRITE K rather than only move positions, so it is one contiguous
    // full-band write per placement instead of a cells_place.
    // v2 DEFAULT.  v2's store is pre-RoPE by construction - the compressed keys are pre-rotation
    // - so the K-shift path does not exist for band cells here at all.  The shift remains only
    // for the trunk's own roll-forward, which is one block of delta.  prerope=0 exists only in
    // the v1 driver, for the A/B.  Gated on the 128K tie test and nothing else.
    bool    prerope = false;   // 09-24 SHIPPING DEFAULT 0 (Mark: "no downside"; PR0*/PR1* 328124-131 all 16/16, 0 bad cells; +25% prefill)
    // 09-23 pin fix (default 0 = the in-place llama_memory_seq_remap every block, as before).
    // N >= 1: every N-th pin move re-places the pinned prefix from the HOST copy, rotated ONCE to
    // the new slot (prerope: one exact rotation; else one queued K-shift from the true position),
    // K only; the moves in between stay remaps.  v1 PINDRIFT1MB: sink K cosine 0.43 at 1M.
    // 09-24: DEFAULT 1 (throughput agent, coordinator's call).  v1 Mistral 1M niah1: pinhost=1 +
    // pre-encode hand-off 16/16 (PINHOST1M_PRE 324442) vs 9/16 without (PRFIX0M 322408); decode
    // cost -0.3%, prefill -4% at 128K on a full H200 (PERFGATE2 324415).  pinhost=0 = the old path.
    int32_t pinhost = 1;
    // 09-24 band check (throughput agent; bandchk=1, diagnostic only, reads and never writes).
    // After every placement (hand-off/block gather, rank-only re-place, event fetch) compare every
    // live band cell's V row byte for byte with the host V of the token its provenance (the cell's
    // ext tok) names, and its K row with that token's host K rotated once in double to the slot
    // the cell claims; the same for every trunk cell at its true position (PIPELINE T24).  Tests
    // the "K placed in one cell order, V in another" hypothesis for the Mistral three-key misses.
    bool    bandchk = false;
    // 09-24 T12 fix: under window-only prefill (nopfgather) take window_update() only once
    // p >= P_on, so a context below L stays dense (it was read through an 8K window + the pins
    // from ~8.3K on; V2PIN1/V2PIN0 T12 32768/32768 logits differ).  t12fix=0 = the old path,
    // which reproduces every row before V2BLD9.
    bool    t12fix = true;
    // 09-24 wordall: with fixed chunks, post a whole word to EVERY chunk that contains it, not only
    // the first chunk to close after it.  DESIGN_v2 §9 guarantees that any span of <= 33 tokens lies
    // wholly inside chunk floor(a/64); the word index broke it: a needle's key word completing in
    // the overlap was posted only to the EARLIER chunk, which can cut the needle's tail off (v2
    // Mistral+Llama 128K key3r ex 13: 19/23 needle tokens resident, the value's last digits not).
    // wordall=0 = the old posting rule (every row before V2BLD10).
    bool    wordall = true;
    // 09-24 PERF (throughput agent).  capblk: during decode, capture once per BLOCK instead of per
    // token.  The pre-RoPE K still has to be read from the graph after every decode (the graph
    // keeps only the last ubatch); V is read from the cache for the whole block at the block
    // boundary or before anything that needs it (block_update, events, band check, a new turn).
    // Byte-identical host store.  capnok: under prerope, skip the cache K read in capture(), which
    // the graph's pre-RoPE K overwrote anyway (half of capture's D2H, prefill and decode).
    // capblk=0 / capnok=0 = the old path.
    bool    capblk = true;
    bool    capnok = true;
    // 09-24 evreenc (throughput agent, after conceptval P.12: Mistral does not read a unit that
    // arrives by a mid-generation event even when it is in the band and unmasked; re-encoding the
    // question and the generation so far with the unit resident restores the read).  evreenc=1:
    // when a DECODE-time event fetches anything, re-run the tail [turn_t0, n_past) -- the request
    // plus every token generated so far -- at the same positions with the band as it now is,
    // overwriting those tokens' K/V in the trunk and in the host store, and continue from the last
    // token's re-encoded logits.  Once per eval() that fetched.  turn_t0 = the pre-encode hand-off's
    // position (the request start); without a pre-encode hand-off, the generation start.
    // evr_max caps the tail (the start moves up; counted in n_reenc_clamp).  Default OFF until
    // validated (T26 + the events-only twin rows).
    bool    evreenc  = true;    // 09-24 SHIPPING DEFAULT 1 (Mark), with evr_budget 2048
    int32_t evr_max  = 1024;
    // 09-24 evr_budget (coordinator): tokens re-encoded per generated TURN, summed over its events.
    // A re-encode that would exceed what is left is not run: that event falls back to the strict
    // path (the unit stays visible to later queries only) and n_reenc_budget_skip counts it.
    // Reset at every turn start (pre_handoff / regather).  0 = unlimited.
    int32_t evr_budget = 2048;
    // 09-24 evr_surp (coordinator): re-encode at a decode-time event only when the triggering
    // generated word's summed surprisal is >= evr_surp nats (15 = the index's admission bar,
    // trig_surp).  Below it the fetch happens but the strict path applies; n_reenc_surp_skip
    // counts it.  0 = off (every fetching event re-encodes).  Default 0 pending the measurement.
    double  evr_surp = 15.0;   // 09-24 FINAL (v2bld23): 15 (Mark, 2026-09-24; LOG 135d: no read lost, decode +5% to +22% where events fire); evrsurp=0 = every fetching event re-encodes
    // reqfetch=0: the request fetches nothing -- the pre-encode hand-off runs with no request words
    // in its query, and events fire only in decode (not on the request's own words while it is
    // encoded).  The far unit can then arrive only through a GENERATED word: the events-only
    // measurement.  Requires prehand=1 (the request is then not yet in the stream at the gather).
    bool    reqfetch = true;
    // 09-24 unit=win<K> (coordinator, RESULTS Q7.5.4): the far unit is a content-agnostic window
    // of +-K tokens around each posted occurrence of a query term (one posting per occurrence,
    // the last post_cap per term), clipped to [n_pin+1, far_hi), merged when windows overlap
    // (next start < current end) across all of the gather's terms, scored by the summed
    // surprisal of the DISTINCT terms it holds (R4), admitted whole in score order (ties: later
    // end, then later start) until the budget, the last one cut to S.  Events fetch the windows
    // of the word's last evt_cap occurrences.  0 = off (the sentence / chunk unit rules).
    int32_t win_k   = 0;     // 09-25 v2bld26 DEFAULT 0 = the unit rules (sentence units; LOG 137, the U verdict 136p); was 256 (09-24 unit=win256); unit=win<K> = windows
    // 09-24 per-source window: terms that enter the query ONLY through the trunk seed get
    // windows of +-win_k_seed (0 = the same as win_k); request and event terms keep +-win_k.
    int32_t win_k_seed = 0;
    // 09-24 reqwin (diagnostic, default 1 = the shipped behavior): under the pre-encode hand-off
    // the "request window" select() reads from the stream (words of the last 2B tokens) is the
    // tail of the HISTORY, because the request is not in the stream yet; the request's own words
    // arrive through `up`.  reqwin=0 drops that window when `up` is given (the harness's
    // question-block query), leaving those words to the seed rule if seed_span covers them.
    bool    reqwin  = false;   // 09-24 DEFAULT 0 (coordinator, pending Mark): a defect fix -- the pre-encode query is the request, not the history tail
    // 09-24 chunkcut (default 1 = the old path): under fixed chunks, close the open word at every
    // chunk boundary.  That cuts words in two at arbitrary points (a PIPELINE F4 violation);
    // chunkcut=0 fixes it.  unit=win<K> never cuts (no legacy rows).
    bool    chunk_cut = false;   // 09-24 DEFAULT 0 (coordinator, pending Mark): the chunk rule no longer cuts words (F4)
    bool    win_log = false;   // record every closed word and the last gather (T27 dump)
    // 09-24 FLOOR (measurement arm, never a default; `floor=1`): the window-only floor -- the
    // trunk (last W tokens at their true positions) and the pinned prefix only, in prefill AND in
    // decode.  No transition, no band, no hand-off gather, no events, no re-encode.  Prefill is
    // byte-for-byte the design's window-only prefill; what differs is that the band is never
    // built, so a row measures what the trunk alone reads.  Below P_on the context stays dense
    // (t12fix), as in the design.
    bool    floor_only = false;
    // 09-24 v2bld24-perf (PERF gate fixes 1 and 2; the slates run on v2bld23, which has none of this).
    // feed: prefill below range in batches of `feed` tokens (whole blocks, one llama_decode each) instead
    //       of one 128-token block per llama_decode; the per-block bookkeeping is unchanged and in order.
    //       Active only where block_update() cannot move the cache (every block start < P_on), in
    //       prefill, with prerope 0 and no clusters.  feed <= B (e.g. feed=128) = the v2bld23 path.
    // capasync: in that batched region, a batch's K/V go to the host while the NEXT batch computes
    //       (the copy runs on cudaStreamPerThread, the compute on the backend's stream; the batch
    //       being copied is complete, the one computing writes other cells).  Same bytes (T28).
    // kvcap: keep the per-layer Q/K/V (and pre-RoPE K) graph outputs even when nothing reads them.
    //       v2bld23 always kept them under --inf (4 ggml_cont per layer per ubatch and per decode
    //       token); v2bld24-perf keeps them only for prerope, clusters, or the QK diagnostics (T29).
    int32_t feed     = 1024;   // 09-25 v2bld32 DEFAULT 1024 = the performance path is the shipping path (LOG 137m); feed=128 = the v2bld26-31 slate path
    bool    capasync = true;
    bool    kvcap    = false;  // 09-25 v2bld32 DEFAULT 0 (the performance path; exact by T29); kvcap=1 = the v2bld26-31 slate path
    // 09-24 v2bld25 per-source units (coordinator): seedsent=1 -> terms that enter the query ONLY through
    // the trunk seed fetch SENTENCE units (the unit index is built by the sentence rule: seedsent sets
    // unit_stride 0, as `sentence=1` does), while request and event terms keep +-win_k windows; windows and
    // seed units are admitted from one list in score order.  Requires win_k > 0.  Default off.
    bool    seed_sent = false;
    bool    seed_sent_user = false;
    // 09-25 v2bld32 (Mark, LOG 137m) feedwin: ABOVE the window-only switch point P_on the prefill is fed in batches of
    // `feed` tokens too, with the trunk rolled at each batch START (= after the previous batch): a query then sees
    // between W and W+feed-1 recent tokens (the trunk width W is a LOWER bound).  Capture unchanged.  feedwin=0 = v2bld31.
    bool    feed_win = true;
    // 09-25 v2bld32 (Mark's question, LOG 137m) store_budget: when the host store holds more than this many tokens,
    // evict whole FAR units by usage at exit: never-used units oldest first, then used units by fewest admissions,
    // oldest first; evicted tokens' host pages are released (madvise) and no read path can fetch them.  0 = unbounded.
    int64_t store_budget = 0;   // 09-25 v2bld31: seedsent was given explicitly (it then wins over unit=win<K>'s default)
    // 09-25 v2bld28 sentfill (coordinator, LOG 137f): the band FILLED to its budget by score under the unit rules.
    // sentfill=N: the +-1..N neighbouring units of every hit unit are secondary candidates at the hit's score x sfd1
    // (+-1) and x sfd2 (+-2); admission keeps going in score order to the budget, then fill-keep as before.  Run with
    // postcap=64.  0 = off (default).
    int32_t sent_fill = 0;
    double  sf_d1 = 0.5, sf_d2 = 0.25;
                                  // 1 admits roughly one posting list, i.e. an event's breadth,
                                  // which is how the seed's TIMING is separated from its BREADTH.

    // EVENT TIMING WITH SEED BREADTH: on the FIRST event of a generation, fire the full seed
    // gather - the trunk's rare words at the seed discount, the same ranking and the same cap as
    // the hand-off seed - instead of only the generated word's postings.  With the hand-off itself
    // unseeded this is the fourth corner of the crossing: the seed's breadth at the event's timing.
    int32_t evt_seed_span = 0;    // 0 = off; otherwise the trunk span to seed from, at the event
    int32_t evt_halo  = 0;        // far tokens fetched either side of a pulled unit.  32 makes a
                                  // ~20-token unit arrive as a ~96-token span, i.e. the v2 chunk
                                  // behaviour, without changing how units are cut.

    // --- window-only prefill.  Prefill runs the TRUNK ALONE (sliding window + the sink at its
    //     slot), capturing every rolled-off token's K/V to the host and building the word counts,
    //     units and index, but never gathering into a band.  The band is built once at the
    //     prompt->decode hand-off from the question block's rare words, and thereafter only by
    //     rare-word events during decode.
    bool    pf_gather = false;    // false: no band at all during prefill (09-24 shipping: false = nopfgather)

    // With the block gather off, the band is written ONLY by the hand-off gather and by events,
    // and nothing moves between events.  Its cells then drift away from the query at one position
    // per token, so their RoPE placement is refreshed only when the drift would push the band's
    // far edge past the model's trained range -- about every 8K tokens on an 8K trunk and a 16K
    // band.  Inside the range the drift is free.
    int32_t shift_every = 8192;   // tokens of drift tolerated before the band is re-placed

    // --- the harness's two admission rules (quality/scripts/inf_data.py: THETA, LEX_CAP_POST).
    //     A word is indexed, and used as a query term, only when its summed surprisal over its
    //     pieces exceeds trig_surp nats; and only the LAST post_cap postings per term are kept.
    //     Both apply to WHOLE WORDS, not pieces: a 15-nat threshold on pieces would throw away
    //     the needle's own keys, since no single piece of "daring-quiver" is that surprising.
    bool    trig_word = true;     // trigger on the word index instead of the token-id index (09-24 shipping: true)
    double  trig_surp = 15.0;     // nats, on the running unigram model in inf_freq
    int32_t post_cap  = 16;       // postings kept per term (0 = unlimited)

    // How the band is topped up to S when the index asks for fewer.  Recency (the original) makes
    // the unclaimed part of the band an extended window, and re-asserts the newest far tokens
    // every gather, so a unit that was triggered once and is not triggered again is unprotected
    // and the ring eventually walks over it.  KEEP instead tops up with what the band already
    // holds, so a triggered unit stays until the index itself displaces it.
    bool    fill_keep = true;     // 09-24 shipping default (fillkeep); was false
    int32_t nthread = 8;
    int32_t debug   = 0;
};

// the largest contiguous run of cells written in one host->device call
#define INF_WRITE_MAX 2048

// where a given token range ended up at the last gather (needle diagnosis)
struct inf_probe {
    int32_t n_tok = 0;       // tokens in the range
    int32_t n_far = 0;       // ... that were in the far region (not trunk, not pinned)
    int32_t n_sel = 0;       // ... that the selection asked for
    int32_t n_band = 0;      // ... that were actually resident in the band
    int32_t n_forced = 0;    // ... that came in via a trigger or the rolled-off tail
    int32_t votes_max = 0;   // best cluster vote count among the non-forced ones
    int32_t rank_min = -1, rank_max = -1;   // slot ranks of the resident ones
    int32_t dist_min = -1, dist_max = -1;   // and the distances those ranks imply
    int32_t unit = -1, unit_t0 = 0, unit_t1 = 0;
    float   unit_score = 0.0f;              // its summed query-token surprisal at the last gather
    int32_t unit_rank = -1;                 // its position in the trigger ranking
    int32_t unit_hits = 0;                  // how many units the triggers hit at all
    int32_t unit_taken = 0;                 // how many the trigger budget could afford
    int32_t band_oldest = -1;               // oldest token the band held
    int32_t p_place = 0, far_hi = 0;
    // --- splice: what sits NEXT TO the needle in slot space (rank order is what the model reads
    //     as adjacent positions).  A unit placed as one contiguous run with its own neighbours is
    //     read as continuous text; a unit whose slots are interleaved with unrelated sentences is
    //     spliced.
    int32_t adj_n    = 0;    // slot-adjacent cells examined (the two just outside [rank_min,rank_max])
    int32_t adj_same = 0;    // ... holding a token from the needle unit itself
    int32_t adj_near = 0;    // ... holding a token within 96 of the needle unit
    int32_t span_n   = 0;    // ranks in [rank_min, rank_max]
    int32_t span_own = 0;    // ... holding a token of the needle unit (span_own == span_n: contiguous)
    int32_t disc     = 0;    // DO NOT USE: rank runs newest-first while token index runs
                             // oldest-first, so this flags every correctly placed pair (LOG 20)
    // --- where else the needle's content could be read from, when its own tokens are not resident
    int32_t unit_n    = 0;   // tokens in the needle's unit
    int32_t unit_band = 0;   // ... resident in the band
    int32_t ctx_n     = 0;   // far tokens within 96 either side of the needle
    int32_t ctx_band  = 0;   // ... resident in the band
    int32_t trunk_n   = 0;   // the needle's tokens that are inside the dense trunk
};

// one unit (sentence)
struct inf_unit {
    int32_t t0 = 0, t1 = 0;    // [t0, t1)
};

// ---------------------------------------------------------------------------------------------
struct inf_ctx {
    llama_context     * lctx  = nullptr;
    const llama_model * model = nullptr;
    const llama_vocab * vocab = nullptr;
    llama_memory_t      mem   = nullptr;
    llama_seq_id        seq   = 0;
    inf_params          p;

    // model geometry
    int32_t n_layer = 0, n_head = 0, n_head_kv = 0, head_dim = 0;
    int32_t n_embd_k_gqa = 0, n_lh = 0, gqa = 0;
    size_t  row = 0;                 // bytes per cell per layer
    int32_t n_ctx = 0;
    int32_t P = 30;                  // feature dims = 2 * n_planes
    int32_t P2 = 465;                // packed upper triangle of Sigma
    int32_t plane0 = 0;              // first slow plane
    std::vector<float> theta;        // per-plane RoPE angular frequency
    int     rope_neox = 0;
    float   kq_scale = 1.0f;

    // stream state
    std::vector<llama_token> toks;
    std::vector<std::string> piece;  // detokenised piece text, by token id
    inf_freq freq;
    int32_t  n_past = 0;
    int32_t  rare_rank = 0;          // ids with rank >= rare_rank count as rare
    std::unordered_set<std::string> stopw;
    std::vector<uint8_t> tok_stop;   // per token id: is a stop word

    inf_hostkv host;

    // units + inverted index
    std::vector<inf_unit> units;
    int32_t unit_open = 0;                                  // first token of the open unit
    std::unordered_map<llama_token, std::vector<int32_t>> inv;

    // whole-word index: the event path keys on words, so "cactus" is one entry whether the
    // tokeniser splits it into one piece or four.  wpend holds the words of the unit still open.
    std::unordered_map<uint64_t, std::vector<int32_t>> winv;   // word hash -> units
    std::vector<uint64_t> wpend;        // word hashes seen since the open unit began
    std::vector<int32_t>  wpend_t0;     // 09-24 (wordall): first token of each pending word
    // 09-24 unit=win<K> (RESULTS Q7.5.4): one posting per OCCURRENCE -- the occurrence's first
    // token -- the last post_cap per term, trunk occurrences included (they count toward the cap
    // and are dropped at the gather).  The window [ws-K, ws+K) is built at the gather.
    std::unordered_map<uint64_t, std::vector<int32_t>> wocc;
    struct inf_woccl { uint64_t h; int32_t ws; double surp; std::string w; };
    std::vector<inf_woccl> wocc_log;    // every closed word (win_log only: the T27 dump)
    std::unordered_map<uint64_t, std::string> wtext;   // 09-24 diagnostic (win_log): hash -> word
    struct inf_win { int32_t lo, hi; double score; std::vector<uint64_t> terms; };
    std::vector<inf_win> last_win;      // merged windows of the last win-mode gather, position order
    std::vector<int32_t> last_win_order;   // their admission order
    int32_t              last_win_taken = 0;
    std::vector<std::pair<uint64_t,float>> last_q;   // the last gather's query terms (win_log only)
    std::vector<int32_t> last_sel_trig;  // sel after step 4, before any fill (win_log only)
    int32_t              last_hi = 0;    // far_hi of the last select()
    std::string  wcur;                  // the word being assembled
    // the words of the last two blocks, for the query side of the trigger: (hash, surprisal, end)
    struct inf_qword { uint64_t h; float surp; int32_t t; };
    std::vector<inf_qword> wlog;
    double       wsurp = 0.0;           // its summed -log p so far
    int32_t      wt0   = 0;             // its first token index
    std::vector<int32_t> u_stamp;
    std::vector<float>   u_score;

    // sentinels, per (layer, KV head) lh = il*n_head_kv + h
    std::vector<std::vector<float>>    clu_mu;      // n_clu_tot * P
    std::vector<std::vector<uint16_t>> clu_sig;     // n_clu_tot * P2, fp16
    std::vector<std::vector<float>>    clu_logn;    // n_clu_tot
    std::vector<std::vector<int32_t>>  clu_mem;     // member token indices, chunk-major
    std::vector<std::vector<uint32_t>> clu_off;     // n_clu_tot + 1 offsets into clu_mem
    int32_t n_chunk_done = 0;

    // mean query of the last block, de-rotated and scaled: [lh][P]
    std::vector<float> qmean;
    bool  have_q = false;

    // band
    std::vector<int32_t> band_tok;    // S entries: token index or -1
    // provenance of each live cell, so the band's COMPOSITION is reportable: 1 = arrived as an
    // index hit, 0 = arrived as recency fill.  Kept cells retain whatever they arrived as.
    std::vector<uint8_t> band_src;
    std::vector<int32_t> band_shifts;  // times this cell's slot position has been changed
    std::unordered_map<int32_t,int32_t> band_of;
    int32_t ring = 0;
    uint32_t band_c0 = 0;             // first band cell index in the KV cache
    bool     two_tier = false;        // the transition has happened
    int32_t  p_place  = 0;            // query position of the last band placement
    int32_t  win_edge = 0;            // window-only prefill: trunk edge already evicted
    bool     evt_seed_done = false;   // the event-time seed gather fires once per generation
    int32_t  p_shift  = 0;            // position of the last RoPE re-placement of the band

    // pinned prefix (tokens 0 .. n_pin), current llama positions
    std::vector<llama_pos> pin_pos;

    // scratch
    std::vector<int32_t>  stamp, votes, smark;
    std::unordered_set<int32_t> smark_hit;   // tokens THIS gather's index asked for
    int32_t               stamp_cur = 0;
    std::vector<int32_t>  cand, forced, sel;
    std::vector<llama_pos>   chk_pos;      // cache state read back after a placement
    std::vector<llama_token> chk_tok;
    std::vector<int32_t>     wrote_cell, wrote_tok;   // cells written by the current gather
    std::vector<int32_t>  last_rank;        // slot rank per band cell, from the last placement
    std::vector<int32_t>  last_hit;         // units the triggers hit, best first
    int32_t               last_hit_taken = 0;
    uint8_t * pin_k = nullptr, * pin_v = nullptr;   // pinned host staging
    ggml_backend_buffer_t pin_kbuf = nullptr, pin_vbuf = nullptr;
    size_t pin_cap = 0;
    std::vector<llama_pos> src_pos, dst_pos;
    std::vector<uint32_t> cell_idx, cellbuf;
    std::vector<llama_pos> map_buf;
    std::vector<float>    bufQ, bufK, bufKV, stageF, qq;
    std::vector<uint16_t> f16buf;
    std::vector<int32_t>  topidx;

    // counters
    int64_t n_blocks = 0, n_gather = 0;
    int64_t n_new_cells = 0, n_kept_cells = 0, n_band_live = 0, n_band_sel = 0, n_sel_real = 0, n_runs = 0;
    int64_t bytes_h2d = 0, bytes_d2h = 0;
    int64_t bytes_h2d_decode = 0;
    int64_t n_trig_units = 0, n_trig_tok = 0, n_clu_tok = 0;
    int64_t n_events = 0, n_event_cells = 0, n_event_runs = 0, n_event_units = 0, n_event_empty = 0;
    // a four-way classification of every event ATTEMPT.  Without this a hash miss is invisible:
    // word_flush returned silently when winv had no entry, so "fired but found nothing" only ever
    // counted the event_gather early returns and never the lookup failures.
    int64_t n_evt_try = 0, n_evt_nohash = 0, n_evt_trunk = 0, n_evt_resident = 0, n_evt_fetch = 0;
    int64_t n_seed_words = 0, n_req_words = 0, n_seed_gathers = 0, n_evt_seed_gathers = 0;
    int64_t n_prehand = 0;             // hand-offs run before the request was encoded
    uint64_t oracle_h = 0;             // the handle's word hash, when oracle_seed is on
    int64_t n_parked  = 0;             // band cells parked by evictafter
    // band composition, accumulated per gather: current hits / kept hits / recency fill
    int64_t n_comp_cur = 0, n_comp_kept = 0, n_comp_fill = 0, n_comp_gathers = 0;
    int64_t n_shift_max = 0;           // most re-placements any live cell has seen
    int64_t n_events_decode = 0, n_event_cells_decode = 0;
    int64_t n_trig_want  = 0;   // trigger tokens asked for before the budget cut: reach
                                // coverage is n_trig_tok / n_trig_want

    int64_t n_decode_tok = 0;
    double  t_capture = 0, t_cluster = 0, t_score = 0, t_gather = 0, t_place = 0, t_decode = 0;
    double  t_score_blocks = 0;
    int64_t n_cluster_chunks = 0;
    // band-vs-cache verification
    int64_t n_chk = 0, n_chk_pos_bad = 0, n_chk_tok_bad = 0, n_chk_byte_bad = 0, n_chk_cells = 0;
    int64_t n_bandof_fixed = 0;
    bool    in_decode = false;
    bool    pre_done  = false;   // the prehand gather has run for this turn
    int32_t pre_hold  = 0;       // tokens over which eval() must not gather again (the request)
    int32_t pf_from   = 2147483647;   // first position at which the prefill gathers (pf_depth)
    size_t  pf_units  = 0;            // units closed as of the last prefill gather (pf_blk < 0)
    // 09-24 evreenc: the turn's first position (the request start under prehand, else the
    // generation start), the pending flag set by a decode-time fetch, and the cost counters
    int32_t turn_t0    = -1;
    bool    reenc_want = false;
    int64_t n_reenc = 0, n_reenc_tok = 0, n_reenc_clamp = 0, n_reenc_fail = 0;
    int64_t n_reenc_budget_skip = 0;   // 09-24 events that fell back to the strict path (budget spent)
    int64_t n_reenc_surp_skip = 0;     // 09-24 fetching events below evr_surp (strict path kept)
    int64_t reenc_turn_tok = 0;        // tokens re-encoded in the current turn
    double  t_reenc = 0;
    // re-run [max(turn_t0, n_past - evr_max), n_past) with the band as it is now; public so the
    // selftest's T26 can drive it directly
    bool    reencode_tail();

    void  init(llama_context * c, const inf_params & pr);
    bool  eval(const llama_token * tok, int32_t n, bool logits_last);
    inf_probe probe(int32_t t0, int32_t t1) const;
    // QK-level diagnostic: at the token just decoded, what share of the FAR softmax mass does the
    // token range [t0,t1) actually receive, per layer and KV head?  Q comes from the graph through
    // llama_get_qkv and K from the cache itself, so both are in the frame attention sees -- no
    // de-roping, and the number is what the model computed rather than a reconstruction of it.
    void qk_mass(int32_t t0, int32_t t1, int32_t d0, int32_t d1, const char * tag, int32_t qtok = 0);
    void  verify_band(int32_t p_now);
    void  wtest(int32_t c0, int32_t m);  // byte-level cache read/write round trip
    void  rewrite_all(int32_t shift, bool do_clear = true, bool do_data = true);   // selftest: rebuild the whole cache from the host store
    void  regather();                 // force a gather at the current position (prompt -> decode)
    // the prehand hand-off: gather with the request's own tokens as the query, before they are fed
    void  pre_handoff(const llama_token * up, int32_t n_up);
    // park every live band cell: remove it from the sequence, leaving the trunk and the pins
    void  park_band();

    // event path: feed one token's piece into the word assembler; when a word completes and is
    // rare, pull its posting-list units into the band at once.
    void  word_feed(int32_t t, llama_token id);
    void  word_flush(int32_t t_end);                  // close the word that ends before t_end
    void  event_gather(const std::vector<int32_t> & us, int32_t t_now);
    // 09-24: the span form (t0, t1, anchor): a span is "in the trunk" when anchor > far_hi.
    // Units pass (t0-halo, t1+halo, t1); win<K> windows pass (ws-K, ws+K, ws+1).
    void  event_gather_spans(const std::vector<std::array<int32_t,3>> & sp, int32_t t_now);
    void  rank_only(int32_t p_now);                   // re-place the band without selecting
    void  window_update(int32_t p_now);               // trunk-only maintenance (no band)
    void  report(const char * tag) const;
    size_t host_bytes() const;
    int32_t band_live() const;

    // the known-answer tests (PIPELINE.md T13-T15) drive selection directly with a synthetic
    // query, so selection is reachable without going through a decode.
    void select(int32_t p, const llama_token * up, int32_t n_up);

    // 09-24 T25: the pinned prefix's cache K against the host copy rotated ONCE (double precision)
    // to each pin's current slot.  Returns false if the pins are not all resident.
    bool pin_host_cos(double & cmin, double & cmean, double & smin, double & dmax, int64_t & nrows);
    // 09-23 pin fix counters (public since 09-24: the selftest's T25 reports them)
    int64_t n_pin_moves = 0, n_pin_host = 0, n_pin_host_fallback = 0;

    // 09-24 band check (bandchk=1, or called directly by the selftest's T24).  Counters are
    // cumulative over the inf_ctx (one per needle example).  A band cell is "meta" bad if the
    // cache's provenance differs from band_tok, "slot" bad if its position is not the slot its
    // rank gives, "dup" if its provenance sits in two cells, "V" bad if its V row differs from
    // the host V of its provenance in any layer, "K" bad if cos(K, host K rotated to the slot)
    // < 0.999 in any layer.  Trunk cells: provenance = the true position.
    void    band_check(const char * where, int32_t p_now, bool trunk_only = false);   // trunk_only: also below L (single tier)
    // 09-24 PERF capblk: the decode tokens whose V is still in the cache only, [cap_lo, cap_hi)
    int32_t cap_lo = 0, cap_hi = 0;
    int64_t n_cap_flush = 0, n_cap_defer = 0;
    void    flush_capture();
    // 09-24 v2bld24-perf: batched feeding + asynchronous capture
    int32_t cap_pend_p0 = 0, cap_pend_n = 0;         // a batch whose K/V are not on the host yet
    int64_t n_fast_batch = 0, n_fast_tok = 0, n_cap_async = 0;
    int64_t n_seedu_units = 0, n_seedu_tok = 0, n_seedu_cand = 0;
    int64_t n_sf_cand = 0;
    // 09-25 v2bld32 feedwin + store budget
    bool    fast_batch_active = false;       // inside eval_fast()'s per-block loop: block_update() must not roll the trunk
    int64_t n_fast_above = 0;                // batched tokens above P_on
    std::vector<int32_t> u_use;              // admissions to the band per unit (usage at exit)
    std::vector<uint8_t> u_evicted, tok_evicted;
    int64_t n_evict_rounds = 0, n_evict_units = 0, n_evict_tok = 0, n_evict_units_unused = 0, n_evict_blocked = 0;
    bool    is_evicted(int32_t t) const { return t >= 0 && t < (int32_t) tok_evicted.size() && tok_evicted[t]; }
    void    note_use_tokens(const std::vector<int32_t> & ts);   // one admission per unit touched
    void    store_evict(int32_t p_now);   // 09-25 v2bld28 sentfill: neighbour units added as candidates   // 09-24 v2bld25 seedsent: seed units admitted / their new tokens / hit
    int32_t feed_cap() const;
    int32_t fast_take(int32_t rem) const;
    bool    eval_fast(llama_batch & batch, const llama_token * tok, int32_t m, bool want_logits);
    void    capture_nosync(int32_t p0, int32_t n);
    void    cap_pending_flush();
    void    feed_stream(const llama_token * tk, int32_t take);
    void    post_stream(int32_t take);
    int64_t n_bchk = 0, n_bchk_live = 0, n_bchk_meta = 0, n_bchk_slot = 0, n_bchk_dup = 0,
            n_bchk_v = 0, n_bchk_k = 0, n_bchk_tr_live = 0, n_bchk_tr_v = 0, n_bchk_tr_k = 0;
    double  bchk_kmin = 2.0, bchk_tr_kmin = 2.0;
    std::vector<int32_t> bchk_bad_t;   // provenance of every bad band/trunk cell (capped at 1M)

private:
    void block_update(int32_t p, const llama_token * up, int32_t n_up);
    void transition(int32_t p);
    void capture(int32_t p0, int32_t n, bool want_q);
    void close_chunks();
    void cluster_chunk(int32_t c);
    void place_band(int32_t p);
    void derope(float * x, int32_t pos, int32_t n_vec, int32_t stride) const;
    // rotate every plane of one f16 key row to pos: the single rotation of the prerope path
    void rope_row_f16(uint8_t * krow, int32_t pos) const;
    void place_rotated();             // contiguous band rewrite, each key rotated once
    void pin_move(const llama_pos * np); // 09-23 pin fix: remap, or (pinhost) host re-place
    std::vector<uint32_t> pin_cell;
};

bool inf_parse_args(inf_params & p, const std::string & key, const std::string & val);
