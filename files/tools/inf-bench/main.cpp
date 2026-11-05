// llama-infinite: driver for the two-tier far-context design ("infinite v1") and for the
// paper13 baselines it is measured against, so that the haystack, the needles and the timing
// harness are literally the same code for every design.
//
//   --mode bench   streaming prefill tok/s over the whole prompt + decode tok/s over N tokens
//   --mode needle  RULER-style single-needle retrieval over a Project Gutenberg haystack
//   --mode dump    feed a long text and print the policy's state
//
//   --inf                     the two-tier design (trunk + host far KV + index + band)
//   --inf-set k=v[,k=v...]    its knobs (W, S, B, pin, roll, chunk, nclu, topc, planes, ...)
//   (no --inf)                stock llama.cpp: dense attention over the whole context
//   (neither)                 dense attention

#include "infinite.h"
#include "../../vendor/nlohmann/json.hpp"   // 09-24 FINAL: --mode prompts (JSONL in/out)
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <cctype>
#include <random>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <string>
#include <unistd.h>
#include <vector>

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static std::vector<llama_token> tokenize(const llama_vocab * v, const std::string & s, bool bos) {
    int n = -llama_tokenize(v, s.c_str(), s.size(), nullptr, 0, bos, true);
    std::vector<llama_token> out(n);
    n = llama_tokenize(v, s.c_str(), s.size(), out.data(), out.size(), bos, true);
    out.resize(std::max(0, n));
    return out;
}

static std::string detok(const llama_vocab * v, const std::vector<llama_token> & t) {
    std::string out; char buf[256];
    for (auto id : t) { int n = llama_token_to_piece(v, id, buf, sizeof(buf), 0, true); if (n > 0) out.append(buf, n); }
    return out;
}

static ggml_type parse_type(const std::string & s) {
    if (s == "f32")  return GGML_TYPE_F32;
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "bf16") return GGML_TYPE_BF16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    return GGML_TYPE_F16;
}

// device memory actually in use, in bytes (total - free on the first GPU device)
static size_t gpu_used() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        size_t f = 0, t = 0;
        ggml_backend_dev_memory(d, &f, &t);
        return t - f;
    }
    return 0;
}

static size_t host_rss() {
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char ln[512]; size_t kb = 0;
    while (fgets(ln, sizeof(ln), f)) if (sscanf(ln, "VmRSS: %zu kB", &kb) == 1) break;
    fclose(f);
    return kb * 1024;
}

// ---------------------------------------------------------------------------
static std::string g_pause_file, g_log_file;

static void log_line(const std::string & s) {
    if (g_log_file.empty()) return;
    FILE * f = fopen(g_log_file.c_str(), "a");
    if (!f) return;
    fprintf(f, "%s\n", s.c_str());
    fclose(f);
}

static std::string stamp() {
    char b[64]; time_t t = time(nullptr); struct tm tmv; localtime_r(&t, &tmv);
    strftime(b, sizeof(b), "%Y-%m-%d %H:%M:%S", &tmv); return b;
}

static bool pause_gate(const char * what) {
    if (g_pause_file.empty()) return false;
    bool paused = false;
    while (access(g_pause_file.c_str(), F_OK) == 0) {
        if (!paused) {
            paused = true;
            const std::string s = "- paused " + stamp() + " (" + std::string(what) + "): " + g_pause_file + " present";
            printf("  [PAUSE] %s\n", s.c_str()); fflush(stdout); log_line(s);
        }
        sleep(30);
    }
    return paused;
}

// ---------------------------------------------------------------------------
// one engine: infinite, paper13 store, or dense
// ---------------------------------------------------------------------------
struct engine {
    bool inf = false;
    inf_params ip;

    llama_context * ctx = nullptr;
    inf_ctx * I = nullptr;
    int32_t n_dense = 0;              // n_past for the dense arm

    bool need_qkv = false;            // 09-24 v2bld24-perf: the QK diagnostics read Q from the graph
    void open(llama_model * model, llama_context_params cp) {
        // 09-24 v2bld24-perf: the per-layer Q/K/V (+ pre-RoPE K) graph outputs cost 4 ggml_cont per
        // layer per ubatch and per decode token; only prerope, clusters and the QK diagnostics read
        // them.  v2bld23 kept them on for every --inf run.  kvcap=1 restores that (T29: same logits).
        cp.kv_capture = inf && (ip.prerope || ip.use_clusters || ip.kvcap || need_qkv);
        ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "no ctx\n"); exit(1); }
        n_dense = 0;
        if (inf) { I = new inf_ctx(); I->init(ctx, ip); }
    }
    void close() {
        delete I; I = nullptr;
        if (ctx) llama_free(ctx);
        ctx = nullptr;
    }
    // Without --inf this is stock llama.cpp: a plain dense decode over the whole context.  There
    // is no third path; the paper13 forget-list store is not part of v2, not even as a baseline.
    // The batch split has to match the module's, or T12 compares two different GEMM schedules
    // rather than two policies: at B = 128 against 512 the logits differ by ~0.1 on a Q8_0 model
    // with nothing else changed.
    bool dense_eval(const llama_token * t, int32_t n, bool logits_last) {
        // 09-24 v2bld24-perf: fed in the module's batched-feeding split (feed, whole blocks), so T12 still
        // compares two policies on one GEMM schedule; feed <= B keeps the v2bld23 128-token split.
        const int32_t nbB = ip.B > 0 ? ip.B : 512;
        const int32_t nb  = ip.feed > nbB ? std::max(nbB, (std::min(ip.feed, 2048) / nbB) * nbB) : nbB;
        for (int32_t i = 0; i < n; i += nb) {
            const int32_t k = n - i < nb ? n - i : nb;
            llama_batch b = llama_batch_init(k, 0, 1);
            for (int32_t j = 0; j < k; ++j) {
                b.token[j]     = t[i + j];
                b.pos[j]       = n_dense + i + j;
                b.n_seq_id[j]  = 1;
                b.seq_id[j][0] = 0;
                b.logits[j]    = (logits_last && i + j == n - 1) ? 1 : 0;
            }
            b.n_tokens = k;
            const int rc = llama_decode(ctx, b);
            llama_batch_free(b);
            if (rc != 0) { fprintf(stderr, "dense decode failed: %d\n", rc); return false; }
        }
        n_dense += n;
        return true;
    }
    bool eval(const llama_token * t, int32_t n, bool logits_last) {
        return inf ? I->eval(t, n, logits_last) : dense_eval(t, n, logits_last);
    }
    void set_decode(bool d) { if (inf) { I->in_decode = d; if (d) I->regather(); } }

    // Feed a prompt whose last n_q tokens are the REQUEST.  Under prehand the hand-off gather runs
    // on the request's tokens before they are encoded, so the far units the question names are
    // resident while the question is being encoded; otherwise this is a plain eval.
    bool eval_turn(const llama_token * t, int32_t n, int32_t n_q, bool logits_last) {
        if (!inf || !I->p.prehand || n_q <= 0 || n_q >= n) return eval(t, n, logits_last);
        // tailq: hold back the last Q tokens of the HISTORY as well, and encode them with the
        // request, after the gather.  The query is then the tail's rare words plus the request's
        // plus the trunk seed, and the tail is encoded with the band visible, so whatever the
        // tail has to bind to a far unit can reach it and stays in the trunk.
        const int32_t Q = I->p.tailq > 0 ? I->p.tailq : 0;
        // pfdepth: the last N tokens of the HISTORY are prefilled with the machinery in place,
        // gathering every pf_blk tokens (or once per unit).  N = -1 asks for the derived W.
        int32_t N = I->p.pf_depth;
        if (N < 0) N = I->p.W;
        I->pf_from = N > 0 ? std::max(0, (n - n_q) - N) : 2147483647;
        int32_t cut = n - n_q - Q;
        if (cut < 1) cut = n - n_q;                 // not enough history: fall back to Q = 0
        if (!eval(t, cut, false)) return false;
        I->pre_handoff(t + cut, n - cut);
        return eval(t + cut, n - cut, logits_last);
    }
    int32_t n_past() const { return inf ? I->n_past : n_dense; }
};

// ---------------------------------------------------------------------------
struct opts {
    std::string model, mode = "bench", prompt = "Hello", books = "data/books";
    std::string rule = "simple:0.123", buckets, cache_k = "f16", cache_v = "f16";
    std::string inf_set, tag = "run", out_file, save, tasks = "niah1";
    std::string pause_file = "", log_file = "", freq_table, freq_cache;
    int  ngl = 99, n_predict = 256, n_ctx = 0, W = 0, S = 0, B = 128;
    int  nex = 16, seed = 0, dump_n = 8, threads = -1, repeats = 2;
    int  reach = 0, reach_batch = 1024, shift_every = 1;
    double reach_margin = 2.0, reach_gain = 0.35;
    int  kvshift = -1;
    int  qk_at = -1;   // call the QK diagnostic after this many generated tokens (-1 = off)
    int  qk_scan = 0;  // per-token QK scan over the question and the answer prefix, for N examples
    int  lossn = 0;    // tokens at the end of the prefill to score in --mode loss
    long pos_base = 0; int pos_n = 8192;   // --mode postest
    bool inf = false, fa = true, resume = false;
    int  T = 32768;
    int  bench_q = 0;  // 09-24: bench mode holds back the last N prompt tokens as the REQUEST (eval_turn)
    std::string t27_dump;  // 09-24: selftest writes the T27 (unit=win<K>) dump here
    int  ex0 = 0;          // 09-24 FINAL: first example index to RUN (items e < ex0 are generated, so the RNG stream is
                           // the same as one run over [0, nex), and skipped): 1M sections of <= 16 examples
    std::string prompts_file, preds_file;   // 09-24 FINAL: --mode prompts (generic prompt-in/answer-out, RULER)
};

static void usage(const char * a0) {
    printf("usage: %s -m MODEL [options]\n"
           "  --mode bench|needle|dump|gen|cont\n"
           "  --inf  --inf-set W=8192,S=16384,B=128,pin=128,roll=128,chunk=2816,nclu=64,topc=4,planes=15,\n"
           "                   qtok=16,shortlist=256,rare=0.02,kmiter=6,nthread=8,noclu=1,notrig=1,noroll=1,debug=1\n"
           "  -c N -T N -n N --repeats N --nex N --tasks L --seed N --books DIR --tag NAME --out FILE --resume\n"
           "  --ex0 N (first example run; items below are generated and skipped)   --mode prompts --prompts IN.jsonl --preds OUT.jsonl\n"
           "  --cache-type-k T --cache-type-v T\n"
           "\n"
           "DEFAULT CONFIGURATION (the configuration of the paper): a bare --inf is it.\n"
           "  units      SENTENCE units for every source (seed, request, events)             [default]\n"
           "  seed       seedspan=4096 seedwords=32 (rarest first) seedfar=1 (a seed word needs a far posting)\n"
           "  re-encode  evreenc=1 evrbudget=2048 evrsurp=15      other: prerope=0 reqwin=0 chunkcut=0 pinhost=1 capblk=1\n"
           "  --inf-set unit=win256 | unit=win512   WINDOWED units: +-256 / +-512 tokens around each posted occurrence of a\n"
           "                   request or event word; the SEED stays on sentence units unless seedsent=0 is also given\n"
           "                   (then the seed takes windows too).  For code or other volume-heavy use (RULER cwe/fwe).\n"
           "                   unit=win0 = sentence units (the default).\n"
           "  prefill    feed=1024 kvcap=0 (the batched performance path is the default; feed=128,kvcap=1 = one block at a time)\n"
           "  trunk      W=8192 is a LOWER bound: above the window-only switch point the prefill runs in 1,024-token batches and the\n"
           "             trunk rolls before each batch, so a query sees between W and W+1,023 recent tokens (feedwin=0 = exactly W)\n"
           "  --inf-set store_budget=N      bound the HOST store to about N tokens: whole far units are evicted by usage at exit\n"
           "                   (never-used oldest first; evicted units can no longer be fetched).  0 = unbounded (default).\n"
           "  --inf-set floor=1             window-only floor (measurement only: no band)\n", a0);
}

// identifies the binary that produced a results file: mtime and size of the running executable,
// which change on every relink whichever source file moved.
static std::string build_stamp() {
    char path[4096];
    const ssize_t n = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (n <= 0) return "unknown";
    path[n] = 0;
    struct stat st;
    if (stat(path, &st) != 0) return "unknown";
    char out[96];
    snprintf(out, sizeof(out), "%llx-%llx",
             (unsigned long long) st.st_mtime, (unsigned long long) st.st_size);
    return out;
}

int main(int argc, char ** argv) {
    opts o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if      (a == "-m")               o.model = next();
        else if (a == "--mode")           o.mode = next();
        else if (a == "-p")               o.prompt = next();
        else if (a == "--books")          o.books = next();
        else if (a == "--inf")            o.inf = true;
        else if (a == "--inf-set")        o.inf_set = next();
        else if (a == "--kv-block")       o.B = std::stoi(next());
        else if (a == "--kv-reach")       o.reach = std::stoi(next());
        else if (a == "--kv-shift-every") o.shift_every = std::stoi(next());
        else if (a == "--repeats")        o.repeats = std::stoi(next());
        else if (a == "--pause-file")     o.pause_file = next();
        else if (a == "--log-file")       o.log_file = next();
        else if (a == "--out")            o.out_file = next();
        else if (a == "--resume")         o.resume = true;
        else if (a == "--qk-scan")       o.qk_scan = std::stoi(next());
        else if (a == "--loss-n")        o.lossn = std::stoi(next());
        else if (a == "--qk-at")          o.qk_at = std::stoi(next());
        else if (a == "--pos-base")       o.pos_base = std::stol(next());
        else if (a == "--pos-n")          o.pos_n = std::stoi(next());
        else if (a == "--kv-freq-table")  o.freq_table = next();
        else if (a == "--kv-freq-cache")  o.freq_cache = next();
        else if (a == "-c")               o.n_ctx = std::stoi(next());
        else if (a == "-T")               o.T = std::stoi(next());
        else if (a == "-n")               o.n_predict = std::stoi(next());
        else if (a == "-ngl")             o.ngl = std::stoi(next());
        else if (a == "-t")               o.threads = std::stoi(next());
        else if (a == "--nex")            o.nex = std::stoi(next());
        else if (a == "--tasks")          o.tasks = next();
        else if (a == "--seed")           o.seed = std::stoi(next());
        else if (a == "--dump-n")         o.dump_n = std::stoi(next());
        else if (a == "--tag")            o.tag = next();
        else if (a == "--kvshift")        o.kvshift = std::stoi(next());
        else if (a == "--no-fa")          o.fa = false;
        else if (a == "--bench-q")        o.bench_q = std::stoi(next());   // 09-24
        else if (a == "--t27-dump")       o.t27_dump = next();              // 09-24 unit=win<K> T27
        else if (a == "--ex0")            o.ex0 = std::stoi(next());        // 09-24 FINAL
        else if (a == "--prompts")        o.prompts_file = next();          // 09-24 FINAL: --mode prompts input (JSONL)
        else if (a == "--preds")          o.preds_file = next();            // 09-24 FINAL: --mode prompts output (JSONL)
        else if (a == "--cache-type-k")   o.cache_k = next();
        else if (a == "--cache-type-v")   o.cache_v = next();
        else { usage(argv[0]); return 1; }
    }
    if (o.model.empty()) { usage(argv[0]); return 1; }
    g_pause_file = o.pause_file; g_log_file = o.log_file;
    if (o.out_file.empty()) o.out_file = "results/" + o.tag + ".txt";

    ggml_backend_load_all();
    llama_log_set([](ggml_log_level l, const char * t, void *) { if (l >= GGML_LOG_LEVEL_WARN) fputs(t, stderr); }, nullptr);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = o.ngl;
    llama_model * model = llama_model_load_from_file(o.model.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const size_t mem_model = gpu_used();

    engine E;
    E.inf = o.inf;
    E.need_qkv = (o.qk_at >= 0 || o.qk_scan > 0);   // 09-24 v2bld24-perf

    if (o.inf) {
        std::stringstream ss(o.inf_set); std::string kv;
        while (std::getline(ss, kv, ',')) {
            if (kv.empty()) continue;
            const size_t eq = kv.find('=');
            const std::string k = eq == std::string::npos ? kv : kv.substr(0, eq);
            const std::string v = eq == std::string::npos ? "1" : kv.substr(eq + 1);
            if (!inf_parse_args(E.ip, k, v)) { fprintf(stderr, "bad --inf-set key '%s'\n", k.c_str()); return 1; }
        }
        E.ip.B = o.B;
        if (o.threads > 0) E.ip.nthread = o.threads;
    } else {
    }

    const int n_ctx = o.n_ctx > 0 ? o.n_ctx
                    : (o.inf   ? E.ip.W + E.ip.S + E.ip.n_pin + 2 + 4*E.ip.B
                    : (o.T + 256));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = n_ctx;
    cp.n_batch  = std::max(std::max(o.B, 512), std::min(E.ip.feed, 2048));   // 09-24 v2bld24-perf: room for batched feeding
    cp.n_ubatch = std::max(o.B, 512);
    cp.type_k   = parse_type(o.cache_k);
    cp.type_v   = parse_type(o.cache_v);
    cp.no_perf  = true;
    if (o.threads > 0) { cp.n_threads = o.threads; cp.n_threads_batch = o.threads; }
    if (!o.fa) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    const char * design = o.inf ? "infinite" : "dense";
    printf("# model %s\n# mode %s  design %s  n_ctx %d  T %d\n", o.model.c_str(), o.mode.c_str(), design, n_ctx, o.T);
    if (o.inf) printf("# inf: W %d  S %d  B %d  pin %d  roll %d  chunk %d  nclu %d  topc %d  planes %d  qtok %d  shortlist %d  rare %.4f  clusters %d  triggers %d\n",
                      E.ip.W, E.ip.S, E.ip.B, E.ip.n_pin, E.ip.n_roll, E.ip.chunk, E.ip.n_clu, E.ip.topc,
                      E.ip.n_planes, E.ip.qtok, E.ip.shortlist, E.ip.rare_tail, (int) E.ip.use_clusters, (int) E.ip.use_triggers);
    if (o.inf) printf("# ship5: prerope %d  evreenc %d  evr_budget %d  unit %s  pinhost %d  capblk %d  |  reqwin %d  chunkcut %d  evr_surp %.1f  win_k_seed %d  seed_span %d\n",   // 09-24 the five shipping decisions (+ pending)
                      (int) E.ip.prerope, (int) E.ip.evreenc, E.ip.evr_budget,
                      E.ip.win_k > 0 ? ("win" + std::to_string(E.ip.win_k)).c_str() : (E.ip.unit_stride > 0 ? "chunk" : "sentence"),
                      E.ip.pinhost, (int) E.ip.capblk, (int) E.ip.reqwin, (int) E.ip.chunk_cut, E.ip.evr_surp,
                      E.ip.win_k_seed, E.ip.seed_span);
    if (o.inf) printf("# inf-flags: prerope %d  prehand %d  pf_gather %d  trig_word %d  evt %d  pinhost %d  unit_stride %d  wordall %d  t12fix %d  bandchk %d  capblk %d  capnok %d  evreenc %d  evr_max %d  evr_budget %d  reqfetch %d  seed_span %d  win_k %d  reqwin %d  chunkcut %d\n",   // 09-24
                      (int) E.ip.prerope, (int) E.ip.prehand, (int) E.ip.pf_gather, (int) E.ip.trig_word,
                      (int) E.ip.evt, E.ip.pinhost, E.ip.unit_stride, (int) E.ip.wordall, (int) E.ip.t12fix, (int) E.ip.bandchk, (int) E.ip.capblk, (int) E.ip.capnok,
                      (int) E.ip.evreenc, E.ip.evr_max, E.ip.evr_budget, (int) E.ip.reqfetch, E.ip.seed_span, E.ip.win_k, (int) E.ip.reqwin, (int) E.ip.chunk_cut);
    // 09-24 FINAL (v2bld23): the seed, evr_surp and the floor arm on one line of their own
    const std::string unit_s = !o.inf ? std::string("dense")
                             : (E.ip.win_k > 0 ? "win" + std::to_string(E.ip.win_k) : (E.ip.unit_stride > 0 ? std::string("chunk") : std::string("sentence")));
    if (o.inf) printf("# final: seed_span %d  seed_words %d  seed_disc %.2f  evr_surp %.1f  floor %d  unit %s  reqwin %d  chunkcut %d  seedsent %d  unit_stride %d  seedfar %d\n",
                      E.ip.seed_span, E.ip.seed_words, E.ip.seed_disc, E.ip.evr_surp, (int) E.ip.floor_only, unit_s.c_str(),
                      (int) E.ip.reqwin, (int) E.ip.chunk_cut, (int) E.ip.seed_sent, E.ip.unit_stride, (int) E.ip.seed_far);
    if (o.inf) printf("# prefill: feedwin %d (trunk width W = %d is a LOWER bound: a query sees W..W+%d recent tokens above P_on)  store_budget %lld\n",
                      (int) E.ip.feed_win, E.ip.W, std::max(0, std::min(E.ip.feed, 2048) - 1), (long long) E.ip.store_budget);
    if (o.inf && E.ip.sent_fill > 0) printf("# sentfill: neighbours +-1..%d  d1 %.2f  d2 %.2f  post_cap %d\n", E.ip.sent_fill, E.ip.sf_d1, E.ip.sf_d2, E.ip.post_cap);
    printf("# perf: feed %d  capasync %d  kvcap %d  n_batch %d  n_ubatch %d  (v2bld24-perf)\n", E.ip.feed, (int) E.ip.capasync,
           (int) E.ip.kvcap, (int) cp.n_batch, (int) cp.n_ubatch);
    printf("# cache k/v %s/%s  fa %d  model GPU %.2f GiB\n", o.cache_k.c_str(), o.cache_v.c_str(), (int) o.fa, mem_model/1073741824.0);

    // ------------------------------------------------------------- prompts (09-24 FINAL, v2bld23)
    // Generic prompt-in / answer-out driver (RULER, or any task).  Input: JSONL, one example per
    // line: {"index", "task", "history", "request", "outputs": [..], "max_new", "metric": "all"|"part"}.
    // HISTORY is encoded with BOS; REQUEST is the user's turn (the question plus the answer prefix).
    // The request boundary is the host's turn_begin() (PIPELINE R1): it comes from the file and is
    // never inferred from the text.  The design is not told the task: the path is exactly the
    // shipping one -- window-only prefill of the history, the pre-encode hand-off on the request's
    // tokens (eval_turn), then greedy decode with events and the event re-encode.  Without --inf
    // this is stock llama.cpp, dense over the whole context, on the same tokens.  Generation is
    // greedy, up to max_new tokens, stopping only at end-of-generation (RULER: temperature 0, no
    // stop words).  The score printed here is RULER's string match (postprocess_pred, then
    // string_match_all / string_match_part per example, case-folded ASCII); the harvest recomputes
    // it in Python with RULER's own metric functions from the preds file.  Lines [ex0, nex) of the
    // file are run (nex < 0 = to the end).
    if (o.mode == "prompts") {
        using njson = nlohmann::json;
        std::ifstream pfin(o.prompts_file);
        if (!pfin) { fprintf(stderr, "prompts: cannot read --prompts '%s'\n", o.prompts_file.c_str()); return 1; }
        FILE * fpred = o.preds_file.empty() ? nullptr : fopen(o.preds_file.c_str(), "a");
        if (!o.preds_file.empty() && !fpred) { fprintf(stderr, "prompts: cannot write --preds '%s'\n", o.preds_file.c_str()); return 1; }
        FILE * fres = fopen(o.out_file.c_str(), "a");
        if (fres) {
            fprintf(fres, "# build %s\n# applied %s  inf-set '%s'  prompts %s  ex [%d,%d)", build_stamp().c_str(), design,
                    o.inf_set.c_str(), o.prompts_file.c_str(), o.ex0, o.nex);
            if (o.inf) fprintf(fres, "  SHIP5 prerope %d evreenc %d evr_budget %d unit %s pinhost %d capblk %d  |  evr_surp %.1f  seed_span %d  seed_words %d  reqwin %d  chunkcut %d  floor %d  prehand %d  evt %d",
                               (int) E.ip.prerope, (int) E.ip.evreenc, E.ip.evr_budget, unit_s.c_str(), E.ip.pinhost, (int) E.ip.capblk,
                               E.ip.evr_surp, E.ip.seed_span, E.ip.seed_words, (int) E.ip.reqwin, (int) E.ip.chunk_cut,
                               (int) E.ip.floor_only, (int) E.ip.prehand, (int) E.ip.evt);
            fprintf(fres, "\n"); fflush(fres);
        }
        auto post = [](std::string s) {          // RULER postprocess_pred: strip, [\x00-\x1f] -> \n, strip
            auto strip = [](const std::string & x) {
                size_t a = 0, b = x.size();
                while (a < b && isspace((unsigned char) x[a])) a++;
                while (b > a && isspace((unsigned char) x[b-1])) b--;
                return x.substr(a, b - a); };
            s = strip(s);
            for (auto & c : s) if ((unsigned char) c < 0x20) c = '\n';
            return strip(s); };
        auto lower = [](std::string s) { for (auto & c : s) c = (char) tolower((unsigned char) c); return s; };
        std::map<std::string, std::pair<double, int>> agg;   // task -> (sum of per-example scores, n)
        std::vector<std::string> task_order;
        const int nv = llama_vocab_n_tokens(vocab);
        const double t_all0 = now_s();
        std::string ln;
        int li = -1, n_run = 0, n_err = 0;
        while (std::getline(pfin, ln)) {
            if (ln.empty()) continue;
            li++;
            if (li < o.ex0) continue;
            if (o.nex >= 0 && li >= o.nex) break;
            njson j;
            try { j = njson::parse(ln); } catch (const std::exception & ex) {
                fprintf(stderr, "prompts: line %d: bad JSON: %s\n", li, ex.what()); n_err++; continue; }
            const std::string task = j.value("task", std::string("task"));
            const int idx = j.value("index", li);
            const std::string hist = j.value("history", std::string());
            const std::string req  = j.value("request", std::string());
            const int max_new = j.value("max_new", 128);
            const std::string metric = j.value("metric", std::string("all"));
            std::vector<std::string> refs;
            if (j.contains("outputs")) for (auto & r : j["outputs"]) refs.push_back(r.get<std::string>());
            if (!agg.count(task)) task_order.push_back(task);

            std::vector<llama_token> prompt = tokenize(vocab, hist, true);
            const auto rq = tokenize(vocab, req, false);
            const int n_q = (int) rq.size();
            prompt.insert(prompt.end(), rq.begin(), rq.end());
            const int n_tok = (int) prompt.size();
            if (!o.inf && n_tok + max_new > n_ctx) {
                printf("  RULER %-16s ex %3d  SKIP  prompt %d + %d new > n_ctx %d (dense)\n", task.c_str(), idx, n_tok, max_new, n_ctx);
                fflush(stdout); n_err++; continue;
            }
            pause_gate("before prompt");
            const double t0 = now_s();
            E.open(model, cp);
            if (!E.eval_turn(prompt.data(), n_tok, n_q, true)) { E.close(); fprintf(stderr, "prompts: eval failed at line %d\n", li); return 1; }
            const double t1 = now_s();
            E.set_decode(true);
            std::string out;
            int n_gen = 0;
            for (int i = 0; i < max_new; ++i) {
                const float * lg = llama_get_logits_ith(E.ctx, -1);
                llama_token best = 0; float bv = -1e30f;
                for (int v = 0; v < nv; ++v) if (lg[v] > bv) { bv = lg[v]; best = v; }
                if (llama_vocab_is_eog(vocab, best)) break;
                out += detok(vocab, {best});
                n_gen++;
                if (!E.eval(&best, 1, true)) break;
            }
            const double t2 = now_s();
            const std::string pp = lower(post(out));
            int nhit = 0;
            for (auto & r : refs) if (pp.find(lower(r)) != std::string::npos) nhit++;
            double sc = 0.0;
            if (!refs.empty()) sc = metric == "part" ? (nhit > 0 ? 1.0 : 0.0) : (double) nhit / (double) refs.size();
            agg[task].first += sc; agg[task].second++;
            n_run++;
            std::string o1 = out; for (auto & c : o1) if (c == '\n' || c == '\r' || c == '\t') c = '/';
            if (o1.size() > 160) o1 = o1.substr(0, 160) + "...";
            std::string gl; for (size_t i = 0; i < refs.size(); ++i) gl += (i ? "|" : "") + refs[i];
            if (gl.size() > 160) gl = gl.substr(0, 160) + "...";
            const long long sel  = E.inf ? (long long) E.I->n_sel_real : 0;
            const long long nga  = E.inf ? (long long) E.I->n_gather : 0;
            const long long newc = E.inf ? (long long) E.I->n_new_cells : 0;
            char line[2048];
            snprintf(line, sizeof(line),
                     "  RULER %-16s ex %3d  score %.3f  hit %d/%zu  ntok %d  nq %d  gen %d  pred [%s]  gold [%s]  "
                     "(%.0fs, prefill %.1f tok/s, design %s, floor %d, RSS %.1f GiB)  VOL sel_tok %lld gathers %lld new_cells %lld  "
                     "REENC n %lld tok %lld ms %.1f evreenc %d budget %d skip %lld surpskip %lld evrsurp %.0f  EVT decode %lld fetched %lld\n",
                     task.c_str(), idx, sc, nhit, refs.size(), n_tok, n_q, n_gen, o1.c_str(), gl.c_str(),
                     t2 - t0, n_tok / std::max(1e-9, t1 - t0), design, o.inf ? (int) E.ip.floor_only : 0, host_rss()/1073741824.0,
                     sel, nga, newc,
                     (long long)(E.inf ? E.I->n_reenc : 0), (long long)(E.inf ? E.I->n_reenc_tok : 0),
                     E.inf ? 1000.0*E.I->t_reenc : 0.0, o.inf ? (int) E.ip.evreenc : -1, o.inf ? E.ip.evr_budget : -1,
                     (long long)(E.inf ? E.I->n_reenc_budget_skip : 0), (long long)(E.inf ? E.I->n_reenc_surp_skip : 0),
                     o.inf ? E.ip.evr_surp : -1.0,
                     (long long)(E.inf ? E.I->n_events_decode : 0), (long long)(E.inf ? E.I->n_evt_fetch : 0));
            printf("%s", line); fflush(stdout);
            if (fres) { fprintf(fres, "%s", line); fflush(fres); }
            if (fpred) {
                njson pj;
                pj["index"] = idx; pj["task"] = task; pj["pred"] = out; pj["outputs"] = refs; pj["metric"] = metric;
                pj["score_driver"] = sc; pj["ntok"] = n_tok; pj["nq"] = n_q; pj["gen"] = n_gen;
                pj["design"] = design; pj["unit"] = unit_s; pj["floor"] = o.inf ? (int) E.ip.floor_only : 0;
                pj["sel_tok"] = sel; pj["gathers"] = nga; pj["new_cells"] = newc;
                pj["reenc"] = E.inf ? (long long) E.I->n_reenc : 0; pj["secs"] = t2 - t0; pj["build"] = build_stamp();
                fprintf(fpred, "%s\n", pj.dump(-1, ' ', false, njson::error_handler_t::replace).c_str()); fflush(fpred);
            }
            E.close();
        }
        printf("\n# prompts: %d run, %d skipped/errors, %.0fs\n", n_run, n_err, now_s() - t_all0);
        for (const auto & t : task_order) {
            const auto & a = agg[t];
            printf("RULER\t%s\tdesign=%s\tT=%d\ttask=%s\tscore=%.2f\tn=%d\tunit=%s\tfloor=%d\tevr_surp=%.1f\tseed_span=%d\tseed_words=%d\n",
                   o.tag.c_str(), design, o.T, t.c_str(), a.second ? 100.0 * a.first / a.second : 0.0, a.second,
                   unit_s.c_str(), o.inf ? (int) E.ip.floor_only : 0, o.inf ? E.ip.evr_surp : -1.0,
                   o.inf ? E.ip.seed_span : -1, o.inf ? E.ip.seed_words : -1);
            if (fres) fprintf(fres, "#   %-16s score %.2f over %d\n", t.c_str(), a.second ? 100.0 * a.first / a.second : 0.0, a.second);
        }
        if (fres) fclose(fres);
        if (fpred) fclose(fpred);
        llama_model_free(model);
        return n_err ? 3 : 0;
    }

    // ------------------------------------------------------------- haystack
    std::string hay_txt;
    std::vector<llama_token> HAY;
    {
        const char * ids[] = {"2600", "135", "1184", "996", "145", "2701"};
        for (const char * b : ids) {
            std::ifstream f(o.books + "/" + b + ".txt");
            if (!f) { fprintf(stderr, "missing book %s/%s.txt\n", o.books.c_str(), b); return 1; }
            std::stringstream ss; ss << f.rdbuf(); std::string t = ss.str();
            size_t a = t.find("*** START OF"); a = a == std::string::npos ? 0 : t.find('\n', a) + 1;
            size_t e = t.find("*** END OF");
            hay_txt += t.substr(a, e == std::string::npos ? std::string::npos : e - a);
            hay_txt += "\n";
        }
        const double t0 = now_s();
        HAY = tokenize(vocab, hay_txt, false);
        printf("# haystack: %zu tokens from 6 distinct Gutenberg books (%.0fs), non-repeating\n", HAY.size(), now_s() - t0);
        if ((int) HAY.size() < o.T) { fprintf(stderr, "haystack too short: %zu < %d\n", HAY.size(), o.T); return 1; }
    }

    const int64_t n_layer = llama_model_n_layer(model);
    const int64_t n_hkv   = llama_model_n_head_kv(model);
    const int64_t hdim    = llama_model_n_embd(model) / llama_model_n_head(model);
    const int64_t kbytes  = (int64_t) ggml_type_size(cp.type_k);
    const int64_t vbytes  = (int64_t) ggml_type_size(cp.type_v);
    const int64_t kv_bytes = (int64_t) n_ctx * n_layer * n_hkv * hdim * (kbytes + vbytes);

    // ----------------------------------------------------------------- bench
    if (o.mode == "bench") {
        std::vector<llama_token> pr; pr.push_back(llama_vocab_bos(vocab));
        pr.insert(pr.end(), HAY.begin(), HAY.begin() + (size_t) o.T - 1);

        printf("# bench: prefill %zu tokens, decode %d tokens, %d repeats, KV cells %d, KV %.2f GiB\n",
               pr.size(), o.n_predict, o.repeats, n_ctx, kv_bytes/1073741824.0);
        std::vector<double> pf, dc, dcs, hos, tmx;   // 09-24: steady decode, hand-off, max token ms
        double gpu_peak = 0, host_peak = 0, h2d_dec = 0, ms_score = 0, ms_clu = 0, occ = 0, hostb = 0;
        double ms_cap = 0, ms_place = 0, ms_sel = 0, newc = 0;
        double rn_sum = 0, rt_sum = 0, rms_sum = 0, rsk_sum = 0, ev_sum = 0;   // 09-24 evreenc, summed over repeats
        int64_t fast_tok_last = 0, cap_async_last = 0;                            // 09-24 v2bld24-perf
        for (int r = 0; r < o.repeats; ++r) {
            pause_gate("before bench repeat");
            E.open(model, cp);
            if (E.inf) {
                int32_t N = E.I->p.pf_depth; if (N < 0) N = E.I->p.W;
                E.I->pf_from = N > 0 ? std::max(0, (int32_t) pr.size() - N) : 2147483647;
            }
            const double t0 = now_s();
            // 09-24: --bench-q N feeds the last N tokens as a request (pre-encode hand-off under
            // prehand), so an evreenc tail is request + generation as in the needle rows
            if (!(o.bench_q > 0 ? E.eval_turn(pr.data(), pr.size(), o.bench_q, true)
                                : E.eval(pr.data(), pr.size(), true))) { E.close(); return 1; }
            const double t1 = now_s();
            const size_t g_pre = gpu_used(), h_pre = host_rss();
            E.set_decode(true);
            const double t1h = now_s();   // 09-24: the hand-off (set_decode -> regather) timed apart
            int n_gen = 0;
            double tok_max = 0;
            const int nv = llama_vocab_n_tokens(vocab);
            for (int i = 0; i < o.n_predict; ++i) {
                const double ta = now_s();
                const float * lg = llama_get_logits_ith(E.ctx, -1);
                llama_token best = 0; float bv = -1e30f;
                for (int v = 0; v < nv; ++v) if (lg[v] > bv) { bv = lg[v]; best = v; }
                if (!E.eval(&best, 1, true)) break;
                n_gen++;
                tok_max = std::max(tok_max, now_s() - ta);
            }
            const double t2 = now_s();
            // 09-24: with bandchk=1, verify the host store of the whole trunk -- the decode tokens
            // included, i.e. the capblk path -- against the cache (outside the timed window)
            if (E.inf && E.I->p.bandchk) E.I->band_check("bench", E.I->n_past, true);
            E.set_decode(false);
            pf.push_back(pr.size()/(t1 - t0));
            dc.push_back(n_gen/(t2 - t1));
            dcs.push_back(n_gen/(t2 - t1h));
            hos.push_back(t1h - t1);
            tmx.push_back(1000.0*tok_max);
            printf("  rep %d  decode_steady %7.2f tok/s  handoff %.3fs  tok_ms_max %.2f\n", r, dcs.back(), hos.back(), tmx.back());
            if (E.inf)   // 09-24 evreenc cost, per repeat
                printf("  rep %d  REENC decode_events %lld fetched %lld  reencodes %lld  tokens %lld  ms %.1f  budget_skips %lld  surp_skips %lld  (evreenc %d, evr_budget %d, evr_surp %.0f)\n",
                       r, (long long) E.I->n_events_decode, (long long) E.I->n_evt_fetch, (long long) E.I->n_reenc,
                       (long long) E.I->n_reenc_tok, 1000.0*E.I->t_reenc, (long long) E.I->n_reenc_budget_skip,
                       (long long) E.I->n_reenc_surp_skip, (int) E.I->p.evreenc, E.I->p.evr_budget, E.I->p.evr_surp);
            if (E.inf) { rn_sum += (double) E.I->n_reenc; rt_sum += (double) E.I->n_reenc_tok; rms_sum += 1000.0*E.I->t_reenc;
                         rsk_sum += (double) E.I->n_reenc_budget_skip; ev_sum += (double) E.I->n_events_decode; }
            gpu_peak  = std::max<double>(gpu_peak, (double) std::max(g_pre, gpu_used()));
            host_peak = std::max<double>(host_peak, (double) std::max(h_pre, host_rss()));
            printf("  rep %d  prefill %8.1f tok/s (%7.1fs)  decode %7.2f tok/s (%6.2fs, %d tok)  GPU %.2f GiB  RSS %.2f GiB\n",
                   r, pf.back(), t1 - t0, dc.back(), t2 - t1, n_gen, gpu_peak/1073741824.0, host_peak/1073741824.0);
            if (E.inf) {
                inf_ctx & I = *E.I;
                I.report(o.tag.c_str());
                h2d_dec  = I.n_decode_tok ? (double) I.bytes_h2d_decode/I.n_decode_tok : 0.0;
                ms_score = I.t_score_blocks ? 1000.0*I.t_score/I.t_score_blocks : 0.0;
                ms_clu   = I.n_cluster_chunks ? 1000.0*I.t_cluster/I.n_cluster_chunks : 0.0;
                ms_cap   = I.n_blocks ? 1000.0*I.t_capture/I.n_blocks : 0.0;
                ms_place = I.n_gather ? 1000.0*I.t_place/I.n_gather : 0.0;
                ms_sel   = I.n_gather ? 1000.0*I.t_gather/I.n_gather : 0.0;
                occ      = I.n_gather ? (double) I.n_sel_real/I.n_gather/I.p.S : 0.0;
                newc     = I.n_gather ? (double) I.n_new_cells/I.n_gather : 0.0;
                hostb    = (double) I.host_bytes();
                fast_tok_last = I.n_fast_tok; cap_async_last = I.n_cap_async;   // 09-24 v2bld24-perf
            }
            fflush(stdout);
            E.close();
        }
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size()/2]; };
        printf("BENCH\t%s\tdesign=%s\tT=%d\tcells=%d\tkv_GiB=%.3f\tgpu_GiB=%.3f\thost_GiB=%.3f\thost_index_GiB=%.3f\t"
               "host_B_per_tok=%.1f\tprefill=%.1f\tdecode=%.2f\th2d_B_per_dec_tok=%.0f\tscore_ms_blk=%.2f\t"
               "cluster_ms_chunk=%.1f\tcapture_ms_blk=%.2f\tselect_ms_blk=%.2f\tplace_ms_blk=%.2f\tband_occ=%.3f\tnew_cells=%.0f\treps=%d\t"
               "decode_steady=%.2f\thandoff_s=%.3f\ttok_ms_max=%.2f\tevreenc=%d\tevr_budget=%d\treenc_per_rep=%.2f\t"
               "reenc_tok_per_rep=%.1f\treenc_ms_per_rep=%.1f\treenc_skips_per_rep=%.2f\tdecode_events_per_rep=%.2f\t"
               "prerope=%d\twin_k=%d\tpinhost=%d\tcapblk=%d\tbench_q=%d\tevr_surp=%.1f\treqwin=%d\tfeed=%d\tcapasync=%d\tkvcap=%d\tfast_tok=%lld\tcap_async=%lld\n",
               o.tag.c_str(), design, o.T, n_ctx, kv_bytes/1073741824.0, gpu_peak/1073741824.0,
               host_peak/1073741824.0, hostb/1073741824.0, o.T ? hostb/o.T : 0.0,
               med(pf), med(dc), h2d_dec, ms_score, ms_clu, ms_cap, ms_sel, ms_place, occ, newc, o.repeats,
               med(dcs), med(hos), med(tmx), o.inf ? (int) E.ip.evreenc : -1, o.inf ? E.ip.evr_budget : -1,
               rn_sum/o.repeats, rt_sum/o.repeats, rms_sum/o.repeats, rsk_sum/o.repeats, ev_sum/o.repeats,
               o.inf ? (int) E.ip.prerope : -1, o.inf ? E.ip.win_k : -1, o.inf ? E.ip.pinhost : -1,
               o.inf ? (int) E.ip.capblk : -1, o.bench_q, o.inf ? E.ip.evr_surp : -1.0, o.inf ? (int) E.ip.reqwin : -1,
               E.ip.feed, (int) E.ip.capasync, (int) E.ip.kvcap, (long long) fast_tok_last, (long long) cap_async_last);
        llama_model_free(model);
        return 0;
    }

    // ------------------------------------------------------------------ dump
    if (o.mode == "dump") {
        E.open(model, cp);
        std::vector<llama_token> pr; pr.push_back(llama_vocab_bos(vocab));
        pr.insert(pr.end(), HAY.begin(), HAY.begin() + (size_t) o.T - 1);
        const double t0 = now_s();
        if (!E.eval(pr.data(), pr.size(), true)) return 1;
        printf("# fed %d tokens in %.1fs (%.1f tok/s)\n", E.n_past(), now_s() - t0, E.n_past()/(now_s() - t0));
        if (E.inf) E.I->report(o.tag.c_str());
        else       printf("# dense arm: no policy state to dump\n");
        printf("# GPU %.2f GiB  RSS %.2f GiB\n", gpu_used()/1073741824.0, host_rss()/1073741824.0);
        E.close();
        llama_model_free(model);
        return 0;
    }

    // ---------------------------------------------------------------- kvtest
    // dense prefill with capture on (no transition: pass a W larger than T), then rebuild the
    // whole KV cache from the host store and continue.  Identical text => the capture, the f16
    // conversion, the row layout and (with --kvshift) the queued RoPE re-rotation are all correct.
    if (o.mode == "kvtest") {
        E.open(model, cp);
        std::vector<llama_token> pr; pr.push_back(llama_vocab_bos(vocab));
        pr.insert(pr.end(), HAY.begin(), HAY.begin() + (size_t) o.T - 1);
        if (!E.eval(pr.data(), pr.size(), true)) return 1;
        if      (E.inf && o.kvshift == -2) E.I->wtest(100, 1024);
        else if (E.inf && o.kvshift == -3) E.I->rewrite_all(0, false, true);   // no clear, data
        else if (E.inf && o.kvshift == -4) E.I->rewrite_all(0, true,  false);  // clear, metadata only
        else if (E.inf && o.kvshift >= 0)  E.I->rewrite_all(o.kvshift);
        std::string out;
        const int nv = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < o.n_predict; ++i) {
            const float * lg = llama_get_logits_ith(E.ctx, -1);
            llama_token best = 0; float bv = -1e30f;
            for (int v = 0; v < nv; ++v) if (lg[v] > bv) { bv = lg[v]; best = v; }
            out += detok(vocab, {best});
            if (!E.eval(&best, 1, true)) break;
        }
        for (auto & c : out) if (c == '\n' || c == '\r') c = '/';
        printf("KVTEST\t%s\tdesign=%s\tT=%d\tshift=%d\tcont=[%s]\n", o.tag.c_str(), design, o.T, o.kvshift, out.c_str());
        E.close();
        llama_model_free(model);
        return 0;
    }

    // ------------------------------------------------------------------ cont
    // feed T haystack tokens and greedily continue: a design whose trunk/band is wrong produces
    // visible garbage here, and the continuation can be compared across designs literally.
    if (o.mode == "cont") {
        E.open(model, cp);
        std::vector<llama_token> pr; pr.push_back(llama_vocab_bos(vocab));
        pr.insert(pr.end(), HAY.begin(), HAY.begin() + (size_t) o.T - 1);
        const double t0 = now_s();
        if (!E.eval(pr.data(), pr.size(), true)) return 1;
        E.set_decode(true);
        std::string out;
        const int nv = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < o.n_predict; ++i) {
            const float * lg = llama_get_logits_ith(E.ctx, -1);
            llama_token best = 0; float bv = -1e30f;
            for (int v = 0; v < nv; ++v) if (lg[v] > bv) { bv = lg[v]; best = v; }
            out += detok(vocab, {best});
            if (!E.eval(&best, 1, true)) break;
        }
        std::string tail = detok(vocab, std::vector<llama_token>(pr.end() - 12, pr.end()));
        for (auto & c : out)  if (c == '\n' || c == '\r') c = '/';
        for (auto & c : tail) if (c == '\n' || c == '\r') c = '/';
        printf("CONT\t%s\tdesign=%s\tT=%d\ttail=[%s]\tcont=[%s]\t(%.0fs)\n", o.tag.c_str(), design, o.T,
               tail.c_str(), out.c_str(), now_s() - t0);
        if (E.inf) E.I->report(o.tag.c_str());
        E.close();
        llama_model_free(model);
        return 0;
    }

    // ------------------------------------------------------------------- gen
    if (o.mode == "gen") {
        E.open(model, cp);
        auto pr = tokenize(vocab, o.prompt, true);
        E.eval(pr.data(), pr.size(), true);
        std::string out;
        const int nv = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < o.n_predict; ++i) {
            const float * lg = llama_get_logits_ith(E.ctx, -1);
            llama_token best = 0; float bv = -1e30f;
            for (int v = 0; v < nv; ++v) if (lg[v] > bv) { bv = lg[v]; best = v; }
            if (llama_vocab_is_eog(vocab, best)) break;
            out += detok(vocab, {best});
            E.eval(&best, 1, true);
        }
        printf("%s%s\n", o.prompt.c_str(), out.c_str());
        E.close();
        llama_model_free(model);
        return 0;
    }

    // ---------------------------------------------------------------- needle
    static const char * ADJ[] = {"ancient","bright","calm","daring","eager","faded","gentle","hidden","icy","jolly","keen","lucky","merry","noble","odd","proud","quiet","rapid","silent","tidy","upper","vivid","wild","young"};
    static const char * RAREW[] = {"zalquith","vorbeckle","mursaline","glenthrop","kavandrel","pyrrholux",
                                   "thessivar","brindolac","wexanthir","quollaben","drimspardo","fenrathul",
                                   "sarcovell","ulbrenthy","nyxaporta","jandricos","omvellrith","tazzuberic",
                                   "hesperloq","cindravek","morbaxial","pluvantis","redwithra","yarnoclase"};
    static const char * NOUN[] = {"anchor","basket","candle","dragon","engine","forest","garden","harbor","island","jacket","kettle","ladder","meadow","needle","orchard","pillar","quiver","river","saddle","temple","umbrella","valley","window","yacht"};
    const std::string PRE = "Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz you about the numbers afterwards.\n";

    std::vector<std::string> tasks;
    { std::stringstream ss(o.tasks); std::string t; while (std::getline(ss, t, ',')) tasks.push_back(t); }

    std::mt19937 rng(1234 + o.seed);
    struct row { std::string task; int dist; int em; int word = -1; };
    std::vector<row> rows;
    std::set<std::string> done;
    bool need_stamp = true;          // does the out file still need a "# build" line?

    if (o.resume) {
        FILE * f = fopen(o.out_file.c_str(), "r");
        if (f) {
            // --resume guarantees the INPUTS are identical across a chained job, not the BINARY.
            // A code change between two links of a chain silently produces one table row scored by
            // two different builds; that happened once and cost a needle at 32K on both models
            // (LOG 14).  Two guards: an explicit build stamp, and, for files written before the
            // stamp existed, the spread of the per-row prefill rate, which is a seam detector
            // because every row in one file is the same length.
            const std::string want = build_stamp();
            std::string have;
            double lo_sp = 1e30, hi_sp = 0.0;
            char ln[4096];
            while (fgets(ln, sizeof(ln), f)) {
                char bs[128];
                if (sscanf(ln, "# build %127s", bs) == 1) { have = bs; continue; }
                char tk[32]; int ex = 0, dist = 0, em = 0;
                if (sscanf(ln, " %31s ex %d needle %d back EM %d", tk, &ex, &dist, &em) == 4) {
                    done.insert(std::string(tk) + "/" + std::to_string(ex));
                    rows.push_back({tk, dist, em});
                    const char * p = strstr(ln, " tok/s");
                    if (p) {
                        const char * q = p; while (q > ln && *(q-1) != ',' && *(q-1) != '(') q--;
                        const double sp = atof(q);
                        if (sp > 0) { if (sp < lo_sp) lo_sp = sp; if (sp > hi_sp) hi_sp = sp; }
                    }
                }
            }
            fclose(f);
            bool drop = false;
            if (!have.empty() && have != want) {
                fprintf(stderr, "resume: REFUSED - %s was written by build %s, this is %s; "
                                "rescoring every needle\n", o.out_file.c_str(), have.c_str(), want.c_str());
                drop = true;
            } else if (have.empty() && !done.empty()) {
                fprintf(stderr, "resume: %s predates build stamping; checking the prefill rate for a seam\n",
                        o.out_file.c_str());
                if (hi_sp > 0 && lo_sp < 1e29 && hi_sp / lo_sp > 1.3) {
                    fprintf(stderr, "resume: REFUSED - prefill rate spans %.1f-%.1f tok/s (ratio %.2f) "
                                    "in one file, which means two builds; rescoring every needle\n",
                            lo_sp, hi_sp, hi_sp / lo_sp);
                    drop = true;
                }
            }
            if (drop) { done.clear(); rows.clear(); remove(o.out_file.c_str()); }
            else {
                need_stamp = (have != want);   // a legacy file gets stamped now, so the NEXT
                                               // resume with a different build is refused
                printf("# resume: %zu needles already scored in %s\n", done.size(), o.out_file.c_str());
            }
        }
    }
    FILE * fout = fopen(o.out_file.c_str(), "a");
    if (fout && need_stamp) { fprintf(fout, "# build %s\n", build_stamp().c_str()); fflush(fout); }
    if (fout) {   // 09-24: the applied configuration, every time the file is opened (evreenc explicit)
        fprintf(fout, "# applied %s  inf-set '%s'", o.inf ? "infinite" : "dense", o.inf_set.c_str());
        if (o.inf) fprintf(fout, "  SHIP5 prerope %d evreenc %d evr_budget %d unit %s pinhost %d capblk %d  |  evr_surp %.1f  win_k_seed %d  evr_max %d  reqfetch %d  win_k %d  prehand %d  evt %d  seed_span %d  reqwin %d  chunkcut %d  clusters %d  fillkeep %d  pf_gather %d  evt_block %d  trigword %d  trigfrac %.2f",
                           (int) E.ip.prerope, (int) E.ip.evreenc, E.ip.evr_budget,
                           E.ip.win_k > 0 ? ("win" + std::to_string(E.ip.win_k)).c_str() : (E.ip.unit_stride > 0 ? "chunk" : "sentence"),
                           E.ip.pinhost, (int) E.ip.capblk,
                           E.ip.evr_surp, E.ip.win_k_seed, E.ip.evr_max, (int) E.ip.reqfetch, E.ip.win_k,
                           (int) E.ip.prehand, (int) E.ip.evt, E.ip.seed_span, (int) E.ip.reqwin, (int) E.ip.chunk_cut,
                           (int) E.ip.use_clusters, (int) E.ip.fill_keep, (int) E.ip.pf_gather, (int) E.ip.evt_block,
                           (int) E.ip.trig_word, E.ip.trig_frac);
        if (o.inf) fprintf(fout, "  FINAL seed_span %d seed_words %d evr_surp %.1f floor %d unit %s ex0 %d",   // 09-24 v2bld23
                           E.ip.seed_span, E.ip.seed_words, E.ip.evr_surp, (int) E.ip.floor_only, unit_s.c_str(), o.ex0);
        fprintf(fout, "\n"); fflush(fout);
    }

    // ---------------------------------------------------------------------------------------
    // postest: THE ABSOLUTE-POSITION CONTROL.  The same tokens, twice, differing only in the
    // absolute positions they are given: 0..N-1 and base..base+N-1.  RoPE is relative, so the
    // logits must be identical -- unless the implementation loses precision at large positions.
    // theta = p * base^(-2i/d) is computed in fp32; at p = 2^20 the fastest planes carry ~0.06 rad
    // of rounding, and the query's error and the key's error do not cancel.  This is dense
    // llama.cpp only: no band, no store, nothing of ours in the path.
    // ---------------------------------------------------------------------------------------
    if (o.mode == "postest") {
        const int N = o.pos_n;
        if ((int) HAY.size() < N + 16) { fprintf(stderr, "haystack too small\n"); return 1; }
        std::vector<llama_token> doc(HAY.begin(), HAY.begin() + N);
        const int nv = llama_vocab_n_tokens(vocab);
        std::vector<float> lA(nv), lB(nv);

        for (int pass = 0; pass < 2; ++pass) {
            const long base = pass ? o.pos_base : 0;
            E.open(model, cp);
            llama_memory_t mem = llama_get_memory(E.ctx);
            llama_batch b = llama_batch_init(512, 0, 1);
            long p = 0;
            for (int i = 0; i < N; ) {
                const int take = std::min(512, N - i);
                b.n_tokens = take;
                for (int j = 0; j < take; ++j) {
                    b.token[j] = doc[i + j];
                    b.pos[j]   = (llama_pos)(p + j);
                    b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = 0;
                }
                if (i + take == N) b.logits[take - 1] = 1;
                if (llama_decode(E.ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
                i += take; p += take;
                // after the FIRST batch of the shifted pass, move everything up to `base`
                if (pass && i == take && base > 0) {
                    llama_memory_seq_add(mem, 0, 0, p, (llama_pos) base);
                    p += base;
                }
            }
            const float * lg = llama_get_logits_ith(E.ctx, -1);
            std::copy(lg, lg + nv, (pass ? lB : lA).begin());
            llama_batch_free(b);
            E.close();
        }
        double dot = 0, na = 0, nb = 0, mx = 0;
        int topA = 0, topB = 0;
        for (int v = 0; v < nv; ++v) {
            dot += (double) lA[v]*lB[v]; na += (double) lA[v]*lA[v]; nb += (double) lB[v]*lB[v];
            mx = std::max(mx, (double) fabsf(lA[v] - lB[v]));
            if (lA[v] > lA[topA]) topA = v;
            if (lB[v] > lB[topB]) topB = v;
        }
        printf("POSTEST\tn=%d\tbase=%ld\tcos=%.8f\tmaxabs=%.6f\ttop1_same=%d\ttopA=%s\ttopB=%s\n",
               N, o.pos_base, dot/sqrt(na*nb), mx, topA == topB ? 1 : 0,
               detok(vocab, {topA}).c_str(), detok(vocab, {topB}).c_str());
        return 0;
    }

    // ---------------------------------------------------------------------------------------
    // selftest: KNOWN-ANSWER TESTS on synthetic documents (v2/PIPELINE.md).  Every assertion is on
    // a COUNTER, never on exact match, because every defect this study found was invisible in EM.
    // Run against v1 first: six are predicted to fail and five of those six are defects already
    // found by hand.
    // ---------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------
    // loss: mean NLL over the last `lossn` tokens of a long prefill, as a function of WHERE the
    // hand-off gather happened.  tailq = Q holds the last Q history tokens back and encodes them
    // after the gather, so Q >= lossn means the whole scored region saw the band and Q = 0 means
    // none of it did - which is what prefill does today.
    // ---------------------------------------------------------------------------------------
    if (o.mode == "loss") {
        const int L_s = o.lossn > 0 ? o.lossn : 4096;
        if ((int) HAY.size() < o.T + 16) { fprintf(stderr, "haystack too small\n"); return 1; }
        std::vector<llama_token> pr; pr.push_back(llama_vocab_bos(vocab));
        pr.insert(pr.end(), HAY.begin(), HAY.begin() + (size_t) o.T - 1);
        const int T = (int) pr.size();
        if (L_s >= T - 1024) { fprintf(stderr, "lossn too large for T\n"); return 1; }

        E.open(model, cp);
        const int Q   = (E.inf && E.I->p.tailq > 0) ? E.I->p.tailq : 0;
        if (E.inf) {
            int32_t N = E.I->p.pf_depth; if (N < 0) N = E.I->p.W;
            E.I->pf_from = N > 0 ? std::max(0, T - N) : 2147483647;
        }
        const int cut = T - L_s;
        const int gpos = T - Q;                       // where the gather lands (Q = 0 -> never)
        const double t0 = now_s();
        if (!E.eval(pr.data(), cut, true)) { E.close(); return 1; }
        if (E.inf && Q >= L_s) { E.I->pre_handoff(pr.data() + cut, T - cut); }

        double nll = 0.0; int n_sc = 0;
        const int nv = llama_vocab_n_tokens(vocab);
        for (int i = cut; i + 1 < T; ++i) {
            if (E.inf && Q > 0 && Q < L_s && i == gpos) E.I->pre_handoff(pr.data() + i, T - i);
            const float * lg = llama_get_logits_ith(E.ctx, -1);
            double mx = -1e30; for (int v = 0; v < nv; ++v) if (lg[v] > mx) mx = lg[v];
            double den = 0.0; for (int v = 0; v < nv; ++v) den += exp((double) lg[v] - mx);
            nll += -((double) lg[pr[i+1]] - mx - log(den));
            n_sc++;
            if (!E.eval(&pr[i+1], 1, true)) break;
        }
        const double band = E.inf ? (double) E.I->band_live() : 0.0;
        printf("LOSS\t%s\tdesign=%s\tT=%d\tlossn=%d\ttailq=%d\tnll=%.5f\tppl=%.4f\tn=%d\t"
               "band_live=%.0f\tgathers=%lld\t%.0fs\n",
               o.tag.c_str(), design, o.T, L_s, Q, nll/std::max(1,n_sc), exp(nll/std::max(1,n_sc)),
               n_sc, band, E.inf ? (long long) E.I->n_gather : 0LL, now_s() - t0);
        fflush(stdout);
        E.close();
        llama_model_free(model);
        return 0;
    }

    // ---------------------------------------------------------------------------------------
    // shifttest: the band's keys are RoPEd at their original absolute position and then K-SHIFTED
    // by a large negative delta - two f32 rotations - while the harness rotates a pre-RoPE key
    // once at the slot distance.  This asks whether that double rotation is lossy at 2^20.
    //   arm N: prefill N tokens natively at positions [base, base+N)
    //   arm S: prefill the same N tokens at [0, N), then seq_add the WHOLE range by base
    // Both then decode the SAME 256 tokens (arm N's greedy continuation) and the logits are
    // compared step by step.  If S diverges from N at 2^20 and not at 2^17, the double rotation
    // at large positions is the 1M bug.
    // ---------------------------------------------------------------------------------------
    if (o.mode == "shifttest") {
        const int N = o.pos_n;
        const long base = o.pos_base;
        const int G = 256;
        if ((int) HAY.size() < N + 16) { fprintf(stderr, "haystack too small\n"); return 1; }
        std::vector<llama_token> doc(HAY.begin(), HAY.begin() + N);
        const int nv = llama_vocab_n_tokens(vocab);
        std::vector<llama_token> gen;
        std::vector<std::vector<float>> LG[2];

        for (int pass = 0; pass < 2; ++pass) {
            E.open(model, cp);
            llama_memory_t mem = llama_get_memory(E.ctx);
            llama_batch b = llama_batch_init(512, 0, 1);
            const long p0 = pass ? 0 : base;          // pass 1 prefills at 0 and is shifted after
            for (int i = 0; i < N; ) {
                const int take = std::min(512, N - i);
                b.n_tokens = take;
                for (int j = 0; j < take; ++j) {
                    b.token[j] = doc[i + j];
                    b.pos[j]   = (llama_pos)(p0 + i + j);
                    b.n_seq_id[j] = 1; b.seq_id[j][0] = 0;
                    b.logits[j] = (i + take == N && j == take - 1) ? 1 : 0;
                }
                if (llama_decode(E.ctx, b) != 0) { fprintf(stderr, "prefill failed\n"); return 1; }
                i += take;
            }
            if (pass) {
                // THE SHIFT PATH: every key was rotated at its position in [0,N) and is now
                // rotated again by +base.  This is what the band does, at 400K-scale deltas.
                llama_memory_seq_add(mem, 0, 0, (llama_pos) N, (llama_pos) base);
            }
            LG[pass].resize(G);
            for (int t = 0; t < G; ++t) {
                const float * lg = llama_get_logits_ith(E.ctx, -1);
                LG[pass][t].assign(lg, lg + nv);
                llama_token nxt;
                if (!pass) {
                    nxt = 0; float bv = -1e30f;
                    for (int v = 0; v < nv; ++v) if (lg[v] > bv) { bv = lg[v]; nxt = v; }
                    gen.push_back(nxt);
                } else nxt = gen[t];                 // the SAME tokens, so logits are comparable
                b.n_tokens = 1; b.token[0] = nxt;
                b.pos[0] = (llama_pos)(base + N + t);
                b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
                if (llama_decode(E.ctx, b) != 0) break;
            }
            llama_batch_free(b);
            E.close();
        }
        int top_disagree = 0; double worst = 0, cos_min = 2.0, sum_mx = 0;
        for (int t = 0; t < G; ++t) {
            const auto & A = LG[0][t]; const auto & B = LG[1][t];
            if (A.empty() || B.empty()) continue;
            double d = 0, na = 0, nb = 0, mx = 0; int ta = 0, tb = 0;
            for (int v = 0; v < nv; ++v) {
                d += (double) A[v]*B[v]; na += (double) A[v]*A[v]; nb += (double) B[v]*B[v];
                const double e = fabs((double) A[v] - (double) B[v]); if (e > mx) mx = e;
                if (A[v] > A[ta]) ta = v;
                if (B[v] > B[tb]) tb = v;
            }
            const double c = (na > 0 && nb > 0) ? d/sqrt(na*nb) : 1.0;
            if (c < cos_min) cos_min = c;
            if (mx > worst) worst = mx;
            sum_mx += mx;
            if (ta != tb) top_disagree++;
            if (t % 64 == 0)
                printf("  step %3d  cos %.6f  max|d| %.4f  top1 %s\n", t, c, mx, ta == tb ? "same" : "DIFFER");
        }
        printf("SHIFTTEST\t%s\tN=%d\tbase=%ld\tsteps=%d\ttop1_differ=%d\tcos_min=%.6f\t"
               "max|d|=%.4f\tmean_max|d|=%.4f\n",
               o.tag.c_str(), N, base, G, top_disagree, cos_min, worst, sum_mx/G);
        fflush(stdout);
        llama_model_free(model);
        return 0;
    }

    if (o.mode == "selftest") {
        int npass = 0, nfail = 0;
        auto T = [&](const char * id, bool ok, const char * what, const std::string & got) {
            printf("TEST %-4s %-4s %-52s %s\n", id, ok ? "PASS" : "FAIL", what, got.c_str());
            (ok ? npass : nfail)++;
        };
        auto S = [](double v) { char b[64]; snprintf(b, sizeof(b), "%.4g", v); return std::string(b); };

        // ---- a synthetic document with planted structure -----------------------------------
        // decimals, abbreviations, a version, a URL, a path, and HARD-WRAPPED prose
        std::string doc = "Some notes follow.\n";
        for (int i = 0; i < 40; ++i) {
            doc += "The ratio was 3.14 and the sample, e.g. the second one, held v0.3 of the\n";
            doc += "software from https://example.org/a/b and /usr/local/share/data which Dr.\n";
            doc += "Smith had prepared for the meeting on the following day of that week.\n";
        }
        // a planted rare word in a known number of places
        const std::string RARE = "zalquithra";
        std::vector<size_t> plant_at;
        for (int i = 0; i < 5; ++i) { doc += "\nThe marker " + RARE + " appears here.\n"; }
        // a word that is common in the document and must not be indexed under a threshold
        for (int i = 0; i < 60; ++i) doc += "The ordinary word appears again and again here.\n";

        auto ids = tokenize(vocab, doc, true);
        const size_t n_doc_head = ids.size();
        // The far tier only exists above L.  Pad with haystack so the same planted structure is
        // exercised with the band built; --T 0 keeps the short below-range document for T12.
        if (o.T > 0) {
            const size_t want = (size_t) o.T;
            size_t k = 0;
            while (ids.size() < want && k < HAY.size()) ids.push_back(HAY[k++]);
            // plant the rare word again in the padded region, 5 more times, evenly spaced
            // (postings are capped, so the count below is a cap test, not a count test)
        }
        printf("# selftest document: %zu tokens (%zu planted head + pad)\n", ids.size(), n_doc_head);

        E.open(model, cp);
        if (!E.inf) { fprintf(stderr, "selftest needs --inf\n"); return 1; }
        if (!E.eval(ids.data(), (int32_t) ids.size(), true)) { E.close(); return 1; }
        inf_ctx * I = E.I;

        // ---- T23: the unigram model counts every token exactly once ------------------------
        // The pre-encode hand-off computes the request words' surprisal, so the request has to be
        // counted BEFORE that -- which is the order the after-encode path sees, since eval() adds
        // every token before block_update() runs.  Counting it twice would weight the request's
        // own words as twice as common; not counting it would weight them as unseen.  Either way
        // the two orderings would rank the same query differently, which is invisible in EM.
        T("T23", I->freq.n_tok == (int64_t) I->n_past,
          "the unigram model counts every token exactly once",
          S((double) I->freq.n_tok) + " counts for " + S((double) I->n_past) + " tokens");

        // ---- T7: every token captured exactly once -----------------------------------------
        T("T7", I->n_past == (int32_t) ids.size(), "every token seen exactly once",
          S(I->n_past) + " of " + S((double) ids.size()));

        // ---- T1/T2: unit cutting ------------------------------------------------------------
        std::vector<int32_t> ulen;
        for (auto & u : I->units) ulen.push_back(u.t1 - u.t0);
        std::sort(ulen.begin(), ulen.end());
        const double umed = ulen.empty() ? 0 : ulen[ulen.size()/2];
        // Under the fixed-chunk rule the median must be the chunk length exactly; under the
        // sentence rule the weaker "not line-sized" bound is all that can be asked.
        const bool fixed_units = I->p.unit_stride > 0;
        T("T2", fixed_units ? (fabs(umed - I->p.unit_len) < 1.0) : (umed >= 64),
          fixed_units ? "unit median length == unit_len" : "unit median length >= 64",
          "median " + S(umed) + " over " + S((double) I->units.size()) + " units"
          + (fixed_units ? " (fixed " + S(I->p.unit_len) + "/" + S(I->p.unit_stride) + ")" : ""));
        // T1: a unit boundary must not fall inside "3.14" -- find the token holding "3" followed
        // by one holding ".14" or "." and check they are in the same unit
        int splits = 0, checked = 0;
        for (size_t k = 1; k + 1 < ids.size(); ++k) {
            const std::string & a = I->piece[ids[k]];
            if (a.find('.') == std::string::npos) continue;
            if (a.size() > 3) continue;
            // a piece containing '.' between two alphanumerics: a decimal or an abbreviation
            const std::string & pv = I->piece[ids[k-1]], & nx = I->piece[ids[k+1]];
            if (pv.empty() || nx.empty()) continue;
            if (!isalnum((unsigned char) pv.back()) || !isalnum((unsigned char) nx[0])) continue;
            checked++;
            // 09-23: with OVERLAPPING fixed units (unit_len > unit_stride) a site is split only if
            // NO unit holds both tokens; the old check compared the LAST unit holding each token,
            // which flags every site that sits on a stride boundary even though the previous
            // (overlapping) unit holds it whole.
            bool any_k = false, any_k1 = false, both = false;
            for (size_t q = 0; q < I->units.size(); ++q) {
                const bool hk  = (int32_t) k   >= I->units[q].t0 && (int32_t) k   < I->units[q].t1;
                const bool hk1 = (int32_t) k+1 >= I->units[q].t0 && (int32_t) k+1 < I->units[q].t1;
                any_k |= hk; any_k1 |= hk1; both |= (hk && hk1);
            }
            if (any_k && any_k1 && !both) splits++;
        }
        T("T1", splits == 0, "no unit boundary inside a decimal or abbreviation",
          S(splits) + " splits of " + S(checked) + " such sites");

        // ---- T3/T4/T5/T6: index contents -----------------------------------------------------
        const auto rid = tokenize(vocab, " " + RARE, false);
        std::string rw; for (auto t : rid) rw += I->piece[t];
        std::string rclean; for (unsigned char c : rw) if (isalnum(c) || c=='-' || c=='_') rclean += (char) c;
        uint64_t rh = 1469598103934665603ull;
        for (unsigned char c : rclean) { if (c>='A'&&c<='Z') c = c-'A'+'a'; rh ^= c; rh *= 1099511628211ull; }
        // The word index is only built when the event path or the word trigger is on; in
        // token-id trigger mode winv is empty BY DESIGN, and a test that reported that as a
        // failure would be testing the mode, not the policy.  Say which mode this is.
        const bool word_mode = I->p.evt || I->p.trig_word;
        printf("# trigger mode: %s (evt=%d trigword=%d)\n",
               word_mode ? "WORD index" : "token-id index, winv not built",
               (int) I->p.evt, (int) I->p.trig_word);
        auto wit = I->winv.find(rh);
        const size_t nrp = wit == I->winv.end() ? 0 : wit->second.size();
        T("T3", word_mode ? (nrp > 0 && nrp <= 5) : I->winv.empty(),
          word_mode ? "a planted rare word posts its own units"
                    : "token-id mode: the word index is correctly absent",
          S((double) nrp) + " postings (planted 5)");
        T("T4", (int32_t) nrp <= I->p.post_cap, "postings capped at post_cap",
          S((double) nrp) + " <= " + S(I->p.post_cap));
        // T5: a COMMON word must not be indexed under the threshold
        const auto oid = tokenize(vocab, " ordinary", false);
        std::string ow; for (auto t : oid) ow += I->piece[t];
        std::string oc; for (unsigned char c : ow) if (isalnum(c)) oc += (char) c;
        uint64_t oh = 1469598103934665603ull;
        for (unsigned char c : oc) { if (c>='A'&&c<='Z') c = c-'A'+'a'; oh ^= c; oh *= 1099511628211ull; }
        const bool common_indexed = I->winv.find(oh) != I->winv.end();
        T("T5", !word_mode || !common_indexed,
          word_mode ? "a word below the surprisal threshold is NOT indexed"
                    : "T5 not applicable in token-id mode",
          common_indexed ? "indexed (no threshold applied)" : "absent");

        // ---- T8/T9/T10: band invariants, contiguity, provenance ------------------------------
        if (I->two_tier) {
            int live = 0, dup = 0; std::unordered_set<int32_t> seen;
            for (int32_t c = 0; c < I->p.S; ++c) if (I->band_tok[c] >= 0) {
                live++; if (!seen.insert(I->band_tok[c]).second) dup++;
            }
            T("T8", dup == 0, "no token occupies two band cells", S(dup) + " duplicates of " + S(live));
            const double rpg = I->n_gather ? (double) I->n_runs / I->n_gather : 0;
            T("T9", rpg < 8.0, "writes stay contiguous (runs per gather)", S(rpg) + " runs/gather");
            const double gsum = I->n_comp_gathers ? (double)(I->n_comp_cur + I->n_comp_kept + I->n_comp_fill)
                                                   / I->n_comp_gathers : 0;
            T("T10", I->n_comp_gathers == 0 || fabs(gsum - live) < 0.05*I->p.S + 2,
              "provenance accounts for every live cell", S(gsum) + " vs " + S(live) + " live");
        } else {
            printf("TEST T8   SKIP  band not built at this length\n");
        }

        // ---- T25: the pinned prefix equals its host copy, rotated once (09-24 pin fix) -------
        // In-place K-shifts round every pin to f16 at every block; the slowest RoPE planes turn by
        // less than half an f16 ulp per 128-position shift and stop turning, so the error grows
        // linearly in angle with the number of moves (v1: sink cos 0.992 after ~1000 moves at 128K,
        // 0.432 after ~8000 at 1M).  pinhost re-places the pins from the host each move.  Below
        // ~500 moves the in-place path still clears 0.999, so the test only discriminates at
        // T >= ~72K (window-only prefill) -- the V2PIN jobs run it at 128K with pinhost=1 and =0.
        {
            double cmin = 2, cmean = 0, smin = 2, dmax = 0; int64_t nr = 0;
            const bool have = I->pin_host_cos(cmin, cmean, smin, dmax, nr);
            const long long mv = (long long) I->n_pin_moves;
            char g[320];
            snprintf(g, sizeof(g), "cos min %.6f mean %.6f sink %.6f max|d| %.4f over %lld rows after %lld pin moves "
                     "(pinhost %d, %lld host re-places, %lld fallbacks)%s", cmin, cmean, smin, dmax, (long long) nr, mv,
                     I->p.pinhost, (long long) I->n_pin_host, (long long) I->n_pin_host_fallback,
                     mv < 500 ? "; < 500 moves: not discriminating" : "");
            if (!have || mv == 0) printf("TEST T25  SKIP  pins not resident or never moved (%lld moves)\n", mv);
            else T("T25", cmin >= 0.999, "pinned K equals the host copy rotated once (cos>=0.999)", g);
        }

        // ---- T24: every band and trunk cell equals its host copy (09-24 band check) ----------
        // V byte for byte, K by cosine against the host key rotated once (double) to the cell's
        // position; band provenance from the cache's own ext tok.  Would have caught D1 (the
        // un-rotated trunk after the transition) and any K-vs-V cell-order mismatch.
        if (!I->two_tier && I->n_past >= I->p.W + I->p.S + I->p.n_pin + 2) {
            // window-only prefill (CORE_PH) builds no band until a request: force the hand-off
            // gather here, exactly as set_decode(true) would, so T24 checks the production layout
            printf("# T24: forcing the hand-off gather at p = %d (window-only prefill)\n", I->n_past);
            I->in_decode = true; I->regather(); I->in_decode = false;
        }
        if (I->two_tier) {
            const int64_t k0 = I->n_bchk_k, v0 = I->n_bchk_v, m0 = I->n_bchk_meta, s0 = I->n_bchk_slot,
                          d0 = I->n_bchk_dup, tk0 = I->n_bchk_tr_k, tv0 = I->n_bchk_tr_v,
                          bl0 = I->n_bchk_live, tl0 = I->n_bchk_tr_live;
            I->band_check("T24", I->p_place);
            const int64_t bad = (I->n_bchk_k - k0) + (I->n_bchk_v - v0) + (I->n_bchk_meta - m0) + (I->n_bchk_slot - s0)
                              + (I->n_bchk_dup - d0) + (I->n_bchk_tr_k - tk0) + (I->n_bchk_tr_v - tv0);
            char g[256];
            snprintf(g, sizeof(g), "band %lld live, trunk %lld live; bad: meta %lld slot %lld dup %lld V %lld K %lld, trunk V %lld K %lld",
                     (long long) (I->n_bchk_live - bl0), (long long) (I->n_bchk_tr_live - tl0),
                     (long long) (I->n_bchk_meta - m0), (long long) (I->n_bchk_slot - s0), (long long) (I->n_bchk_dup - d0),
                     (long long) (I->n_bchk_v - v0), (long long) (I->n_bchk_k - k0),
                     (long long) (I->n_bchk_tr_v - tv0), (long long) (I->n_bchk_tr_k - tk0));
            T("T24", bad == 0 && (I->n_bchk_tr_live - tl0) > 0, "band and trunk cells equal their host copies", g);
        } else {
            printf("TEST T24  SKIP  no two-tier layout at this length\n");
        }

        // ---- T20: events off means no event attempts ------------------------------------------
        T("T20", I->p.evt || I->n_evt_try == 0, "events off => zero event attempts",
          S((double) I->n_evt_try) + " attempts");

        // ---- T18: a hash miss must be COUNTABLE ----------------------------------------------
        const bool countable = true;   // the counter exists in this build iff it compiles
        T("T18", countable && (I->n_evt_nohash + I->n_evt_trunk + I->n_evt_resident + I->n_evt_fetch
                               == I->n_evt_try),
          "event attempts classify exhaustively",
          S((double)(I->n_evt_nohash + I->n_evt_trunk + I->n_evt_resident + I->n_evt_fetch))
          + " of " + S((double) I->n_evt_try));

        // ---- T6: one hash across case, punctuation and tokenizer split ---------------------
        {
            const char * var[3] = { " Cactus.", " cactus", " CACTUS," };
            uint64_t h[3];
            for (int v = 0; v < 3; ++v) {
                const auto vid = tokenize(vocab, var[v], false);
                std::string w; for (auto t : vid) w += I->piece[t];
                std::string c; for (unsigned char ch : w) if (isalnum(ch)||ch=='-'||ch=='_') c += (char) ch;
                uint64_t x = 1469598103934665603ull;
                for (unsigned char ch : c) { if (ch>='A'&&ch<='Z') ch = ch-'A'+'a'; x ^= ch; x *= 1099511628211ull; }
                h[v] = x;
            }
            T("T6", h[0] == h[1] && h[1] == h[2], "case/punctuation variants share one hash",
              h[0]==h[1]&&h[1]==h[2] ? "one hash" : "three hashes");
        }

        // ---- T13/T14: the candidate list after a query with one planted match ---------------
        if (I->two_tier) {
            const auto qid = tokenize(vocab, "\nWhat about " + RARE + "?\n", false);
            I->in_decode = true;
            I->select(I->n_past, qid.data(), (int32_t) qid.size());
            // 09-23: the old T13/T14 looked for UNIT ids inside I->cand, which holds TOKEN
            // positions from the cluster vote (step 3), so they could never pass.  The units the
            // triggers hit are I->last_hit.  The planted units are the word index's postings for
            // RARE in word mode and the token-id index's postings for RARE's pieces otherwise.
            std::unordered_set<int32_t> planted;
            if (word_mode) { if (wit != I->winv.end()) for (auto u : wit->second) planted.insert(u); }
            else for (auto id : rid) { auto it2 = I->inv.find(id); if (it2 != I->inv.end()) for (auto u : it2->second) planted.insert(u); }
            const size_t ncand = I->last_hit.size();
            bool planted_in = false;
            for (auto u : I->last_hit) if (planted.count(u)) { planted_in = true; break; }
            T("T13", planted_in, "a unit matching a query word is a trigger hit",
              S((double) ncand) + " trigger-hit units, " + S((double) planted.size()) + " planted-word units");
            // T14: a hit unit sharing no query word means the hit list is not index-driven
            size_t unrelated = 0;
            for (auto u : I->last_hit) if (!planted.count(u)) unrelated++;
            T("T14", unrelated * 2 < ncand + 1, "trigger hits are index-driven, not back-filled",
              S((double) unrelated) + " of " + S((double) ncand) + " hit units hold no planted query word");
            // ---- T15: demand is bounded by k * post_cap * unit_len ---------------------------
            const double ulen_mean = I->units.empty() ? 0 :
                (double) I->n_past / (double) I->units.size();
            const double bound = (double) I->n_req_words * I->p.post_cap * ulen_mean;
            const double want  = (double) I->n_trig_want;
            T("T15", want <= bound + ulen_mean + 1 || I->n_req_words == 0,
              "retrieval demand is bounded by k*post_cap*unit_len",
              "want " + S(want) + " bound " + S(bound) + " S " + S(I->p.S));
            I->in_decode = false;
        }

        // ---- T11: an over-sized configuration must be REFUSED --------------------------------
        {
            const int32_t L = (int32_t) llama_n_ctx(E.ctx);
            const bool oversized = I->p.W + I->p.S + I->p.n_pin + 2 > L;
            // init() returns void, so there is no channel through which it could refuse
            const bool can_refuse = false;
            T("T11", can_refuse, "an impossible W+S+n_pin+2 > L config is refused",
              can_refuse ? "refused" : std::string("init() returns void; L=") + S(L)
              + " W+S+pin+2=" + S(I->p.W + I->p.S + I->p.n_pin + 2)
              + (oversized ? " (OVERSIZED AND ACCEPTED)" : " (fits here)"));
        }

        // ---- T21: the resume key must include the CONFIGURATION, not only the binary ---------
        {
            const std::string k1 = build_stamp();
            // the same binary with a different policy must produce a different key
            const bool config_in_key = false;   // build_stamp() reads only /proc/self/exe
            T("T21", config_in_key, "the resume key covers the configuration",
              "key=" + k1 + " (executable mtime+size only)");
        }

        const int32_t L_ctx = (int32_t) llama_n_ctx(E.ctx);
        E.close();

        // ---- T26: evreenc is exact (09-24) --------------------------------------------------------
        // Two fresh contexts on the same history (rounded down to a block boundary so the tail sits
        // inside one block and no block_update() separates the arms), run one after the other so
        // only one extra KV cache is resident (MIG).  B: the pre-encode hand-off, the planted rare
        // word's units fetched by an event BEFORE anything of the tail is encoded, then request + K
        // forced "generated" tokens prefilled in one batch = "a fresh prefill of context + band +
        // tail".  A: the same hand-off, the request encoded, the K tokens decoded, then the same
        // event fetch and the tail re-encoded (reencode_tail).  Asserted: A's re-encoded logits equal
        // B's (max |d| <= 1e-3, same argmax); the logits after one more decoded token agree too (the
        // whole tail's K/V, not only its last row); A's host store equals its cache after the
        // re-encode (band_check, 0 bad); and A's logits BEFORE the re-encode (the strict late
        // arrival) differ from B's by more than the tolerance, or the test would not discriminate.
        // Events are off in both arms (event_gather is driven directly), so nothing the test feeds
        // can fire a second, unmatched fetch.
        {
            const int32_t Bk = E.ip.B;
            const int32_t n_hist = ((int32_t) ids.size() / Bk) * Bk;
            const int32_t Lr = E.ip.W + E.ip.S + E.ip.n_pin + 2;
            if (!E.ip.prehand || n_hist < Lr) {
                printf("TEST T26  SKIP  needs prehand=1 and a history >= L (%d < %d or prehand %d)\n",
                       n_hist, Lr, (int) E.ip.prehand);
            } else {
                const auto rq = tokenize(vocab, "\nWhat does the note say about the meeting?\nAnswer:", false);
                const auto gn = tokenize(vocab, " The marker is", false);
                inf_params tp = E.ip; tp.evt = false; tp.trig_word = true;
                const int nv = llama_vocab_n_tokens(vocab);
                auto amax = [&](const std::vector<float> & v) { return (int) (std::max_element(v.begin(), v.end()) - v.begin()); };
                auto dmax = [&](const std::vector<float> & a, const std::vector<float> & b) {
                    double m = 0; for (int i = 0; i < nv; ++i) m = std::max(m, fabs((double) a[i] - (double) b[i])); return m; };
                auto grab = [&](llama_context * c) { const float * l = llama_get_logits_ith(c, -1); return std::vector<float>(l, l + nv); };
                auto units_of = [&](inf_ctx * X) {   // the planted word's postings, most recent first, capped at evt_cap
                    std::vector<int32_t> u;
                    auto it = X->winv.find(rh);
                    if (it != X->winv.end())
                        for (int32_t i = (int32_t) it->second.size() - 1; i >= 0 && (int32_t) u.size() < tp.evt_cap; --i)
                            u.push_back(it->second[i]);
                    return u;
                };

                // arm B: the fetch before the tail, then one prefill of the tail
                engine Bn; Bn.inf = true; Bn.ip = tp; Bn.open(model, cp);
                bool okB = Bn.eval(ids.data(), n_hist, false);
                const std::vector<int32_t> usB = units_of(Bn.I);
                Bn.I->pre_handoff(rq.data(), (int32_t) rq.size());
                const int64_t fB0 = Bn.I->n_evt_fetch, cellsB0 = Bn.I->n_event_cells;
                Bn.I->event_gather(usB, Bn.I->n_past);
                const int64_t fetchedB = Bn.I->n_evt_fetch - fB0, cellsB = Bn.I->n_event_cells - cellsB0;
                std::vector<llama_token> tail = rq; tail.insert(tail.end(), gn.begin(), gn.end());
                okB = okB && Bn.eval(tail.data(), (int32_t) tail.size(), true);
                const std::vector<float> l_ref = grab(Bn.ctx);
                llama_token nx = (llama_token) amax(l_ref);
                Bn.set_decode(true);
                okB = okB && Bn.eval(&nx, 1, true);
                const std::vector<float> n_ref = grab(Bn.ctx);
                Bn.close();

                // arm A: strict late arrival, then the re-encode
                engine A; A.inf = true; A.ip = tp; A.open(model, cp);
                bool okA = A.eval(ids.data(), n_hist, false);
                const std::vector<int32_t> usA = units_of(A.I);
                A.I->pre_handoff(rq.data(), (int32_t) rq.size());
                okA = okA && A.eval(rq.data(), (int32_t) rq.size(), false);
                A.set_decode(true);
                for (auto g : gn) okA = okA && A.eval(&g, 1, true);
                const std::vector<float> l_strict = grab(A.ctx);
                const int64_t fA0 = A.I->n_evt_fetch, cellsA0 = A.I->n_event_cells;
                A.I->event_gather(usA, A.I->n_past);
                const int64_t fetchedA = A.I->n_evt_fetch - fA0, cellsA = A.I->n_event_cells - cellsA0;
                const int32_t t0A = A.I->turn_t0, npA = A.I->n_past;
                // 09-24 evr_budget: with a budget smaller than the tail the re-encode must be skipped
                // (strict path kept: logits untouched, one skip counted, nothing charged to the turn)
                const int32_t bud_save = A.I->p.evr_budget;
                A.I->p.evr_budget = 1;
                const int64_t sk0 = A.I->n_reenc_budget_skip, rn0 = A.I->n_reenc;
                okA = okA && A.I->reencode_tail();
                const std::vector<float> l_skip = grab(A.ctx);
                const bool skip_ok = (A.I->n_reenc_budget_skip - sk0) == 1 && A.I->n_reenc == rn0 && A.I->reenc_turn_tok == 0
                                  && memcmp(l_skip.data(), l_strict.data(), sizeof(float) * (size_t) nv) == 0;
                A.I->p.evr_budget = bud_save;
                okA = okA && A.I->reencode_tail();
                const std::vector<float> l_re = grab(A.ctx);
                const int64_t tk0 = A.I->n_bchk_tr_k, tv0 = A.I->n_bchk_tr_v, bk0 = A.I->n_bchk_k, bv0 = A.I->n_bchk_v,
                              tl0 = A.I->n_bchk_tr_live;
                A.I->band_check("T26", A.I->p_place);
                const int64_t hbad = (A.I->n_bchk_tr_k - tk0) + (A.I->n_bchk_tr_v - tv0) + (A.I->n_bchk_k - bk0) + (A.I->n_bchk_v - bv0);
                const int64_t tlive = A.I->n_bchk_tr_live - tl0;
                okA = okA && A.eval(&nx, 1, true);
                const std::vector<float> n_re = grab(A.ctx);
                const long long rn = (long long) A.I->n_reenc, rt = (long long) A.I->n_reenc_tok;
                const double rms = 1000.0*A.I->t_reenc;
                A.close();

                const double d_re = dmax(l_re, l_ref), d_nx = dmax(n_re, n_ref), d_st = dmax(l_strict, l_ref);
                double lmax = 0; for (int i = 0; i < nv; ++i) lmax = std::max(lmax, fabs((double) l_ref[i]));
                int nbyte = 0; for (int i = 0; i < nv; ++i) if (memcmp(&l_re[i], &l_ref[i], sizeof(float)) != 0) nbyte++;
                const bool same_am = amax(l_re) == amax(l_ref) && amax(n_re) == amax(n_ref);
                const double tol = 1e-3;
                const bool ok = okA && okB && fetchedA == 1 && fetchedB == 1 && cellsA == cellsB && cellsA > 0
                             && usA == usB && d_re <= tol && d_nx <= tol && same_am && hbad == 0 && tlive > 0 && d_st > tol
                             && skip_ok;
                char g[896];
                snprintf(g, sizeof(g), "tail [%d,%d) = %d tokens (request %zu + gen %zu), event units %zu, cells %lld/%lld; "
                         "re vs fresh max|d| %.3g (%d/%d floats differ, |logit|max %.1f), next token max|d| %.3g, argmax %s; "
                         "strict vs fresh max|d| %.3g (must exceed %.0e); host store after re-encode: %lld bad of %lld trunk cells; "
                         "reencodes %lld tokens %lld %.1f ms; budget-1 skip %s",
                         t0A, npA, npA - t0A, rq.size(), gn.size(), usA.size(), (long long) cellsA, (long long) cellsB,
                         d_re, nbyte, nv, lmax, d_nx, same_am ? "same" : "DIFFERENT", d_st, tol,
                         (long long) hbad, (long long) tlive, rn, rt, rms, skip_ok ? "ok (strict kept, 1 counted)" : "WRONG");
                T("T26", ok, "evreenc: re-encoded tail == fresh prefill of ctx+band+tail", g);
            }
        }


        // ---- T27: unit=win<K> band composition vs an independent Python computation (09-24) ---
        // A fresh context with win_log on (win_k from the run, else 256) prefills the document to a
        // block boundary; the query is the planted word plus the (up to) 8 far terms with the most
        // above-threshold occurrences, so the cap, clipping, merging and scoring all get exercised.
        // The pre-encode hand-off gathers on it.  The dump holds the inputs (every closed word with
        // its position and surprisal, the query terms and weights, the parameters) and the outputs
        // (the merged windows, their admission order, the admitted token list before any fill, and
        // how many of those tokens are resident in the band).  t27check.py recomputes the rule from
        // the inputs alone and gives the verdict; the job's exit code is its exit code.
        {
            const int32_t Bk = E.ip.B;
            const int32_t n_hist = ((int32_t) ids.size() / Bk) * Bk;
            const int32_t Lr = E.ip.W + E.ip.S + E.ip.n_pin + 2;
            if (o.t27_dump.empty() || !E.ip.prehand || n_hist < Lr) {
                printf("TEST T27  SKIP  needs --t27-dump FILE, prehand=1 and a history >= L\n");
            } else {
                inf_params tp = E.ip; tp.evt = false; tp.trig_word = true; tp.win_log = true;
                // 09-25 v2bld31: T27 checks the WINDOW rule against t27check.py, which models windows only, so every
                // query term (request and seed) takes windows here; the seed's sentence units under unit=win<K> are the
                // U slate's S256 / S512 path (LOG 136p-136q), not this test's
                tp.seed_sent = false;
                if (tp.win_k <= 0) tp.win_k = 256;
                engine X; X.inf = true; X.ip = tp; X.open(model, cp);
                const bool okx = X.eval(ids.data(), n_hist, false);
                inf_ctx * J = X.I;
                const int32_t hi0 = n_hist - tp.W;
                std::unordered_map<uint64_t, std::pair<int, std::string>> cnt;
                for (const auto & ev : J->wocc_log)
                    if (ev.surp > tp.trig_surp && ev.ws < hi0) { auto & c = cnt[ev.h]; c.first++; c.second = ev.w; }
                std::vector<std::pair<int, std::string>> cands;
                for (const auto & kv : cnt) if (kv.second.first >= 2) cands.push_back(kv.second);
                std::sort(cands.begin(), cands.end(), [](const std::pair<int,std::string> & a, const std::pair<int,std::string> & b) {
                    return a.first != b.first ? a.first > b.first : a.second < b.second; });
                std::string rq = "\nWhat do the notes say about " + RARE;
                for (size_t i = 0; i < cands.size() && i < 8; ++i) rq += ", " + cands[i].second;
                rq += "?\n";
                const auto rqi = tokenize(vocab, rq, false);
                J->pre_handoff(rqi.data(), (int32_t) rqi.size());
                int nres = 0;
                for (int32_t t : J->last_sel_trig) if (J->band_of.count(t)) nres++;
                FILE * f = fopen(o.t27_dump.c_str(), "w");
                if (!f) { fprintf(stderr, "T27: cannot write %s\n", o.t27_dump.c_str()); return 1; }
                fprintf(f, "P %d %d %d %d %d %.17g %d %d %d %d\n", tp.win_k, tp.S, J->last_hi, tp.n_pin, tp.post_cap,
                        tp.trig_surp, (int32_t) (tp.trig_frac * tp.S), (int) tp.use_roll, tp.n_roll, (int) okx);
                for (const auto & kv : J->last_q) fprintf(f, "Q %llu %.17g\n", (unsigned long long) kv.first, (double) kv.second);
                for (const auto & ev : J->wocc_log)
                    fprintf(f, "O %llu %d %.17g %s\n", (unsigned long long) ev.h, ev.ws, ev.surp, ev.w.c_str());
                for (const auto & m : J->last_win) {
                    fprintf(f, "W %d %d %.17g", m.lo, m.hi, m.score);
                    for (uint64_t h2 : m.terms) fprintf(f, " %llu", (unsigned long long) h2);
                    fprintf(f, "\n");
                }
                fprintf(f, "A"); for (int32_t i : J->last_win_order) fprintf(f, " %d", i); fprintf(f, "\n");
                fprintf(f, "N %d\n", J->last_win_taken);
                fprintf(f, "T"); for (int32_t t : J->last_sel_trig) fprintf(f, " %d", t); fprintf(f, "\n");
                fprintf(f, "R %d %zu\n", nres, J->last_sel_trig.size());
                fclose(f);
                printf("TEST T27  DUMP  %s: query '%s' -> %zu terms; %zu words logged; %zu merged windows, %d admitted; "
                       "%zu tokens admitted, %d resident in the band; far_hi %d (verdict: t27check.py)\n",
                       o.t27_dump.c_str(), rq.substr(1, rq.size() - 2).c_str(), J->last_q.size(), J->wocc_log.size(),
                       J->last_win.size(), J->last_win_taken, J->last_sel_trig.size(), nres, J->last_hi);
                fflush(stdout);
                X.close();
            }
        }

        // ---- T28 / T29 (09-24 v2bld24-perf) ------------------------------------------------------
        // T28: asynchronous capture is exact.  Two design contexts prefill the same below-range document
        // (every block start < P_on: the batched region), capasync=1 and capasync=0; every token's K and
        // V in the host store must be byte-identical, and so must the logits.  T29: dropping the unused
        // Q/K/V graph outputs changes nothing: kvcap=0 vs kvcap=1 logits byte-identical.
        {
            const int32_t Lb2 = ((E.ip.W + E.ip.S + E.ip.n_pin + 1) / E.ip.B) * E.ip.B;
            const int32_t n_sub = std::min<int32_t>((int32_t) HAY.size(), Lb2);
            std::vector<llama_token> sub(HAY.begin(), HAY.begin() + n_sub);
            const int nv = llama_vocab_n_tokens(vocab);
            // 09-25 v2bld26: the batched path is not the compiled default any more (feed 128), so the test sets
            // it: the capture under test is the performance path's (feed 1024 unless a larger feed is given)
            const int32_t tfeed = E.ip.feed > E.ip.B ? E.ip.feed : 1024;
            engine A; A.inf = true; A.ip = E.ip; A.ip.feed = tfeed; A.ip.capasync = true;  A.ip.kvcap = false; A.open(model, cp);
            const bool okA = A.eval(sub.data(), n_sub, true);
            std::vector<float> lA(llama_get_logits_ith(A.ctx, -1), llama_get_logits_ith(A.ctx, -1) + nv);
            engine Sy; Sy.inf = true; Sy.ip = E.ip; Sy.ip.feed = tfeed; Sy.ip.capasync = false; Sy.ip.kvcap = false; Sy.open(model, cp);
            const bool okS = Sy.eval(sub.data(), n_sub, true);
            std::vector<float> lS(llama_get_logits_ith(Sy.ctx, -1), llama_get_logits_ith(Sy.ctx, -1) + nv);
            int64_t kbad = 0, vbad = 0;
            const size_t tb = A.I->host.tok_bytes;
            for (int32_t t = 0; t < n_sub; ++t) {
                if (memcmp(A.I->host.k(t), Sy.I->host.k(t), tb) != 0) kbad++;
                if (memcmp(A.I->host.v(t), Sy.I->host.v(t), tb) != 0) vbad++;
            }
            const bool lsame = memcmp(lA.data(), lS.data(), sizeof(float) * (size_t) nv) == 0;
            const long long fa = (long long) A.I->n_fast_tok, ca = (long long) A.I->n_cap_async, fs = (long long) Sy.I->n_fast_tok;
            const long long cs = (long long) Sy.I->n_cap_async;
            const long long ncA = (long long) A.I->host.n_tok;
            Sy.close();
            T("T28", okA && okS && kbad == 0 && vbad == 0 && lsame && fa == n_sub && ca > 0 && fs == n_sub && cs == 0 && tfeed > E.ip.B,
              "async capture: host store == synchronous capture, byte for byte",
              "T " + std::to_string(n_sub) + " (feed " + std::to_string(tfeed) + "): batched tokens " + std::to_string(fa) + "/" + std::to_string(fs)
              + ", async captures " + std::to_string(ca) + " vs " + std::to_string(cs) + "; K rows differ " + std::to_string(kbad)
              + ", V rows differ " + std::to_string(vbad) + " of " + std::to_string(n_sub) + " tokens (host n_tok " + std::to_string(ncA)
              + "); logits " + (lsame ? "identical" : "DIFFER"));
            engine K; K.inf = true; K.ip = E.ip; K.ip.feed = tfeed; K.ip.kvcap = true; K.open(model, cp);
            const bool okK = K.eval(sub.data(), n_sub, true);
            std::vector<float> lK(llama_get_logits_ith(K.ctx, -1), llama_get_logits_ith(K.ctx, -1) + nv);
            K.close();
            int nd = 0; double mx = 0;
            for (int i = 0; i < nv; ++i) { if (memcmp(&lA[i], &lK[i], sizeof(float)) != 0) nd++; mx = std::max(mx, fabs((double) lA[i] - (double) lK[i])); }
            T("T29", okK && nd == 0, "no Q/K/V graph outputs (kvcap=0) == kvcap=1, logits byte for byte",
              "T " + std::to_string(n_sub) + ": " + std::to_string(nd) + "/" + std::to_string(nv) + " logits differ, max |d| " + S(mx));
            A.close();
        }

        // ---- T30 (09-25 v2bld32): the trunk width W is a LOWER bound under batched above-range prefill -------------
        // A design context prefills P_on + 5,000 tokens with feed 1024 and feedwin 1.  Asserted: the batched path ran
        // above P_on; every position of [p - W, p) is resident (the trunk is at least W); no position of
        // [n_pin + 1, p - W - feed) is resident (sampled every 61 tokens: nothing older than one batch past the lower
        // bound survives the roll); the host store equals the cache on the trunk (band_check, 0 bad); host n_tok == p.
        {
            const int32_t P_on30 = E.ip.W + E.ip.S + E.ip.n_pin + 2;
            const int32_t n30 = std::min<int32_t>((int32_t) ids.size(), ((P_on30 + 5000) / E.ip.B) * E.ip.B);
            engine F; F.inf = true; F.ip = E.ip; F.ip.feed = E.ip.feed > E.ip.B ? E.ip.feed : 1024; F.ip.feed_win = true;
            F.open(model, cp);
            const bool ok30 = n30 > P_on30 && F.eval(ids.data(), n30, true);
            inf_ctx * I30 = F.I;
            const int32_t pp = I30->n_past, W30 = I30->p.W, fd = std::min(I30->p.feed, 2048);
            int miss_trunk = 0, stale = 0, sampled = 0;
            uint32_t cell = 0;
            for (int32_t t = pp - W30; t < pp; ++t) if (!llama_memory_kv_cells_of(I30->mem, I30->seq, t, t + 1, &cell)) miss_trunk++;
            for (int32_t t = I30->p.n_pin + 1; t < pp - W30 - fd; t += 61) { sampled++; if (llama_memory_kv_cells_of(I30->mem, I30->seq, t, t + 1, &cell)) stale++; }
            const int64_t tk0 = I30->n_bchk_tr_k, tv0 = I30->n_bchk_tr_v, tl0 = I30->n_bchk_tr_live;
            I30->band_check("T30", pp, true);
            const int64_t hb = (I30->n_bchk_tr_k - tk0) + (I30->n_bchk_tr_v - tv0), tl = I30->n_bchk_tr_live - tl0;
            const long long fa = (long long) I30->n_fast_above, nt = (long long) I30->host.n_tok;
            F.close();
            T("T30", ok30 && fa > 0 && miss_trunk == 0 && stale == 0 && hb == 0 && tl >= W30 && nt == pp,
              "above P_on the trunk is at least W (batched feeding, roll per batch)",
              "T " + std::to_string(n30) + " (P_on " + std::to_string(P_on30) + ", feed " + std::to_string(fd) + "): batched above P_on " + std::to_string(fa)
              + " tokens; [p-W,p) missing " + std::to_string(miss_trunk) + "; resident below p-W-feed " + std::to_string(stale) + "/" + std::to_string(sampled)
              + " sampled; trunk live " + std::to_string(tl) + " (>= W " + std::to_string(W30) + "), host==cache bad " + std::to_string(hb) + "; host n_tok " + std::to_string(nt));
        }

        // ---- T12: BELOW L the design must be the dense model, byte for byte ----------------
        // This is also acceptance test (4).  A document that fits inside the trained range must
        // produce zero gathers and logits identical to --inf off.
        // 09-24: across the below-range span, not one length: under window-only prefill the old
        // path evicted past W from ~8.3K on, so 9.3K and 16.4K are the lengths that expose it.
        const int32_t Lb = E.ip.W + E.ip.S + E.ip.n_pin + 1;   // the last dense length (P_on - 1)
        for (const int32_t n_sub : { E.ip.W + E.ip.n_pin + 1000, (E.ip.W + Lb) / 2, std::min(L_ctx - 1024, Lb) })
        {
            std::vector<llama_token> sub(HAY.begin(), HAY.begin() + n_sub);
            const int nv = llama_vocab_n_tokens(vocab);

            engine D; D.inf = false;   // stock llama.cpp, dense over the whole context
            D.open(model, cp);
            D.eval(sub.data(), n_sub, true);
            std::vector<float> la(llama_get_logits_ith(D.ctx, -1),
                                  llama_get_logits_ith(D.ctx, -1) + nv);
            D.close();

            engine G; G.inf = true; G.ip = E.ip;
            G.open(model, cp);
            G.eval(sub.data(), n_sub, true);
            const float * lb = llama_get_logits_ith(G.ctx, -1);
            double mx = 0; int nbytediff = 0;
            for (int i = 0; i < nv; ++i) {
                const double d = fabs((double) la[i] - (double) lb[i]);
                if (d > mx) mx = d;
                if (memcmp(&la[i], &lb[i], sizeof(float)) != 0) nbytediff++;
            }
            const int64_t g = G.I->n_gather;
            const bool identical = (nbytediff == 0);
            T("T12", g == 0 && identical, "below L: zero gathers and logits identical to dense",
              "T " + std::to_string(n_sub) + ": " + S((double) g) + " gathers, " + S((double) nbytediff) + "/" + S((double) nv)
              + " logits differ, max |d| " + S(mx));
            G.close();
        }

        printf("\nSELFTEST\tpass=%d\tfail=%d\n", npass, nfail);
        return nfail ? 2 : 0;
    }

    const auto pre_ids = tokenize(vocab, PRE, true);
    const double t_all = now_s();
    const int n_pred_needle = 10;

    for (const auto & task : tasks) {
        const int nkeys = (task == "niah1" || task.rfind("hop", 0) == 0) ? 1 : 3;
        // key3r: the DISCRIMINATION row.  Three same-format needles with RARE handles at
        // different depths; the question names one.  Single-needle measures arrival, this
        // measures discrimination.  key3rc asks about a fourth handle that is NOT in the
        // document at all: the frame control, where any number emitted is a false positive.
        const bool k3   = (task == "key3r" || task == "key3rc");
        const bool k3c  = (task == "key3rc");
        for (int e = 0; e < o.nex; ++e) {
            // ---------------------------------------------------------------------------------
            // TWO-HOP.  The question contains no rare word, so the block-boundary gather has
            // nothing to trigger on and the far unit is never fetched: the digits fail by
            // construction.  The bridge sentence sits inside the trunk, so the model can read it
            // and emit <word> itself -- and THAT generated word is what the event path triggers
            // on, mid-block, in decode.  The far unit then arrives before the digits are due.
            // ---------------------------------------------------------------------------------
            // hop2t is the TASK-VALIDITY control: identical text, but the "far" sentence is put
            // inside the trunk, so no retrieval is involved at all.  If the model cannot answer
            // hop2t, the question is unanswerable and a hop2 score of 0 says nothing about the
            // event path.  Every measurement of hop2 should be read against its hop2t control.
            // ---------------------------------------------------------------------------------
            // CHAINED HOP (hop3).  The only task that can license the event tier as built.
            //   bridge, in the trunk:      " The item we discussed is <A>."
            //   far unit A:                " The partner of <A> is <B>."
            //   far unit B, far from A:    " The digits recorded for <B> are <7 digits>."
            //   question, containing neither A nor B's name as a handle to B
            // The seed reaches unit A because <A> is in the trunk.  <B> appears NOWHERE in the
            // prompt or the trunk, so unit B cannot be fetched by any gather at the hand-off; it
            // can only arrive when the model GENERATES <B> mid-answer.  So OFF must fail on the
            // digits and ON must not, and that is the events-only case hop2 could not provide.
            // hop3t is the validity control: both far sentences placed inside the trunk.
            // ---------------------------------------------------------------------------------
            // ---------------------------------------------------------------------------------
            // 09-24 EVENTS-ONLY TWIN (evtwin).  One far needle with a rare handle, placed below the
            // back-fill floor; the question names the handle (the lexical twin) and asks for a
            // name-first answer, so the model GENERATES the handle before the digits.  Run with
            // reqfetch=0 the request fetches nothing (no request words at the hand-off, no events
            // on the request's own words), so the unit can arrive only through the event on the
            // generated handle: the llama.cpp twin of conceptval P.12 arm E (evreenc=0) and Ep
            // (evreenc=1).  With reqfetch=1 it is the reference R (the pre-encode hand-off fetches
            // the unit from the question).  evtwint = the needle inside the trunk (validity).
            // ---------------------------------------------------------------------------------
            // evtwinc (09-24 v2bld15) adds the harness's condition: ONE worked example with its own
            // code, in the trunk just before the question (cv_think's scratch prompt carries one;
            // its code took ~3x the needle's attention in the failures).  The example sits in the
            // history, so no arm fetches anything for it.
            if (task == "evtwin" || task == "evtwint" || task == "evtwinc") {
                const bool trunk_ctl = (task == "evtwint");
                const bool compc     = (task == "evtwinc");
                const std::string word = RAREW[(rng() % 24)];
                const std::string val  = std::to_string(1000000 + (int)(rng() % 9000000));
                std::vector<llama_token> cex_ids;
                if (compc) {
                    std::string C = RAREW[(rng() % 24)];
                    while (C == word) C = RAREW[(rng() % 24)];
                    const std::string CM = std::to_string(1000000 + (int)(rng() % 9000000));
                    cex_ids = tokenize(vocab,
                        "\nHere is an example of the question and its answer.\n"
                        "What is the special magic number for " + C + " mentioned in the provided text?\n"
                        "The special magic number for " + C + " mentioned in the provided text is " + CM + ".\n", false);
                }
                const auto nd_ids = tokenize(vocab, " One of the special magic numbers for " + word + " is: " + val + ".", false);
                // 09-24 v2bld14: the first frame ("Answer in the form: The item is X. Its number is Y.
                // The item is") failed its premise -- with no needle visible Mistral never names the
                // handle (ETW_M32_E0/E1 328371/2: word 0, events 0).  The answer frame now stops at
                // "The special magic number for", so the handle is the model's first generated word
                // (a copy from the question) and the digits follow a few tokens later.
                const auto qe_ids = tokenize(vocab,
                    "\nWhat is the special magic number for " + word + " mentioned in the provided text?\n"
                    "The special magic number for", false);
                int budget = o.T - (int) pre_ids.size() - (int) qe_ids.size() - (int) nd_ids.size() - (int) cex_ids.size();
                if (budget <= 4096 || budget >= (int) HAY.size()) { fprintf(stderr, "bad evtwin budget\n"); return 1; }
                const int start = (int)(rng() % (HAY.size() - budget));
                const int floor_t = o.T - E.ip.W - E.ip.S - 256;
                if (!trunk_ctl && floor_t < 1024) { fprintf(stderr, "evtwin: no room below the back-fill floor\n"); return 1; }
                const int dfar = trunk_ctl ? (budget - 2000)
                                           : (int)((0.05 + 0.85*(rng()/(double) rng.max())) * floor_t);
                std::vector<llama_token> prompt = pre_ids;
                prompt.insert(prompt.end(), HAY.begin()+start,        HAY.begin()+start+dfar);
                const int pos_far = (int) prompt.size();
                prompt.insert(prompt.end(), nd_ids.begin(), nd_ids.end());
                prompt.insert(prompt.end(), HAY.begin()+start+dfar,   HAY.begin()+start+budget);
                if (compc) prompt.insert(prompt.end(), cex_ids.begin(), cex_ids.end());   // history, in the trunk
                const int q_markE = (int) prompt.size();          // the turn boundary
                prompt.insert(prompt.end(), qe_ids.begin(), qe_ids.end());

                if (e < o.ex0 || done.count(task + "/" + std::to_string(e))) continue;   // 09-24 --ex0: sections of one item stream
                pause_gate("before evtwin");
                const double t0 = now_s();
                E.open(model, cp);
                if (!E.eval_turn(prompt.data(), prompt.size(), (int) prompt.size() - q_markE, true)) { E.close(); return 1; }
                // residency of the needle at the first generated token: 0 under reqfetch=0 is the
                // premise of the events-only row
                int res0 = -1, ntok0 = (int) nd_ids.size();
                if (E.inf) { const inf_probe pb0 = E.I->probe(pos_far, pos_far + ntok0); res0 = pb0.n_band + pb0.trunk_n; }
                E.set_decode(true);
                const int64_t ev0 = E.inf ? E.I->n_events_decode : 0;
                const int64_t fe0 = E.inf ? E.I->n_evt_fetch : 0;
                std::string out;
                int gen_at_fetch = -1;              // generated tokens before the first decode-time fetch
                const int nvE = llama_vocab_n_tokens(vocab);
                for (int i = 0; i < 24; ++i) {
                    const float * lg = llama_get_logits_ith(E.ctx, -1);
                    llama_token best = 0; float bv = -1e30f;
                    for (int v = 0; v < nvE; ++v) if (lg[v] > bv) { bv = lg[v]; best = v; }
                    if (llama_vocab_is_eog(vocab, best)) break;
                    out += detok(vocab, {best});
                    if (!E.eval(&best, 1, true)) break;
                    if (E.inf && gen_at_fetch < 0 && E.I->n_evt_fetch > fe0) gen_at_fetch = i + 1;
                }
                int res1 = -1;
                if (E.inf) { const inf_probe pb1 = E.I->probe(pos_far, pos_far + ntok0); res1 = pb1.n_band + pb1.trunk_n; }
                const double secs = now_s() - t0;
                const int em_dig  = out.find(val)  != std::string::npos;
                const int em_word = out.find(word) != std::string::npos;
                const int dist    = (int) prompt.size() - pos_far;
                std::string o1 = out; for (auto & c : o1) if (c == '\n' || c == '\r') c = '/';
                char line[1400];
                snprintf(line, sizeof(line),
                         "  %-7s ex %2d  needle %6d back  EM %d  word %d  gold %s (%s)  pred %-44s  "
                         "(%.0fs, %.1f tok/s, design %s, needle resident %d/%d at gen0, %d/%d at end, decode events %lld, "
                         "fetched %lld, first fetch after gen %d, RSS %.1f GiB)  REENC n %lld tok %lld ms %.1f evreenc %d budget %d skip %lld surpskip %lld evrsurp %.0f\n",
                         task.c_str(), e, dist, em_dig, em_word, val.c_str(), word.c_str(), o1.c_str(),
                         secs, prompt.size()/secs, design, res0, ntok0, res1, ntok0,
                         (long long)((E.inf ? E.I->n_events_decode : 0) - ev0),
                         (long long)((E.inf ? E.I->n_evt_fetch : 0) - fe0), gen_at_fetch,
                         host_rss()/1073741824.0,
                         (long long)(E.inf ? E.I->n_reenc : 0), (long long)(E.inf ? E.I->n_reenc_tok : 0),
                         E.inf ? 1000.0*E.I->t_reenc : 0.0, (int) E.ip.evreenc, E.ip.evr_budget,
                         (long long)(E.inf ? E.I->n_reenc_budget_skip : 0),
                         (long long)(E.inf ? E.I->n_reenc_surp_skip : 0), E.ip.evr_surp);
                printf("%s", line); fflush(stdout);
                if (fout) { fprintf(fout, "%s", line); fflush(fout); }
                if (e == 0 && E.inf && o.dump_n > 0) E.I->report(o.tag.c_str());
                rows.push_back({task, dist, em_dig, em_word});
                E.close();
                continue;
            }
            if (task == "hop3" || task == "hop3t") {
                const bool trunk_ctl = (task == "hop3t");
                const std::string A = RAREW[(rng() % 24)];
                std::string B = RAREW[(rng() % 24)];
                while (B == A) B = RAREW[(rng() % 24)];
                const std::string val = std::to_string(1000000 + (int)(rng() % 9000000));
                const auto uA = tokenize(vocab, " The partner of " + A + " is " + B + ".", false);
                const auto uB = tokenize(vocab, " The digits recorded for " + B + " are " + val + ".", false);
                const auto br = tokenize(vocab, " The item we discussed is " + A + ".", false);
                const auto q3 = tokenize(vocab,
                    "\nWhat number goes with the partner of the item we discussed?\n"
                    "Answer in the form: The item is X. Its partner is Y. The number is Z.\n"
                    "The item is", false);
                int budget = o.T - (int) pre_ids.size() - (int) q3.size()
                                 - (int) uA.size() - (int) uB.size() - (int) br.size();
                if (budget <= 8192 || budget >= (int) HAY.size()) { fprintf(stderr, "bad hop3 budget\n"); return 1; }
                const int start = (int)(rng() % (HAY.size() - budget));
                const int floor_t = o.T - E.ip.W - E.ip.S - 256;
                const int dbr = budget - 2000;                       // the bridge, inside the trunk
                int dA, dB;
                if (trunk_ctl) { dA = dbr - 1200; dB = dbr - 600; }  // control: both in the trunk
                else {
                    if (floor_t < 4096) { fprintf(stderr, "hop3: no room below the back-fill floor\n"); return 1; }
                    dA = (int)((0.05 + 0.35*(rng()/(double) rng.max())) * floor_t);
                    dB = (int)((0.55 + 0.35*(rng()/(double) rng.max())) * floor_t);
                }
                if (!(dA + 64 < dB && dB + 64 < dbr)) { fprintf(stderr, "hop3 geometry\n"); return 1; }

                std::vector<llama_token> prompt = pre_ids;
                prompt.insert(prompt.end(), HAY.begin()+start,        HAY.begin()+start+dA);
                const int posA = (int) prompt.size();
                prompt.insert(prompt.end(), uA.begin(), uA.end());
                prompt.insert(prompt.end(), HAY.begin()+start+dA,     HAY.begin()+start+dB);
                const int posB = (int) prompt.size();
                prompt.insert(prompt.end(), uB.begin(), uB.end());
                prompt.insert(prompt.end(), HAY.begin()+start+dB,     HAY.begin()+start+dbr);
                prompt.insert(prompt.end(), br.begin(), br.end());
                prompt.insert(prompt.end(), HAY.begin()+start+dbr,    HAY.begin()+start+budget);
                const int q_mark3 = (int) prompt.size();          // the turn boundary
                prompt.insert(prompt.end(), q3.begin(), q3.end());

                if (e < o.ex0 || done.count(task + "/" + std::to_string(e))) continue;   // 09-24 --ex0: sections of one item stream
                pause_gate("before hop3");
                const double t0 = now_s();
                E.open(model, cp);
                if (!E.eval_turn(prompt.data(), prompt.size(), (int) prompt.size() - q_mark3, true)) { E.close(); return 1; }
                E.set_decode(true);
                const int64_t ev0 = E.inf ? E.I->n_events_decode : 0;

                std::string out;
                const int nv3 = llama_vocab_n_tokens(vocab);
                for (int i = 0; i < 40; ++i) {
                    const float * lg = llama_get_logits_ith(E.ctx, -1);
                    llama_token best = 0; float bv = -1e30f;
                    for (int v = 0; v < nv3; ++v) if (lg[v] > bv) { bv = lg[v]; best = v; }
                    if (llama_vocab_is_eog(vocab, best)) break;
                    out += detok(vocab, {best});
                    if (!E.eval(&best, 1, true)) break;
                }
                const double secs = now_s() - t0;
                const int hitA = out.find(A)   != std::string::npos;
                const int hitB = out.find(B)   != std::string::npos;   // the SECOND hop
                const int em   = out.find(val) != std::string::npos;
                std::string o1 = out; for (auto & c : o1) if (c == '\n' || c == '\r') c = '/';
                char line[1400];
                snprintf(line, sizeof(line),
                         "  %-6s ex %2d  needle %6d back  EM %d  word %d  hopA %d  gold %s (%s->%s)  "
                         "pred %-46s  (%.0fs, %.1f tok/s, decode events %lld, A@%d B@%d, RSS %.1f GiB)\n",
                         task.c_str(), e, (int) prompt.size() - posB, em, hitB, hitA, val.c_str(),
                         A.c_str(), B.c_str(), o1.c_str(), secs, prompt.size()/secs,
                         (long long)((E.inf ? E.I->n_events_decode : 0) - ev0),
                         (int) prompt.size() - posA, (int) prompt.size() - posB,
                         host_rss()/1073741824.0);
                if (E.inf) {   // 09-24 evreenc: re-encodes, tokens, ms (appended; the row format is unchanged before it)
                    const size_t L0 = strlen(line);
                    if (L0 > 0 && line[L0-1] == '\n') line[L0-1] = 0;
                    const size_t L1 = strlen(line);
                    snprintf(line + L1, sizeof(line) - L1, "  REENC n %lld tok %lld ms %.1f evreenc %d budget %d skip %lld surpskip %lld evrsurp %.0f\n",
                             (long long) E.I->n_reenc, (long long) E.I->n_reenc_tok, 1000.0*E.I->t_reenc,
                             (int) E.I->p.evreenc, E.I->p.evr_budget, (long long) E.I->n_reenc_budget_skip,
                             (long long) E.I->n_reenc_surp_skip, E.I->p.evr_surp);
                    if (E.I->p.sent_fill > 0) {   // 09-25 v2bld28 sentfill
                        const size_t L4 = strlen(line); if (L4 > 0 && line[L4-1] == '\n') line[L4-1] = 0;
                        const size_t L5 = strlen(line);
                        snprintf(line + L5, sizeof(line) - L5, "  SFILL cand %lld\n", (long long) E.I->n_sf_cand);
                    }
                    if (E.I->p.seed_sent) {   // 09-24 v2bld25 seedsent
                        const size_t L2 = strlen(line); if (L2 > 0 && line[L2-1] == '\n') line[L2-1] = 0;
                        const size_t L3 = strlen(line);
                        snprintf(line + L3, sizeof(line) - L3, "  SEEDU units %lld tok %lld hit %lld\n", (long long) E.I->n_seedu_units,
                                 (long long) E.I->n_seedu_tok, (long long) E.I->n_seedu_cand);
                    }
                }
                printf("%s", line); fflush(stdout);
                if (fout) { fprintf(fout, "%s", line); fflush(fout); }
                if (e == 0 && E.inf && o.dump_n > 0) E.I->report(o.tag.c_str());
                rows.push_back({task, (int) prompt.size() - posB, em, hitB});
                E.close();
                continue;
            }
            // hop2c / hop2ct add a NEAR COMPETITOR immediately before the question: a worked
            // example on a different item, in the same format, with its own digits.  It is the
            // direct test of "competing attention close back confuses distant reads" - the far
            // copy has to win against a same-format answer sitting a few hundred tokens away.
            if (task == "hop2" || task == "hop2t" || task == "hop2c" || task == "hop2ct") {
                const bool trunk_ctl = (task == "hop2t" || task == "hop2ct");
                const bool comp      = (task == "hop2c" || task == "hop2ct");
                const std::string word = RAREW[(rng() % 24)];
                const std::string val  = std::to_string(1000000 + (int)(rng() % 9000000));
                // "digits recorded" against the question's "number": a near-synonym, so the
                // semantic step is small, and NO content token is shared, so the question can
                // never trigger the needle's unit.  Sharing even one content token would hand the
                // index a handle - see LOG 43 for why no word in this haystack is common enough
                // to be shared safely.
                const auto nd_ids = tokenize(vocab, " The digits recorded for " + word + " are " + val + ".", false);
                const auto br_ids = tokenize(vocab, " The item we discussed is " + word + ".", false);
                // The question must share NO content word with the needle sentence, or the
                // lexical trigger finds the needle's unit from the question alone and the event
                // path is never needed.  The first version asked "what is the secret code for the
                // item we discussed", which repeats "secret code" verbatim from the needle; the
                // smoke scored 2/2 with zero events, which is how that was caught.  The question
                // now shares only the BRIDGE's words, and the bridge is in the trunk.
                std::vector<llama_token> cmp_ids;
                if (comp) {
                    std::string C = RAREW[(rng() % 24)];
                    while (C == word) C = RAREW[(rng() % 24)];
                    const std::string CM = std::to_string(1000000 + (int)(rng() % 9000000));
                    std::string ct = "\nHere is a worked example of the same kind of question.\n"
                                     " The digits recorded for " + C + " are " + CM + ".\n"
                                     " The item we discussed is " + C + ".\n"
                                     "What number goes with the item we discussed?\n"
                                     "Answer in the form: The item is X. Its number is Y.\n"
                                     "The item is " + C + ". Its number is " + CM + ".\n";
                    // pad the block out to roughly 300 tokens so the competitor is a block, not a line
                    while ((int) tokenize(vocab, ct, false).size() < 300)
                        ct += "That is how such a question is answered in this document.\n";
                    cmp_ids = tokenize(vocab, ct, false);
                }
                const auto q2_ids = tokenize(vocab,
                    // REVERTED to the wording whose hop2t control is measured (0.875 Mistral,
                    // 1.000 Llama).  The worked-example version confused the base models and took
                    // Mistral's first hop from 13/16 to 7/16 -- and I changed it WITHOUT re-running
                    // hop2t first, which is the whole point of having a control.  The template echo
                    // is handled in the SCORING instead: a row is a first-hop success only if the
                    // word is produced, and EM is reported conditional on that.
                    "\nWhat number goes with the item we discussed?\n"
                    "Answer in the form: The item is X. Its number is Y.\nThe item is", false);
                int budget = o.T - (int) pre_ids.size() - (int) q2_ids.size()
                                 - (int) nd_ids.size() - (int) br_ids.size() - (int) cmp_ids.size();
                if (budget <= 4096 || budget >= (int) HAY.size()) { fprintf(stderr, "bad hop2 budget\n"); return 1; }
                const int start = (int)(rng() % (HAY.size() - budget));
                // The band is back-filled with the most recent far tokens to keep its S cells full
                // (select() step 5), so a needle within S + W of the end is in the band whatever the
                // index does.  At 32K that covers two thirds of the far region.  Put the needle
                // strictly BELOW that floor, so reaching it requires the index and nothing else.
                const int floor_t = o.T - E.ip.W - E.ip.S - 256;
                if (floor_t < 1024) { fprintf(stderr, "hop2: no room below the back-fill floor\n"); return 1; }
                const int dbr   = budget - 2000;                                              // inside the trunk
                const int dfar  = trunk_ctl ? (dbr - 600)
                                            : (int)((0.05 + 0.85*(rng()/(double) rng.max())) * floor_t);
                if (dbr <= dfar + 64) { fprintf(stderr, "hop2 geometry\n"); return 1; }
                (void) trunk_ctl;

                std::vector<llama_token> prompt = pre_ids;
                prompt.insert(prompt.end(), HAY.begin()+start,          HAY.begin()+start+dfar);
                const int pos_far = (int) prompt.size();
                prompt.insert(prompt.end(), nd_ids.begin(), nd_ids.end());
                prompt.insert(prompt.end(), HAY.begin()+start+dfar,     HAY.begin()+start+dbr);
                const int pos_br = (int) prompt.size();
                prompt.insert(prompt.end(), br_ids.begin(), br_ids.end());
                prompt.insert(prompt.end(), HAY.begin()+start+dbr,      HAY.begin()+start+budget);
                if (comp) prompt.insert(prompt.end(), cmp_ids.begin(), cmp_ids.end());
                const int q_mark2 = (int) prompt.size();          // the turn boundary
                prompt.insert(prompt.end(), q2_ids.begin(), q2_ids.end());

                if (e < o.ex0 || done.count(task + "/" + std::to_string(e))) continue;   // 09-24 --ex0: sections of one item stream
                pause_gate("before hop2");
                const double t0 = now_s();
                E.open(model, cp);
                if (E.inf && E.I->p.oracle_seed) {
                    // the driver knows the handle; hand its word hash to the policy
                    std::string cl; for (unsigned char ch : word) if (isalnum(ch)||ch=='-'||ch=='_') cl += (char) ch;
                    uint64_t oh = 1469598103934665603ull;
                    for (unsigned char ch : cl) { if (ch>='A'&&ch<='Z') ch = ch-'A'+'a'; oh ^= ch; oh *= 1099511628211ull; }
                    E.I->oracle_h = oh;
                    printf("  ORACLESEED ex %d handle '%s' hash %016llx\n", e, word.c_str(),
                           (unsigned long long) oh); fflush(stdout);
                }
                if (o.qk_scan > 0 && E.inf && e < o.qk_scan) {
                    // Per-token scan over the BRIDGE sentence as well as the question: does the
                    // bridge's own encoding read far unit A, when the bridge sits in the trunk?
                    if (!E.eval(prompt.data(), pos_br, false)) { E.close(); return 1; }
                    const int a0 = pos_far, a1 = pos_far + (int) nd_ids.size();
                    for (int bi = pos_br; bi < (int) br_ids.size() + pos_br; ++bi) {
                        if (!E.eval(prompt.data() + bi, 1, false)) { E.close(); return 1; }
                        char tg[128];
                        snprintf(tg, sizeof(tg), "Btok %d '%s'", bi - pos_br, E.I->piece[prompt[bi]].c_str());
                        E.I->qk_mass(a0, a1, 0, 0, tg, 0);
                    }
                    const int rest = q_mark2 - (pos_br + (int) br_ids.size());
                    if (rest > 0 && !E.eval(prompt.data() + pos_br + (int) br_ids.size(), rest, false)) { E.close(); return 1; }
                    if (E.I->p.prehand) E.I->pre_handoff(prompt.data() + q_mark2, (int) prompt.size() - q_mark2);
                    for (int qi = q_mark2; qi < (int) prompt.size(); ++qi) {
                        if (!E.eval(prompt.data() + qi, 1, qi + 1 == (int) prompt.size())) { E.close(); return 1; }
                        char tg[128];
                        snprintf(tg, sizeof(tg), "Qtok %d '%s'", qi - q_mark2, E.I->piece[prompt[qi]].c_str());
                        E.I->qk_mass(a0, a1, 0, 0, tg, 0);
                    }
                } else
                if (!E.eval_turn(prompt.data(), prompt.size(), (int) prompt.size() - q_mark2, true)) { E.close(); return 1; }
                // 09-25 v2bld27 H2DIAG (read-only; LOG 137e): the needle's residency at the question, its sentence
                // unit(s), and the handle's surprisal, so every hop2 miss can be classified
                const int nnd = (int) nd_ids.size();
                int h2_res0 = -1;
                inf_probe h2p0{};
                if (E.inf) { h2p0 = E.I->probe(pos_far, pos_far + nnd); h2_res0 = h2p0.n_band + h2p0.trunk_n; }
                E.set_decode(true);
                const int64_t ev0 = E.inf ? E.I->n_events_decode : 0;
                const int64_t ec0 = E.inf ? E.I->n_event_cells_decode : 0;

                std::string out;
                const int nv2 = llama_vocab_n_tokens(vocab);
                for (int i = 0; i < 28; ++i) {
                    const float * lg = llama_get_logits_ith(E.ctx, -1);
                    llama_token best = 0; float bv = -1e30f;
                    for (int v = 0; v < nv2; ++v) if (lg[v] > bv) { bv = lg[v]; best = v; }
                    if (llama_vocab_is_eog(vocab, best)) break;
                    out += detok(vocab, {best});
                    if (!E.eval(&best, 1, true)) break;
                }
                if (E.inf) {   // 09-25 v2bld27 H2DIAG (read-only)
                    const inf_probe h2p1 = E.I->probe(pos_far, pos_far + nnd);
                    const int h2_res1 = h2p1.n_band + h2p1.trunk_n;
                    // handle and digit token offsets inside the needle
                    const std::string ndt = " The digits recorded for " + word + " are " + val + ".";
                    const size_t wpos = ndt.find(word), dpos = ndt.find(val);
                    int hs = -1, ds = -1; size_t acc = 0;
                    for (int i = 0; i < nnd; ++i) {
                        const size_t a0 = acc; acc += detok(vocab, {nd_ids[i]}).size();
                        if (hs < 0 && acc > wpos) hs = i;
                        if (ds < 0 && acc > dpos) ds = i;
                        (void) a0;
                    }
                    auto unit_of = [&](int32_t t) {
                        int32_t lo = 0, hi = (int32_t) E.I->units.size();
                        while (lo < hi) { const int32_t md = (lo + hi)/2; if (E.I->units[md].t1 <= t) lo = md + 1; else hi = md; }
                        return (lo < (int32_t) E.I->units.size() && E.I->units[lo].t0 <= t) ? lo : -1; };
                    const int32_t uh = hs >= 0 ? unit_of(pos_far + hs) : -1, ud = ds >= 0 ? unit_of(pos_far + ds) : -1;
                    const int32_t ua = unit_of(pos_far), ub = unit_of(pos_far + nnd - 1);
                    const int nunits = (ua >= 0 && ub >= 0) ? ub - ua + 1 : -1;
                    int cap = 0; std::string us;
                    for (int32_t u = ua; u >= 0 && u <= ub; ++u) {
                        const int32_t L = E.I->units[u].t1 - E.I->units[u].t0;
                        if (L >= E.I->p.unit_max) cap++;
                        char b2[64]; snprintf(b2, sizeof(b2), "%s[%d,%d)%d", us.empty() ? "" : " ", E.I->units[u].t0, E.I->units[u].t1, L); us += b2;
                    }
                    std::string cl; for (unsigned char ch : word) if (isalnum(ch)||ch=='-'||ch=='_') cl += (char) ch;
                    uint64_t hh = 1469598103934665603ull;
                    for (unsigned char ch : cl) { if (ch>='A'&&ch<='Z') ch = ch-'A'+'a'; hh ^= ch; hh *= 1099511628211ull; }
                    double s_br = -1, s_nd = -1; int32_t t_br = -1;
                    for (const auto & e2 : E.I->wlog) if (e2.h == hh) {
                        if (e2.t >= pos_br && e2.t < pos_br + (int32_t) br_ids.size() + 2) { s_br = e2.surp; t_br = e2.t; }
                        if (e2.t >= pos_far && e2.t < pos_far + nnd + 2) s_nd = e2.surp;
                    }
                    printf("  H2DIAG ex %2d  res0 %d/%d  res1 %d/%d  sel0 %d forced0 %d  needle_units %d%s  handle_unit %d digits_unit %d same %d  cap %d  "
                           "handle '%s' hash %016llx surp_bridge %.1f (t %d) surp_needle %.1f  word %d EM %d\n",
                           e, h2_res0, nnd, h2_res1, nnd, h2p0.n_sel, h2p0.n_forced, nunits, (" " + us).c_str(), uh, ud, (int) (uh >= 0 && uh == ud),
                           cap, word.c_str(), (unsigned long long) hh, s_br, t_br, s_nd,
                           (int) (out.find(word) != std::string::npos), (int) (out.find(val) != std::string::npos));
                    fflush(stdout);
                }
                const double secs = now_s() - t0;
                const int em_dig  = out.find(val)  != std::string::npos;
                const int em_word = out.find(word) != std::string::npos;
                const int dist    = (int) prompt.size() - pos_far;
                std::string o1 = out; for (auto & c : o1) if (c == '\n' || c == '\r') c = '/';
                char line[1200];
                snprintf(line, sizeof(line),
                         "  %-6s ex %2d  needle %6d back  EM %d  word %d  gold %s (%s)  pred %-40s  "
                         "(%.0fs, %.1f tok/s, design %s, decode events %lld, cells %lld, bridge@%d, RSS %.1f GiB)  "
                         "VOL sel_tok %lld gathers %lld new_cells %lld\n",
                         task.c_str(), e, dist, em_dig, em_word, val.c_str(), word.c_str(), o1.c_str(),
                         secs, prompt.size()/secs, design,
                         (long long)((E.inf ? E.I->n_events_decode : 0) - ev0),
                         (long long)((E.inf ? E.I->n_event_cells_decode : 0) - ec0),
                         (int) prompt.size() - pos_br, host_rss()/1073741824.0,
                         (long long)(E.inf ? E.I->n_sel_real : 0), (long long)(E.inf ? E.I->n_gather : 0),
                         (long long)(E.inf ? E.I->n_new_cells : 0));
                if (E.inf) {   // 09-24 evreenc (appended)
                    const size_t L0 = strlen(line);
                    if (L0 > 0 && line[L0-1] == '\n') line[L0-1] = 0;
                    const size_t L1 = strlen(line);
                    snprintf(line + L1, sizeof(line) - L1, "  REENC n %lld tok %lld ms %.1f evreenc %d budget %d skip %lld surpskip %lld evrsurp %.0f\n",
                             (long long) E.I->n_reenc, (long long) E.I->n_reenc_tok, 1000.0*E.I->t_reenc,
                             (int) E.I->p.evreenc, E.I->p.evr_budget, (long long) E.I->n_reenc_budget_skip,
                             (long long) E.I->n_reenc_surp_skip, E.I->p.evr_surp);
                    if (E.I->p.sent_fill > 0) {   // 09-25 v2bld28 sentfill
                        const size_t L4 = strlen(line); if (L4 > 0 && line[L4-1] == '\n') line[L4-1] = 0;
                        const size_t L5 = strlen(line);
                        snprintf(line + L5, sizeof(line) - L5, "  SFILL cand %lld\n", (long long) E.I->n_sf_cand);
                    }
                    if (E.I->p.seed_sent) {   // 09-24 v2bld25 seedsent
                        const size_t L2 = strlen(line); if (L2 > 0 && line[L2-1] == '\n') line[L2-1] = 0;
                        const size_t L3 = strlen(line);
                        snprintf(line + L3, sizeof(line) - L3, "  SEEDU units %lld tok %lld hit %lld\n", (long long) E.I->n_seedu_units,
                                 (long long) E.I->n_seedu_tok, (long long) E.I->n_seedu_cand);
                    }
                }
                printf("%s", line); fflush(stdout);
                if (fout) { fprintf(fout, "%s", line); fflush(fout); }
                if (e == 0 && E.inf && o.dump_n > 0) E.I->report(o.tag.c_str());
                rows.push_back({task, dist, em_dig, em_word});
                E.close();
                continue;
            }

            std::vector<std::string> keys, vals;
            while ((int) keys.size() < nkeys) {
                std::string k = k3 ? std::string(RAREW[rng() % 24])
                                   : std::string(ADJ[rng() % 24]) + "-" + NOUN[rng() % 24];
                if (std::find(keys.begin(), keys.end(), k) == keys.end()) keys.push_back(k);
            }
            std::string absent;                       // the fourth handle, never in the document
            if (k3) {
                do { absent = RAREW[rng() % 24]; }
                while (std::find(keys.begin(), keys.end(), absent) != keys.end());
            }
            for (int i = 0; i < nkeys; ++i) vals.push_back(std::to_string(1000000 + (int)(rng() % 9000000)));
            const int ask = (int)(rng() % nkeys);
            const std::string named = k3c ? absent : keys[ask];
            const std::string query =
                "\nWhat is the special magic number for " + named + " mentioned in the provided text?\n"
                "The special magic number for " + named + " mentioned in the provided text is";
            const auto q_ids = tokenize(vocab, query, false);
            std::vector<std::vector<llama_token>> n_ids;
            for (int i = 0; i < nkeys; ++i)
                n_ids.push_back(tokenize(vocab, " One of the special magic numbers for " + keys[i] + " is: " + vals[i] + ".", false));

            int budget = o.T - (int) pre_ids.size() - (int) q_ids.size();
            for (auto & n : n_ids) budget -= (int) n.size();
            if (budget <= 0 || budget >= (int) HAY.size()) { fprintf(stderr, "bad budget\n"); return 1; }
            const int start = (int)(rng() % (HAY.size() - budget));

            std::vector<double> depths;
            if (k3) {
                // the three needles must be at least 16K tokens apart, so a hit on one cannot be
                // a neighbourhood hit on another
                const double sep = 16384.0 / (double) std::max(1, budget);
                for (int tries = 0; tries < 200; ++tries) {
                    depths.clear();
                    for (int i = 0; i < nkeys; ++i) depths.push_back((rng()/(double) rng.max()) * 0.95);
                    std::sort(depths.begin(), depths.end());
                    bool ok = true;
                    for (int i = 1; i < nkeys; ++i) if (depths[i] - depths[i-1] < sep) ok = false;
                    if (ok) break;
                }
                if (depths.empty() || depths.back() - depths.front() < 2*sep) {
                    for (int i = 0; i < nkeys; ++i) depths[i] = 0.05 + 0.45*i;   // forced spacing
                }
            } else {
                for (int i = 0; i < nkeys; ++i) depths.push_back((rng()/(double) rng.max()) * 0.95);
                std::sort(depths.begin(), depths.end());
            }
            std::vector<int> order(nkeys); for (int i = 0; i < nkeys; ++i) order[i] = i;
            std::shuffle(order.begin(), order.end(), rng);
            std::vector<std::pair<int,int>> ins;
            for (int i = 0; i < nkeys; ++i) ins.push_back({(int)(depths[order[i]] * budget), i});
            std::sort(ins.begin(), ins.end());

            std::vector<llama_token> prompt = pre_ids;
            std::vector<int> pos(nkeys, 0);
            int cur = 0;
            for (auto & pr : ins) {
                prompt.insert(prompt.end(), HAY.begin() + start + cur, HAY.begin() + start + pr.first);
                pos[pr.second] = (int) prompt.size();
                prompt.insert(prompt.end(), n_ids[pr.second].begin(), n_ids[pr.second].end());
                cur = pr.first;
            }
            prompt.insert(prompt.end(), HAY.begin() + start + cur, HAY.begin() + start + budget);
            const int q_mark = (int) prompt.size();          // the turn boundary
            prompt.insert(prompt.end(), q_ids.begin(), q_ids.end());

            if (e < o.ex0 || done.count(task + "/" + std::to_string(e))) continue;   // 09-24 --ex0: sections of one item stream
            pause_gate("before needle");
            const double t0 = now_s();
            E.open(model, cp);
            if (o.qk_scan > 0 && E.inf && e < o.qk_scan) {
                // Per-token QK scan: feed the history, run the hand-off, then feed the question ONE
                // TOKEN AT A TIME and report the far mass each token puts on the needle's unit.
                // This says WHERE the far unit is read - the question's frame words, the handle,
                // the answer prefix, or only the digits.
                if (!E.eval(prompt.data(), q_mark, false)) { E.close(); return 1; }
                if (E.I->p.prehand) E.I->pre_handoff(prompt.data() + q_mark, (int) prompt.size() - q_mark);
                const int d0s = pos[ask], d1s = pos[ask] + (int) n_ids[ask].size();
                // Mark the question's NOUN PHRASE - the handle the question asks about - so the
                // report says plainly which question tokens read the unit.  The harness's twin
                // reads when the unit is visible from the query-block start, the first frame word,
                // or the handle, and not when it becomes visible only at the last question token.
                const auto hid = tokenize(vocab, keys[ask], false);
                std::vector<char> is_np(prompt.size() - q_mark, 0);
                for (int qi = q_mark; qi + (int) hid.size() <= (int) prompt.size(); ++qi) {
                    bool hit = true;
                    for (int b3 = 0; b3 < (int) hid.size(); ++b3)
                        if (prompt[qi + b3] != hid[b3]) { hit = false; break; }
                    if (hit) for (int b3 = 0; b3 < (int) hid.size(); ++b3) is_np[qi - q_mark + b3] = 1;
                }
                for (int qi = q_mark; qi < (int) prompt.size(); ++qi) {
                    if (!E.eval(prompt.data() + qi, 1, qi + 1 == (int) prompt.size())) { E.close(); return 1; }
                    char tg[160];
                    snprintf(tg, sizeof(tg), "Qtok %d/%d %-6s '%s'", qi - q_mark,
                             (int) prompt.size() - q_mark,
                             is_np[qi - q_mark] ? "[NP]" : (qi == q_mark ? "[START]" : "[frame]"),
                             E.I->piece[prompt[qi]].c_str());
                    E.I->qk_mass(d0s, d1s, 0, 0, tg, 0);
                }
            } else
            if (!E.eval_turn(prompt.data(), prompt.size(), (int) prompt.size() - q_mark, true)) { E.close(); return 1; }
            // 09-25 v2bld32 store budget: how much of the asked needle was evicted from the host store before its question
            int st_nev = -1; const int st_nn = (int) n_ids[ask].size();
            if (E.inf && E.I->p.store_budget > 0) { st_nev = 0; for (int t2 = pos[ask]; t2 < pos[ask] + st_nn; ++t2) if (E.I->is_evicted(t2)) st_nev++; }
            std::string probe_s;
            if (E.inf) {
                // residency at the LAST PROMPT GATHER: this is the band the prompt logits, and so
                // the first generated token, were computed against.
                const inf_probe p0b = E.I->probe(pos[ask], pos[ask] + (int) n_ids[ask].size());
                char p0buf[512];
                snprintf(p0buf, sizeof(p0buf),
                         "  PROBE0 ex %2d  far %d/%d  sel %d  band %d  forced %d  rank [%d,%d]  p %d  "
                         "unit %d/%d band  ctx %d/%d band  trunk %d",
                         e, p0b.n_far, p0b.n_tok, p0b.n_sel, p0b.n_band, p0b.n_forced,
                         p0b.rank_min, p0b.rank_max, p0b.p_place,
                         p0b.unit_band, p0b.unit_n, p0b.ctx_band, p0b.ctx_n, p0b.trunk_n);
                printf("%s\n", p0buf); fflush(stdout);
            }
            E.set_decode(true);

            if (E.inf) {
                const inf_probe pb = E.I->probe(pos[ask], pos[ask] + (int) n_ids[ask].size());
                char pbuf[768];
                snprintf(pbuf, sizeof(pbuf),
                         "  PROBE ex %2d  tok [%d,%d)  far %d/%d  sel %d  band %d  forced %d  votes_max %d  "
                         "rank [%d,%d]  dist [%d,%d]  unit %d [%d,%d) score %.2f rank %d of %d hit (%d taken)  "
                         "band_oldest %d  far_hi %d  p %d  adj %d/%d same %d near  span %d/%d own  "
                         "unit %d/%d band  ctx %d/%d band  trunk %d",
                         e, pos[ask], pos[ask] + (int) n_ids[ask].size(), pb.n_far, pb.n_tok, pb.n_sel, pb.n_band,
                         pb.n_forced, pb.votes_max, pb.rank_min, pb.rank_max, pb.dist_min, pb.dist_max,
                         pb.unit, pb.unit_t0, pb.unit_t1, pb.unit_score, pb.unit_rank, pb.unit_hits,
                         pb.unit_taken, pb.band_oldest, pb.far_hi, pb.p_place,
                         pb.adj_same, pb.adj_n, pb.adj_near, pb.span_own, pb.span_n,
                         pb.unit_band, pb.unit_n, pb.ctx_band, pb.ctx_n, pb.trunk_n);
                probe_s = pbuf;
                printf("%s\n", probe_s.c_str()); fflush(stdout);
            }

            std::string out;
            const int nv = llama_vocab_n_tokens(vocab);
            for (int i = 0; i < n_pred_needle; ++i) {
                const float * lg = llama_get_logits_ith(E.ctx, -1);
                llama_token best = 0; float bv = -1e30f;
                for (int v = 0; v < nv; ++v) if (lg[v] > bv) { bv = lg[v]; best = v; }
                if (llama_vocab_is_eog(vocab, best)) break;
                out += detok(vocab, {best});
                if (!E.eval(&best, 1, true)) break;
                if (E.inf && o.qk_scan > 0 && e < o.qk_scan && i < 12) {
                    // the answer prefix, token by token: the other half of the scan
                    char tg[128];
                    snprintf(tg, sizeof(tg), "Atok %d '%s'", i, E.I->piece[best].c_str());
                    E.I->qk_mass(pos[ask], pos[ask] + (int) n_ids[ask].size(), 0, 0, tg, 0);
                }
                if (E.inf && o.qk_at >= 0 && i == o.qk_at) {
                    char qt[64]; snprintf(qt, sizeof(qt), "ex%d@gen%d", e, i);
                    // locate the DIGIT tokens inside the needle so the unit's mass can be split
                    // between the digits and the handle/frame words around them
                    const auto dig = tokenize(vocab, vals[ask], false);
                    int dstart = -1;
                    for (int a2 = 0; a2 + (int) dig.size() <= (int) n_ids[ask].size(); ++a2) {
                        bool ok = true;
                        for (int b2 = 0; b2 < (int) dig.size(); ++b2)
                            if (n_ids[ask][a2+b2] != dig[b2]) { ok = false; break; }
                        if (ok) { dstart = a2; break; }
                    }
                    const int dd0 = dstart >= 0 ? pos[ask] + dstart : 0;
                    const int dd1 = dstart >= 0 ? dd0 + (int) dig.size() : 0;
                    E.I->qk_mass(pos[ask], pos[ask] + (int) n_ids[ask].size(), dd0, dd1, qt);
                }
            }
            if (E.inf && E.I->p.bandchk) {
                // 09-24 band check summary for this example: every placement's check, summed, and
                // how many bad cells carry the asked needle's (or any needle's) provenance
                inf_ctx * I = E.I;
                const int a0 = pos[ask], a1 = pos[ask] + (int) n_ids[ask].size();
                int bad_ask = 0, bad_any = 0;
                for (int32_t t : I->bchk_bad_t) {
                    if (t >= a0 && t < a1) bad_ask++;
                    for (int i = 0; i < nkeys; ++i)
                        if (t >= pos[i] && t < pos[i] + (int) n_ids[i].size()) { bad_any++; break; }
                }
                printf("  BANDCHK ex %2d  checks %lld  band live %lld  meta %lld  slot %lld  dup %lld  V %lld  K %lld  "
                       "kcos_min %.6f  |  trunk live %lld  V %lld  K %lld  kcos_min %.6f  |  bad on asked needle %d  on any needle %d\n",
                       e, (long long) I->n_bchk, (long long) I->n_bchk_live, (long long) I->n_bchk_meta,
                       (long long) I->n_bchk_slot, (long long) I->n_bchk_dup, (long long) I->n_bchk_v,
                       (long long) I->n_bchk_k, I->bchk_kmin > 1.5 ? -1.0 : I->bchk_kmin,
                       (long long) I->n_bchk_tr_live, (long long) I->n_bchk_tr_v, (long long) I->n_bchk_tr_k,
                       I->bchk_tr_kmin > 1.5 ? -1.0 : I->bchk_tr_kmin, bad_ask, bad_any);
                fflush(stdout);
            }
            const double secs = now_s() - t0;
            int em = out.find(vals[ask]) != std::string::npos;
            if (k3c) {
                // the frame control: nothing in the document answers this question, so a PASS is
                // emitting none of the three numbers.  Any number emitted is a false positive.
                bool any = false;
                for (int i = 0; i < nkeys; ++i) if (out.find(vals[i]) != std::string::npos) any = true;
                em = any ? 0 : 1;
            }
            const int dist = (int) prompt.size() - pos[ask];
            std::string o1 = out; for (auto & c : o1) if (c == '\n' || c == '\r') c = '/';
            const double occ = E.inf && E.I->n_gather ? (double) E.I->n_sel_real/E.I->n_gather/E.I->p.S : 0.0;
            char line[1024];
            snprintf(line, sizeof(line),
                     "  %-6s ex %2d  needle %6d back  EM %d  gold %s  pred %-24s  (%.0fs, %.1f tok/s, design %s, band_occ %.3f, RSS %.1f GiB)"
                     "  VOL sel_tok %lld gathers %lld new_cells %lld\n",
                     task.c_str(), e, dist, em, vals[ask].c_str(), o1.c_str(), secs, prompt.size()/secs, design, occ,
                     host_rss()/1073741824.0,
                     (long long)(E.inf ? E.I->n_sel_real : 0), (long long)(E.inf ? E.I->n_gather : 0),
                     (long long)(E.inf ? E.I->n_new_cells : 0));
            if (E.inf) {   // 09-24: evreenc explicit on every row (appended)
                const size_t L0 = strlen(line);
                if (L0 > 0 && line[L0-1] == '\n') line[L0-1] = 0;
                const size_t L1 = strlen(line);
                snprintf(line + L1, sizeof(line) - L1, "  REENC n %lld tok %lld ms %.1f evreenc %d budget %d skip %lld surpskip %lld evrsurp %.0f\n",
                         (long long) E.I->n_reenc, (long long) E.I->n_reenc_tok, 1000.0*E.I->t_reenc,
                         (int) E.I->p.evreenc, E.I->p.evr_budget, (long long) E.I->n_reenc_budget_skip,
                             (long long) E.I->n_reenc_surp_skip, E.I->p.evr_surp);
                if (E.I->p.sent_fill > 0) {   // 09-25 v2bld28 sentfill
                    const size_t L4 = strlen(line); if (L4 > 0 && line[L4-1] == '\n') line[L4-1] = 0;
                    const size_t L5 = strlen(line);
                    snprintf(line + L5, sizeof(line) - L5, "  SFILL cand %lld\n", (long long) E.I->n_sf_cand);
                }
                if (E.I->p.seed_sent) {   // 09-24 v2bld25 seedsent
                    const size_t L2 = strlen(line); if (L2 > 0 && line[L2-1] == '\n') line[L2-1] = 0;
                    const size_t L3 = strlen(line);
                    snprintf(line + L3, sizeof(line) - L3, "  SEEDU units %lld tok %lld hit %lld\n", (long long) E.I->n_seedu_units,
                             (long long) E.I->n_seedu_tok, (long long) E.I->n_seedu_cand);
                }
                if (E.I->p.store_budget > 0) {   // 09-25 v2bld32 store budget
                    const size_t L6 = strlen(line); if (L6 > 0 && line[L6-1] == '\n') line[L6-1] = 0;
                    const size_t L7 = strlen(line);
                    snprintf(line + L7, sizeof(line) - L7, "  STORE budget %lld rounds %lld evicted_units %lld (never-used %lld) evicted_tok %lld blocked %lld needle_evicted %d/%d live %lld\n",
                             (long long) E.I->p.store_budget, (long long) E.I->n_evict_rounds, (long long) E.I->n_evict_units,
                             (long long) E.I->n_evict_units_unused, (long long) E.I->n_evict_tok, (long long) E.I->n_evict_blocked,
                             st_nev, st_nn, (long long) (E.I->host.n_tok - E.I->n_evict_tok));
                }
            }
            printf("%s", line); fflush(stdout);
            if (fout) { fprintf(fout, "%s", line); if (!probe_s.empty()) fprintf(fout, "%s\n", probe_s.c_str()); fflush(fout); }
            if (e == 0 && E.inf && o.dump_n > 0) E.I->report(o.tag.c_str());
            rows.push_back({task, dist, em});
            E.close();
        }
    }

    auto bucket_of = [](int d) { return d < 16384 ? 0 : (d < 65536 ? 1 : 2); };
    const char * bn[3] = {"<16K", "16-64K", ">64K"};
    printf("\n# needle retrieval, tag %s, design %s, T = %d, %d examples per task\n", o.tag.c_str(), design, o.T, o.nex);
    for (const auto & task : tasks) {
        int n = 0, s = 0, bs[3] = {0,0,0}, bc[3] = {0,0,0};
        for (auto & r : rows) if (r.task == task) { n++; s += r.em; bs[bucket_of(r.dist)] += r.em; bc[bucket_of(r.dist)]++; }
        if (!n) continue;
        printf("  %-8s EM %6.3f  n %3d |", task.c_str(), (double) s/n, n);
        for (int b = 0; b < 3; ++b) { if (bc[b]) printf(" %6s %.3f (%2d)", bn[b], (double) bs[b]/bc[b], bc[b]); else printf(" %6s   -     ", bn[b]); }
        printf("\n");
        printf("NEEDLE\t%s\tdesign=%s\tT=%d\ttask=%s\tEM=%.4f\tn=%d\tevreenc=%d\tevr_budget=%d\tprerope=%d\twin_k=%d\tpinhost=%d\tcapblk=%d\tunit=%s\tfloor=%d\tseed_span=%d\tseed_words=%d\tevr_surp=%.1f\tseed=%d\tex0=%d\tsum=%d\n", o.tag.c_str(), design, o.T, task.c_str(), (double) s/n, n,
               o.inf ? (int) E.ip.evreenc : -1, o.inf ? E.ip.evr_budget : -1, o.inf ? (int) E.ip.prerope : -1,
               o.inf ? E.ip.win_k : -1, o.inf ? E.ip.pinhost : -1, o.inf ? (int) E.ip.capblk : -1,
               unit_s.c_str(), o.inf ? (int) E.ip.floor_only : 0, o.inf ? E.ip.seed_span : -1, o.inf ? E.ip.seed_words : -1,
               o.inf ? E.ip.evr_surp : -1.0, o.seed, o.ex0, s);   // 09-24 FINAL: + unit/floor/seed/evr_surp, the item seed, the section start, the EM count
        // two-hop: the first hop is producing the rare word at all, and it is a property of the
        // prompt rather than of retrieval.  Report it separately and report EM conditional on it,
        // so a prompt artefact cannot be read as a retrieval failure.
        {
            int nw = 0, sw = 0, ncond = 0, scond = 0;
            for (auto & r : rows) if (r.task == task && r.word >= 0) {
                nw++; sw += r.word;
                if (r.word) { ncond++; scond += r.em; }
            }
            if (nw) {
                printf("  %-8s first hop (word produced) %.3f (%d/%d)   EM | word %s\n",
                       task.c_str(), (double) sw/nw, sw, nw,
                       ncond ? (std::to_string(scond) + "/" + std::to_string(ncond)).c_str() : "-");
                printf("NEEDLEC\t%s\tdesign=%s\tT=%d\ttask=%s\tword=%.4f\tEMcond=%.4f\tnw=%d\tncond=%d\n",
                       o.tag.c_str(), design, o.T, task.c_str(), (double) sw/nw,
                       ncond ? (double) scond/ncond : 0.0, nw, ncond);
            }
        }
    }
    printf("# total %.0fs\n", now_s() - t_all);

    if (fout) {
        fprintf(fout, "\n# needle retrieval, tag %s, design %s, T = %d, %d examples per task\n", o.tag.c_str(), design, o.T, o.nex);
        for (const auto & task : tasks) {
            int n = 0, s = 0;
            for (auto & r : rows) if (r.task == task) { n++; s += r.em; }
            if (n) fprintf(fout, "#   %-8s EM %.3f over %d needles\n", task.c_str(), (double) s/n, n);
        }
        fclose(fout);
    }

    llama_model_free(model);
    return 0;
}
