// The online unigram rarity model used by the infinite-context policy.
//
// The policy needs a per-token surprisal to decide which words are worth indexing (the trigger
// threshold) and which are worth querying with.  Shipping a frequency table per model does not
// generalise across tokenizers, so the rarity is estimated online from the tokens the context has
// already seen; the cold start falls back to token-id order, which BPE and SentencePiece build in
// merge order, so it is already roughly frequency-ordered.
//
// Rarity cutoffs are expressed as *tail occurrence mass*, which is portable across tokenizers,
// rather than as a literal type rank, which is not.
//
// Provenance: carried unchanged from the paper13 "window + forget-list store" experiment, where it
// was `kvs_freq`.  It is the only part of that experiment v2 keeps; the store itself is gone.
#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

struct inf_freq {
    int32_t              n_vocab = 0;
    int64_t              n_tok   = 0;
    std::vector<int64_t> cnt;      // running occurrence counts, indexed by token id
    std::vector<int32_t> rnk;      // rank of each id by (cnt desc, id asc); 0 = most frequent
    std::vector<int64_t> tail;     // tail[i] = occurrences held by the ids of rank >= i
    int64_t              n_refresh = -1;

    void    init(int32_t V);
    void    add(llama_token t) { if (t >= 0 && t < n_vocab) { cnt[t]++; n_tok++; } }
    void    refresh(bool force = false);         // rebuild rnk/tail (cheap, called once per block)
    double  logp(llama_token t) const;           // natural log of (cnt + 0.5) / (n + 0.5 V)
    double  smax() const;                        // surprisal of an unseen piece = the scale of logp
    int32_t rank_of(llama_token t) const { return (t >= 0 && t < n_vocab) ? rnk[t] : n_vocab; }
    int32_t rank_for_tail(double c) const;       // smallest N whose tail occurrence mass is <= c
    bool    load_table(const std::string & path);// optional ground-truth table (validation only)
    bool    save_counts(const std::string & path) const;
    bool    load_counts(const std::string & path);
};
