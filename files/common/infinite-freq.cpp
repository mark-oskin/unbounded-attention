#include "infinite-freq.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>

void inf_freq::init(int32_t V) {
    n_vocab = V;
    cnt.assign(V, 0);
    rnk.assign(V, 0);
    tail.assign(V + 1, 0);
    n_tok = 0;
    n_refresh = -1;
    refresh(true);
}

void inf_freq::refresh(bool force) {
    // refresh geometrically: the ranking moves slowly once a few thousand tokens are in
    if (!force && n_refresh >= 0 && n_tok < n_refresh + 1024 + n_refresh / 8) return;
    n_refresh = n_tok;

    std::vector<int32_t> order(n_vocab);
    std::iota(order.begin(), order.end(), 0);
    // key: count descending, then token id ascending.  With no counts yet this is exactly
    // the token-id order, which is the cold-start fallback (BPE/SPM merge order).
    std::stable_sort(order.begin(), order.end(),
                     [&](int32_t a, int32_t b) { return cnt[a] > cnt[b]; });
    for (int32_t i = 0; i < n_vocab; ++i) rnk[order[i]] = i;

    tail[n_vocab] = 0;
    for (int32_t i = n_vocab - 1; i >= 0; --i) tail[i] = tail[i + 1] + cnt[order[i]];
}

double inf_freq::logp(llama_token t) const {
    const double c = (t >= 0 && t < n_vocab) ? (double) cnt[t] : 0.0;
    return std::log((c + 0.5) / ((double) n_tok + 0.5 * (double) n_vocab));
}

double inf_freq::smax() const {
    return -std::log(0.5 / ((double) n_tok + 0.5 * (double) n_vocab));
}

int32_t inf_freq::rank_for_tail(double c) const {
    if (n_tok <= 0) return (int32_t) (c * n_vocab);            // cold start: id-order quantile
    const int64_t want = (int64_t) (c * (double) n_tok);
    // tail[] is non-increasing: find the smallest N with tail[N+1] <= want
    int32_t lo = 0, hi = n_vocab;
    while (lo < hi) {
        const int32_t mid = (lo + hi) / 2;
        if (tail[mid + 1] <= want) hi = mid; else lo = mid + 1;
    }
    return lo;
}

bool inf_freq::load_table(const std::string & path) {
    std::ifstream f(path);
    if (!f) return false;
    int32_t id; int64_t c; int n = 0;
    while (f >> id >> c) { if (id >= 0 && id < n_vocab) { cnt[id] += c; n_tok += c; n++; } }
    refresh(true);
    return n > 0;
}

bool inf_freq::save_counts(const std::string & path) const {
    std::ofstream f(path);
    if (!f) return false;
    for (int32_t i = 0; i < n_vocab; ++i) if (cnt[i]) f << i << ' ' << cnt[i] << '\n';
    return true;
}

bool inf_freq::load_counts(const std::string & path) { return load_table(path); }