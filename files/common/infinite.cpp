#include "infinite.h"
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_set>
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <functional>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <numeric>
#include <thread>
#include <mutex>
#include <unordered_map>

// ---------------------------------------------------------------------------------------------
static double inf_now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static inline bool inf_word_start(const std::string & pc);   // defined with the event path
static inline uint64_t inf_word_hash(const std::string & w);

// fp16 <-> fp32 through a 64K table (the quadratic term reads tens of millions of halves a block)
static const float * inf_f16tab() {
    static std::vector<float> tab;
    if (tab.empty()) {
        tab.resize(65536);
        for (int i = 0; i < 65536; ++i) {
            const uint16_t h = (uint16_t) i;
            const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
            const uint32_t exp  = (h >> 10) & 0x1f;
            const uint32_t man  = h & 0x3ff;
            uint32_t bits;
            if (exp == 0) {
                if (man == 0) { bits = sign; }
                else {
                    int e = -1; uint32_t m = man;
                    do { m <<= 1; e++; } while (!(m & 0x400));
                    bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3ff) << 13);
                }
            } else if (exp == 31) {
                bits = sign | 0x7f800000u | (man << 13);
            } else {
                bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
            }
            float f; memcpy(&f, &bits, 4); tab[i] = f;
        }
    }
    return tab.data();
}

static inline uint16_t inf_f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int32_t  e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (e >= 31)      return (uint16_t)(sign | 0x7c00);
    if (e <= 0) {
        if (e < -10)  return (uint16_t) sign;
        m |= 0x800000;
        const uint32_t sh = (uint32_t)(14 - e);
        return (uint16_t)(sign | (m >> sh));
    }
    return (uint16_t)(sign | ((uint32_t) e << 10) | (m >> 13));
}

template <typename F>
static void inf_parallel(int32_t nthread, int32_t n, F f) {
    if (nthread <= 1 || n <= 1) { for (int32_t i = 0; i < n; ++i) f(i); return; }
    std::atomic<int32_t> next(0);
    const int32_t nt = std::min<int32_t>(nthread, n);
    std::vector<std::thread> th;
    th.reserve(nt);
    for (int32_t t = 0; t < nt; ++t) {
        th.emplace_back([&]() { for (;;) { const int32_t i = next.fetch_add(1); if (i >= n) break; f(i); } });
    }
    for (auto & x : th) x.join();
}

// Host->device staging goes through a PINNED buffer: ggml_backend_tensor_set() from pageable
// memory runs at ~5 GB/s, from pinned at ~20 GB/s, and the gather moves hundreds of MiB a block.
static uint8_t * inf_pin_alloc(size_t n, ggml_backend_buffer_t * out) {
    *out = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        ggml_backend_buffer_type_t bt = ggml_backend_dev_host_buffer_type(d);
        if (!bt) break;
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(bt, n);
        if (!buf) break;
        *out = buf;
        return (uint8_t *) ggml_backend_buffer_get_base(buf);
    }
    return (uint8_t *) malloc(n);
}

// the closed-class words that never trigger a far read (paper13's list)
static const char * INF_STOP =
    "the a an and or but if then than that this these those of to in on at by for with from into over under "
    "is are was were be been being am do does did done have has had having will would shall should can could "
    "may might must not no nor so as it its it's he she they them their there here we you i my your our his her "
    "him us me who whom whose which what when where why how all any both each few more most other some such only "
    "own same too very s t don now just also about after again against before below between during further once "
    "out off up down while because until although though upon within without among across toward towards per";

// ---------------------------------------------------------------------------------------------
// host far K/V
// ---------------------------------------------------------------------------------------------
inf_hostkv::~inf_hostkv() {
    for (auto * s : kslab) free(s);
    for (auto * s : vslab) free(s);
}

void inf_hostkv::init(size_t row_, int32_t n_layer_, size_t slab_bytes) {
    row       = row_;
    n_layer   = n_layer_;
    tok_bytes = row * (size_t) n_layer;
    slab_tok  = std::max<size_t>(1, slab_bytes / tok_bytes);
    n_tok     = 0;
}

void inf_hostkv::ensure(int64_t t) {
    const size_t need = (size_t)(t / (int64_t) slab_tok) + 1;
    while (kslab.size() < need) {
        uint8_t * a = (uint8_t *) malloc(slab_tok * tok_bytes);
        uint8_t * b = (uint8_t *) malloc(slab_tok * tok_bytes);
        if (!a || !b) { fprintf(stderr, "infinite: out of host memory for the far KV store\n"); exit(1); }
        kslab.push_back(a);
        vslab.push_back(b);
    }
    if (t + 1 > n_tok) n_tok = t + 1;
}

// ---------------------------------------------------------------------------------------------
void inf_ctx::init(llama_context * c, const inf_params & pr) {
    lctx  = c;
    model = llama_get_model(c);
    vocab = llama_model_get_vocab(model);
    mem   = llama_get_memory(c);
    p     = pr;

    n_layer   = llama_model_n_layer(model);
    n_head    = llama_model_n_head(model);
    n_head_kv = llama_model_n_head_kv(model);
    head_dim  = llama_model_n_embd(model) / n_head;
    n_embd_k_gqa = n_head_kv * head_dim;
    n_lh      = n_layer * n_head_kv;
    gqa       = n_head / n_head_kv;
    n_ctx     = (int32_t) llama_n_ctx(c);
    kq_scale  = 1.0f / sqrtf((float) head_dim);

    int32_t nl = 0, ne = 0;
    row = llama_memory_kv_row_bytes(mem, &nl, &ne);
    if (nl != n_layer || ne != n_embd_k_gqa || row != (size_t) n_embd_k_gqa * 2) {
        fprintf(stderr, "infinite: the K/V cache must be f16 and unified (got row %zu, n_layer %d, n_embd_k_gqa %d)\n",
                row, nl, ne);
        exit(1);
    }

    // RoPE planes
    std::vector<float> th(head_dim);
    const int32_t np = llama_model_rope_thetas(model, th.data(), head_dim);
    if (np <= 0) { fprintf(stderr, "infinite: llama_model_rope_thetas failed\n"); exit(1); }
    th.resize(np);
    theta = th;
    rope_neox = (llama_model_rope_type(model) == LLAMA_ROPE_TYPE_NEOX) ? 1 : 0;
    p.n_planes = std::min(p.n_planes, np);
    plane0 = np - p.n_planes;
    P  = 2 * p.n_planes;
    P2 = P * (P + 1) / 2;

    const int32_t V = llama_vocab_n_tokens(vocab);
    freq.init(V);
    piece.resize(V);
    tok_stop.assign(V, 0);
    {
        std::string s(INF_STOP), w;
        size_t a = 0;
        while (a < s.size()) {
            size_t b = s.find(' ', a);
            if (b == std::string::npos) b = s.size();
            if (b > a) stopw.insert(s.substr(a, b - a));
            a = b + 1;
        }
        char buf[256];
        for (int32_t i = 0; i < V; ++i) {
            const int n = llama_token_to_piece(vocab, i, buf, sizeof(buf), 0, true);
            piece[i] = n > 0 ? std::string(buf, n) : std::string();
            std::string core;
            for (char ch : piece[i]) {
                const unsigned char u = (unsigned char) ch;
                if (isalnum(u) || u >= 0x80) core += (char) tolower(u);
            }
            tok_stop[i] = core.empty() || stopw.count(core) ? 1 : 0;
        }
    }

    host.init(row, n_layer);

    clu_mu.assign(n_lh, {});
    clu_sig.assign(n_lh, {});
    clu_logn.assign(n_lh, {});
    clu_mem.assign(n_lh, {});
    clu_off.assign(n_lh, {});
    for (int32_t i = 0; i < n_lh; ++i) clu_off[i].push_back(0);

    qmean.assign((size_t) n_lh * P, 0.0f);

    band_tok.assign(p.S, -1);
    band_c0 = (uint32_t) (n_ctx - p.S);
    ring = 0;
    pin_pos.resize(p.n_pin + 1);
    for (int32_t i = 0; i <= p.n_pin; ++i) pin_pos[i] = i;

    bufQ.resize((size_t) p.B * n_head * head_dim);
    bufKV.resize((size_t) p.B * n_embd_k_gqa);
    f16buf.resize((size_t) p.B * n_embd_k_gqa);
    qq.resize(P2);

    pin_cap = (size_t) INF_WRITE_MAX * n_layer * row;
    pin_k = inf_pin_alloc(pin_cap, &pin_kbuf);
    pin_v = inf_pin_alloc(pin_cap, &pin_vbuf);
    if (!pin_k || !pin_v) { fprintf(stderr, "infinite: staging allocation failed\n"); exit(1); }

    inf_f16tab();

    if (p.debug) {
        fprintf(stderr, "infinite: n_layer %d n_head %d n_head_kv %d head_dim %d row %zu n_ctx %d band_c0 %u\n",
                n_layer, n_head, n_head_kv, head_dim, row, n_ctx, band_c0);
        fprintf(stderr, "infinite: planes %d..%d of %d  theta[%d] %.3e  theta[%d] %.3e  rope %s\n",
                plane0, np - 1, np, plane0, theta[plane0], np - 1, theta[np - 1], rope_neox ? "neox" : "norm");
    }
}

// de-rotate n_vec vectors of head_dim floats (stride apart) by RoPE at `pos`, in place,
// keeping only the slow planes' 2*n_planes dims, written compactly to the front of each vector.
void inf_ctx::derope(float * x, int32_t pos, int32_t n_vec, int32_t stride) const {
    const int32_t np = (int32_t) theta.size();
    for (int32_t j = 0; j < p.n_planes; ++j) {
        const int32_t pl = plane0 + j;
        const float a  = theta[pl] * (float) pos;
        const float cs = cosf(a), sn = sinf(a);
        const int32_t i0 = rope_neox ? pl            : 2*pl;
        const int32_t i1 = rope_neox ? pl + np       : 2*pl + 1;
        for (int32_t v = 0; v < n_vec; ++v) {
            float * q = x + (size_t) v * stride;
            const float u = q[i0], w = q[i1];
            // roped: u = u0 c - w0 s, w = u0 s + w0 c  ->  u0 = u c + w s, w0 = -u s + w c
            q[i0] = u*cs + w*sn;
            q[i1] = -u*sn + w*cs;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// capture: read the per-layer Q/K/V of the block that was just decoded
// ---------------------------------------------------------------------------------------------
// Rotate one f16 key row to `pos`, every plane, using the model's own thetas.  This is the
// SINGLE rotation of the prerope path: the stored key was never rotated, so there is no second
// rotation to compound with however far the cell has travelled.
void inf_ctx::rope_row_f16(uint8_t * krow, int32_t pos) const {
    const int32_t np = (int32_t) theta.size();
    const float * f16 = inf_f16tab();
    uint16_t * k = (uint16_t *) krow;
    for (int32_t hh = 0; hh < n_head_kv; ++hh) {
        uint16_t * v = k + (size_t) hh * head_dim;
        for (int32_t pl = 0; pl < np; ++pl) {
            const float a = theta[pl] * (float) pos;
            const float cs = cosf(a), sn = sinf(a);
            const int32_t i0 = rope_neox ? pl      : 2*pl;
            const int32_t i1 = rope_neox ? pl + np : 2*pl + 1;
            if (i0 >= head_dim || i1 >= head_dim) continue;
            const float u = f16[v[i0]], w = f16[v[i1]];
            const float tmp[2] = { u*cs - w*sn, u*sn + w*cs };
            ggml_fp16_t o[2];
            ggml_fp32_to_fp16_row(tmp, o, 2);
            v[i0] = *(const uint16_t *) &o[0];
            v[i1] = *(const uint16_t *) &o[1];
        }
    }
}

// Placement for the prerope path: rewrite the band contiguously, each key rotated once to the
// slot it is about to occupy.  No shift is queued, so a cell re-placed thirty times has still
// been rotated exactly once from the stored key.
// 09-23 (T9 + perf): one process-wide pinned staging buffer for the rotated K of a whole band, so
// a contiguous band is written in ONE run (one tensor_set per layer) instead of S/INF_WRITE_MAX
// runs.  Process-wide because an inf_ctx is created per example and the staging buffers are
// never freed; a per-instance 1 GiB buffer would leak once per example.
static uint8_t *              g_rot_k    = nullptr;
static ggml_backend_buffer_t  g_rot_kbuf = nullptr;
static size_t                 g_rot_cap  = 0;

// rope_row_f16 with the per-plane cos/sin precomputed: the angle depends only on (pos, plane),
// so one cos/sin per plane per CELL serves every layer and KV head (n_layer * n_head_kv = 256x
// fewer sincos calls than rotating each row from scratch).  Same arithmetic, same result.
static inline void inf_rope_row_cs(uint16_t * k, const float * cs, const float * sn, int32_t np,
                                   int32_t n_head_kv, int32_t head_dim, bool neox, const float * f16) {
    for (int32_t hh = 0; hh < n_head_kv; ++hh) {
        uint16_t * v = k + (size_t) hh * head_dim;
        for (int32_t pl = 0; pl < np; ++pl) {
            const int32_t i0 = neox ? pl      : 2*pl;
            const int32_t i1 = neox ? pl + np : 2*pl + 1;
            if (i0 >= head_dim || i1 >= head_dim) continue;
            const float u = f16[v[i0]], w = f16[v[i1]];
            const float tmp[2] = { u*cs[pl] - w*sn[pl], u*sn[pl] + w*cs[pl] };
            ggml_fp16_t o[2];
            ggml_fp32_to_fp16_row(tmp, o, 2);
            v[i0] = *(const uint16_t *) &o[0];
            v[i1] = *(const uint16_t *) &o[1];
        }
    }
}

// Placement for the prerope path: rewrite the band's K, each key rotated once to the slot it is
// about to occupy.  No shift is queued, so a cell re-placed thirty times has still been rotated
// exactly once from the stored key.  V is NOT rewritten: a live cell's V was written when its
// token entered the cell (ring append or event) and values are never roped (09-23: halves the
// placement's host->device bytes).
void inf_ctx::place_rotated() {
    const size_t n = cell_idx.size();
    if (!n) return;
    // cell_idx is in RANK order; kv_cells_write() takes one contiguous ascending cell run, so the
    // placement is ordered by cell and written as maximal contiguous runs (no 2048-cell cap).
    std::vector<size_t> ord(n);
    for (size_t r = 0; r < n; ++r) ord[r] = r;
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return cell_idx[a] < cell_idx[b]; });
    std::vector<uint32_t>  cells(n);
    std::vector<int32_t>   toks(n);
    std::vector<llama_pos> dsts(n);
    for (size_t i = 0; i < n; ++i) {
        const size_t r = ord[i];
        cells[i] = cell_idx[r];
        toks[i]  = band_tok[cell_idx[r] - band_c0];
        dsts[i]  = dst_pos[r];
    }
    const size_t need = n * (size_t) n_layer * row;
    if (g_rot_cap < need) {
        if (g_rot_kbuf) ggml_backend_buffer_free(g_rot_kbuf); else if (g_rot_k) free(g_rot_k);
        g_rot_kbuf = nullptr;
        g_rot_k = inf_pin_alloc(need, &g_rot_kbuf);
        if (!g_rot_k) { fprintf(stderr, "infinite: rotation staging allocation failed (%zu B)\n", need); exit(1); }
        g_rot_cap = need;
    }
    const int32_t np = (int32_t) theta.size();
    std::vector<float> CS(n * (size_t) np), SN(n * (size_t) np);
    for (size_t i = 0; i < n; ++i)
        for (int32_t pl = 0; pl < np; ++pl) {
            const float a = theta[pl] * (float) dsts[i];
            CS[i*np + pl] = cosf(a); SN[i*np + pl] = sinf(a);
        }
    const float * f16 = inf_f16tab();
    size_t done = 0;
    while (done < n) {
        size_t m = 1;
        while (done + m < n && cells[done + m] == cells[done] + m) ++m;
        inf_parallel(p.nthread, n_layer, [&](int32_t li) {
            for (size_t j = 0; j < m; ++j) {
                uint8_t * dk = g_rot_k + ((size_t) li*m + j)*row;
                memcpy(dk, host.k(toks[done + j]) + (size_t) li*row, row);
                inf_rope_row_cs((uint16_t *) dk, &CS[(done + j)*np], &SN[(done + j)*np], np,
                                n_head_kv, head_dim, rope_neox, f16);
            }
        });
        src_pos.resize(m);
        for (size_t j = 0; j < m; ++j) src_pos[j] = dsts[done + j];   // src == dst: no shift
        llama_memory_kv_cells_write(mem, seq, cells[done], m,
                                    src_pos.data(), dsts.data() + done, g_rot_k, nullptr,
                                    toks.data() + done);
        bytes_h2d += (int64_t) m * n_layer * (int64_t) row;
        n_runs++;
        done += m;
    }
}

// 09-24 PERF capblk: write the V (and, without prerope, the K) of the deferred decode tokens to
// the host store: one kv_cells_read per layer per tensor for the whole run instead of per token.
void inf_ctx::flush_capture() {
    if (cap_hi <= cap_lo) { cap_lo = cap_hi = 0; return; }
    const double t0 = inf_now();
    const int32_t p0 = cap_lo, n = cap_hi - cap_lo;
    cap_lo = cap_hi = 0;
    llama_synchronize(lctx);
    host.ensure((int64_t) p0 + n - 1);
    cellbuf.resize(n);
    if (!llama_memory_kv_cells_of(mem, seq, p0, p0 + n, cellbuf.data())) {
        fprintf(stderr, "infinite: flush_capture: cells for [%d, %d) not all present\n", p0, p0 + n);
        exit(1);
    }
    const bool want_k = !p.prerope;   // prerope: K went to the host from the graph, per token
    llama_memory_kv_cells_read(mem, seq, cellbuf.data(), (size_t) n, want_k ? pin_k : nullptr, pin_v);
    bytes_d2h += (want_k ? 2ll : 1ll) * n * n_layer * (int64_t) row;
    auto scatter = [&](int32_t il) {
        for (int32_t j = 0; j < n; ++j) {
            if (want_k) memcpy(host.k(p0 + j) + (size_t) il*row, pin_k + ((size_t) il*n + j)*row, row);
            memcpy(host.v(p0 + j) + (size_t) il*row, pin_v + ((size_t) il*n + j)*row, row);
        }
    };
    if (n <= 8) for (int32_t il = 0; il < n_layer; ++il) scatter(il);
    else inf_parallel(p.nthread, n_layer, scatter);
    n_cap_flush++;
    t_capture += inf_now() - t0;
}

void inf_ctx::capture(int32_t p0, int32_t n, bool want_q) {
    const double t0 = inf_now();

    // 09-24 PERF capblk: a decode token (no clusters) only takes its pre-RoPE K from the graph now
    // -- the graph holds the last ubatch only -- and defers its V to one read per block.
    if (p.capblk && in_decode && !p.use_clusters && n < p.B) {
        if (cap_hi > cap_lo && cap_hi != p0) flush_capture();   // keep the pending run contiguous
        llama_synchronize(lctx);
        host.ensure((int64_t) p0 + n - 1);
        if (p.prerope) {
            const int32_t qrow = n_head_kv * head_dim;
            bufK.resize((size_t) qrow * n);
            for (int32_t li = 0; li < n_layer; ++li) {
                if (llama_get_qkv(lctx, li, 3, 0, n, bufK.data()) < 0) {
                    fprintf(stderr, "infinite: prerope capture failed at layer %d\n", li);
                    exit(1);
                }
                for (int32_t j = 0; j < n; ++j)
                    ggml_fp32_to_fp16_row(bufK.data() + (size_t) j*qrow,
                                          (ggml_fp16_t *)(host.k(p0 + j) + (size_t) li*row), qrow);
            }
        }
        if (cap_hi <= cap_lo) cap_lo = p0;
        cap_hi = p0 + n;
        n_cap_defer += n;
        t_capture += inf_now() - t0;
        if (cap_hi % p.B == 0) flush_capture();                // the block is complete
        return;
    }
    if (cap_hi > cap_lo) flush_capture();                       // a new turn after deferred tokens

    // llama_decode() runs the graph with ggml_backend_sched_graph_compute_async() and does NOT
    // synchronise before returning -- llama_get_logits() is what normally forces the wait.  The
    // CUDA backend's get_tensor uses cudaStreamPerThread, which is not ordered against the
    // compute stream, so reading the cache (or any graph tensor) here without this call races
    // with the decode that just produced it.  That race, not the design, was what made every
    // far-context read garbage.
    llama_synchronize(lctx);

    host.ensure((int64_t) p0 + n - 1);

    // K/V come straight out of the KV CACHE, in its own f16 row format: that is exactly what
    // attention reads, so the host copy is byte-identical to the GPU copy and there is no
    // dependence on which ggml node held Kcur/Vcur (reading the graph outputs instead produced
    // wrong values -- see LOG.md).  For a freshly decoded block the cells are contiguous, so
    // this is one device->host copy per layer per tensor.
    cellbuf.resize(n);
    if (!llama_memory_kv_cells_of(mem, seq, p0, p0 + n, cellbuf.data())) {
        fprintf(stderr, "infinite: cells for [%d, %d) not all present\n", p0, p0 + n);
        exit(1);
    }
    // 09-24 PERF capnok: under prerope the cache K is overwritten below by the graph's pre-RoPE K,
    // so it is not read at all
    const bool rd_k = !(p.capnok && p.prerope);
    llama_memory_kv_cells_read(mem, seq, cellbuf.data(), (size_t) n, rd_k ? pin_k : nullptr, pin_v);
    bytes_d2h += (rd_k ? 2ll : 1ll) * n * n_layer * (int64_t) row;

    // prerope: overwrite the K half with the key as it was BEFORE the rotation, read from the
    // graph output capture_kpre() kept.  V is unaffected -- values are never roped.  The cache
    // copy above is still what the trunk uses; only what the HOST stores changes.
    if (p.prerope) {
        const int32_t qrow = n_head_kv * head_dim;
        bufK.resize((size_t) qrow * n);
        for (int32_t li = 0; li < n_layer; ++li) {
            if (llama_get_qkv(lctx, li, 3, 0, n, bufK.data()) < 0) {
                fprintf(stderr, "infinite: prerope capture failed at layer %d\n", li);
                exit(1);
            }
            for (int32_t j = 0; j < n; ++j)
                ggml_fp32_to_fp16_row(bufK.data() + (size_t) j*qrow,
                                      (ggml_fp16_t *)(pin_k + ((size_t) li*n + j)*row), qrow);
        }
    }

    // [layer][cell][row] -> the host store's [token][layer][row]
    inf_parallel(p.nthread, n_layer, [&](int32_t il) {
        for (int32_t j = 0; j < n; ++j) {
            memcpy(host.k(p0 + j) + (size_t) il*row, pin_k + ((size_t) il*n + j)*row, row);
            memcpy(host.v(p0 + j) + (size_t) il*row, pin_v + ((size_t) il*n + j)*row, row);
        }
    });

    if (want_q && p.use_clusters) {   // the mean query only feeds the cluster sentinels
        const int32_t nq = std::min(n, p.qtok);
        const int32_t t0q = n - nq;
        std::fill(qmean.begin(), qmean.end(), 0.0f);
        const int32_t qrow = n_head * head_dim;
        for (int32_t il = 0; il < n_layer; ++il) {
            if (llama_get_qkv(lctx, il, 0, t0q, nq, bufQ.data()) < 0) {
                fprintf(stderr, "infinite: llama_get_qkv(Q) failed at layer %d\n", il); exit(1);
            }
            bytes_d2h += (int64_t) nq * qrow * 4;
            for (int32_t j = 0; j < nq; ++j) {
                float * qr = bufQ.data() + (size_t) j * qrow;
                derope(qr, p0 + t0q + j, n_head, head_dim);
                for (int32_t h = 0; h < n_head; ++h) {
                    const int32_t kh = h / gqa;
                    float * dst = qmean.data() + ((size_t) il * n_head_kv + kh) * P;
                    const float * s = qr + (size_t) h * head_dim;
                    for (int32_t e = 0; e < p.n_planes; ++e) {
                        const int32_t pl = plane0 + e;
                        const int32_t i0 = rope_neox ? pl : 2*pl;
                        const int32_t i1 = rope_neox ? pl + (int32_t) theta.size() : 2*pl + 1;
                        dst[2*e]   += s[i0];
                        dst[2*e+1] += s[i1];
                    }
                }
            }
        }
        const float sc = kq_scale / (float) (nq * gqa);
        for (auto & v : qmean) v *= sc;
        have_q = true;
    }

    t_capture += inf_now() - t0;
}

// ---------------------------------------------------------------------------------------------
// units (sentences) and the inverted index on rare tokens
// ---------------------------------------------------------------------------------------------
static bool inf_is_break(const std::string & s) {
    for (char c : s) if (c == '.' || c == '!' || c == '?' || c == '\n') return true;
    return false;
}

// ---------------------------------------------------------------------------------------------
// clustering: one chunk, all (layer, KV head) pairs
// ---------------------------------------------------------------------------------------------
void inf_ctx::cluster_chunk(int32_t c) {
    const double t0 = inf_now();

    const int32_t a = c * p.chunk;
    const int32_t b = std::min<int32_t>(a + p.chunk, (int32_t) toks.size());
    const int32_t m = b - a;
    if (m < p.n_clu) return;

    const int32_t K = p.n_clu;
    const int32_t np = (int32_t) theta.size();

    // de-rotation tables for the slow planes over the chunk's positions (incremental rotation)
    std::vector<float> cs((size_t) p.n_planes * m), sn((size_t) p.n_planes * m);
    for (int32_t j = 0; j < p.n_planes; ++j) {
        const double th = theta[plane0 + j];
        for (int32_t i = 0; i < m; ++i) {
            const double ang = th * (double) (a + i);
            cs[(size_t) j*m + i] = (float) cos(ang);
            sn[(size_t) j*m + i] = (float) sin(ang);
        }
    }

    inf_parallel(p.nthread, n_lh, [&](int32_t lh) {
        const int32_t il = lh / n_head_kv;
        const int32_t h  = lh % n_head_kv;

        // features: m points of P dims, de-rotated
        std::vector<float> X((size_t) m * P);
        const float * tab = inf_f16tab();
        for (int32_t i = 0; i < m; ++i) {
            const uint16_t * kr = (const uint16_t *) (host.k(a + i) + (size_t) il * row) + (size_t) h * head_dim;
            float * x = X.data() + (size_t) i * P;
            for (int32_t j = 0; j < p.n_planes; ++j) {
                const int32_t pl = plane0 + j;
                const int32_t i0 = rope_neox ? pl : 2*pl;
                const int32_t i1 = rope_neox ? pl + np : 2*pl + 1;
                const float u = tab[kr[i0]], w = tab[kr[i1]];
                const float C = cs[(size_t) j*m + i], S = sn[(size_t) j*m + i];
                x[2*j]   =  u*C + w*S;
                x[2*j+1] = -u*S + w*C;
            }
        }

        // k-means, deterministic stride init
        std::vector<float> C((size_t) K * P);
        for (int32_t k = 0; k < K; ++k) {
            const int32_t i = (int32_t) ((int64_t) k * m / K);
            memcpy(C.data() + (size_t) k*P, X.data() + (size_t) i*P, sizeof(float)*P);
        }
        std::vector<int32_t> lab(m, 0), cnt(K, 0);
        std::vector<float>   sum((size_t) K * P);
        std::vector<float>   cn2(K);
        for (int32_t it = 0; it < p.km_iter; ++it) {
            for (int32_t k = 0; k < K; ++k) {
                float s = 0; const float * ck = C.data() + (size_t) k*P;
                for (int32_t e = 0; e < P; ++e) s += ck[e]*ck[e];
                cn2[k] = 0.5f*s;
            }
            for (int32_t i = 0; i < m; ++i) {
                const float * x = X.data() + (size_t) i*P;
                int32_t best = 0; float bv = -1e30f;
                for (int32_t k = 0; k < K; ++k) {
                    const float * ck = C.data() + (size_t) k*P;
                    float d = -cn2[k];
                    for (int32_t e = 0; e < P; ++e) d += x[e]*ck[e];
                    if (d > bv) { bv = d; best = k; }
                }
                lab[i] = best;
            }
            if (it + 1 == p.km_iter) break;
            std::fill(sum.begin(), sum.end(), 0.0f);
            std::fill(cnt.begin(), cnt.end(), 0);
            for (int32_t i = 0; i < m; ++i) {
                float * s = sum.data() + (size_t) lab[i]*P;
                const float * x = X.data() + (size_t) i*P;
                for (int32_t e = 0; e < P; ++e) s[e] += x[e];
                cnt[lab[i]]++;
            }
            for (int32_t k = 0; k < K; ++k) {
                if (cnt[k] == 0) continue;
                const float inv = 1.0f/(float) cnt[k];
                float * ck = C.data() + (size_t) k*P;
                const float * s = sum.data() + (size_t) k*P;
                for (int32_t e = 0; e < P; ++e) ck[e] = s[e]*inv;
            }
        }

        // sentinels + CSR membership
        std::fill(cnt.begin(), cnt.end(), 0);
        for (int32_t i = 0; i < m; ++i) cnt[lab[i]]++;
        std::vector<uint32_t> off(K + 1, 0);
        for (int32_t k = 0; k < K; ++k) off[k+1] = off[k] + (uint32_t) cnt[k];
        std::vector<int32_t> memb(m);
        { std::vector<uint32_t> cur(off.begin(), off.end() - 1);
          for (int32_t i = 0; i < m; ++i) memb[cur[lab[i]]++] = a + i; }

        auto & MU  = clu_mu[lh];
        auto & SG  = clu_sig[lh];
        auto & LN  = clu_logn[lh];
        auto & ME  = clu_mem[lh];
        auto & OF  = clu_off[lh];
        const size_t base0 = MU.size() / P;
        MU.resize((base0 + K) * P);
        SG.resize((base0 + K) * P2);
        LN.resize(base0 + K);
        const uint32_t mbase = (uint32_t) ME.size();
        ME.insert(ME.end(), memb.begin(), memb.end());
        OF.resize(base0 + K + 1);
        for (int32_t k = 0; k <= K; ++k) OF[base0 + k] = mbase + off[k];

        std::vector<float> mu(P), sg((size_t) P2);
        for (int32_t k = 0; k < K; ++k) {
            const uint32_t s0 = off[k], s1 = off[k+1];
            const int32_t nk = (int32_t) (s1 - s0);
            LN[base0 + k] = nk > 0 ? logf((float) nk) : -1e30f;
            std::fill(mu.begin(), mu.end(), 0.0f);
            std::fill(sg.begin(), sg.end(), 0.0f);
            if (nk > 0) {
                for (uint32_t s = s0; s < s1; ++s) {
                    const float * x = X.data() + (size_t)(memb[s] - a)*P;
                    for (int32_t e = 0; e < P; ++e) mu[e] += x[e];
                }
                const float inv = 1.0f/(float) nk;
                for (int32_t e = 0; e < P; ++e) mu[e] *= inv;
                for (uint32_t s = s0; s < s1; ++s) {
                    const float * x = X.data() + (size_t)(memb[s] - a)*P;
                    int32_t o = 0;
                    for (int32_t i2 = 0; i2 < P; ++i2) {
                        const float di = x[i2] - mu[i2];
                        for (int32_t j2 = i2; j2 < P; ++j2) sg[o++] += di*(x[j2] - mu[j2]);
                    }
                }
                for (int32_t e = 0; e < P2; ++e) sg[e] *= inv;
            }
            memcpy(MU.data() + (base0 + k)*P, mu.data(), sizeof(float)*P);
            uint16_t * d = SG.data() + (base0 + k)*P2;
            for (int32_t e = 0; e < P2; ++e) d[e] = inf_f32_to_f16(sg[e]);
        }
    });

    n_cluster_chunks++;
    t_cluster += inf_now() - t0;
}

void inf_ctx::close_chunks() {
    // with the cluster tier off there is nothing to fit and nothing to score against, so the
    // k-means pass and the (n, mu, Sigma) sentinels it produces are skipped outright.  This is
    // what makes the trigger-only configuration cheaper rather than merely unused.
    if (!p.use_clusters) { n_chunk_done = (int32_t) (n_past / p.chunk); return; }
    while ((int64_t)(n_chunk_done + 1) * p.chunk <= (int64_t) n_past) {
        cluster_chunk(n_chunk_done);
        n_chunk_done++;
    }
}

// ---------------------------------------------------------------------------------------------
// selection: triggered units + top clusters per (layer, KV head) + the last rolled-off tokens
// ---------------------------------------------------------------------------------------------
void inf_ctx::select(int32_t p_now, const llama_token * up, int32_t n_up) {
    const int32_t hi = p_now - p.W;                 // tokens [n_pin+1, hi) are far
    forced.clear(); cand.clear(); sel.clear();
    stamp_cur++;
    if ((int32_t) stamp.size() < hi) { stamp.resize(std::max(2*hi, 1 << 16), 0); votes.resize(stamp.size(), 0); }

    auto push_forced = [&](int32_t t) {
        if (t <= p.n_pin || t >= hi) return;
        if (is_evicted(t)) { n_evict_blocked++; return; }   // 09-25 v2bld32 store budget: an evicted token cannot be fetched
        if (stamp[t] == stamp_cur) return;
        stamp[t] = stamp_cur; votes[t] = 1 << 20; forced.push_back(t);
    };

    // 1. the last n_roll rolled-off tokens
    if (p.use_roll) for (int32_t t = std::max(p.n_pin + 1, hi - p.n_roll); t < hi; ++t) push_forced(t);

    // 2. the units triggered by the block's rare words.  A unit scores the summed surprisal
    //    -log p(id) of the distinct non-stop query tokens it contains, so a unit that holds BOTH
    //    halves of a rare pair outranks one that holds a single moderately rare word; units are
    //    taken in score order until the trigger budget is spent.  Query tokens whose posting list
    //    is longer than max_post units are not discriminative and are skipped.
    //    The query window is the previous block plus whatever of the current block is known: both
    //    are model INPUT, so this is causal.
    if (p.use_triggers) {
        if ((int32_t) u_stamp.size() < (int32_t) units.size()) {
            u_stamp.resize(units.size() + 4096, 0);
            u_score.resize(u_stamp.size(), 0.0f);
        }
        std::vector<int32_t> hit;
        bool win_done = false;   // 09-24 unit=win<K> admitted its windows itself
        if (p.trig_word) {
            // the harness's rule: query terms are WHOLE WORDS above trig_surp nats, and each term
            // carries at most post_cap postings.  The upcoming block's words are assembled here so
            // the query still sees the block it is about to decode.
            std::unordered_map<uint64_t,float> q;
            std::unordered_set<uint64_t> seed_only;   // 09-24 per-source window: terms that came only from the seed
            const int32_t req_lo = p_now - 2*p.B;                 // the request's own words
            int32_t nreq = 0, nseed = 0;
            std::vector<inf_qword> seedcand;
            // 09-24 reqwin=0: when the hand-off supplies the request itself (n_up > 0, the
            // pre-encode ordering), the stream's trailing 2B tokens are HISTORY, not the request;
            // leave them to the seed rule instead of weighting them as request words
            const bool use_win = p.reqwin || n_up == 0;
            for (const auto & e : wlog) {
                if (e.surp <= p.trig_surp) continue;
                if (e.t > req_lo && use_win) {
                    if (q.emplace(e.h, e.surp).second) nreq++;
                    else q[e.h] = std::max(q[e.h], e.surp);       // a request word keeps full weight
                } else if (p.seed_span > 0 && e.t > p_now - p.seed_span) {
                    // seedindexed: only trunk words that HAVE a far posting list are candidates,
                    // so a cap of 1 takes the most surprising word that can actually fetch
                    // something rather than the most surprising word in the trunk
                    if (p.seed_indexed && winv.find(e.h) == winv.end()) continue;
                    // 09-25 v2bld29 seedfar (LOG 137h): a candidate must have a posting in the FAR region (a unit that
                    // closed below the trunk edge).  seedindexed only asks for SOME posting, and every trunk word has
                    // one (its own trunk unit), so it filtered nothing (SD_SI rows identical to the base, 137h).
                    if (p.seed_far) {
                        auto itf = winv.find(e.h);
                        bool far = false;
                        if (itf != winv.end()) for (int32_t u : itf->second) if (u >= 0 && u < (int32_t) units.size() && units[u].t1 <= hi) { far = true; break; }
                        if (!far) continue;
                    }
                    seedcand.push_back({e.h, (float) (p.seed_disc * e.surp), e.t});
                }
            }
            // oracle_seed replaces the whole candidate list with the one term the driver says
            // is the handle: same timing, same placement, term quality held at the oracle
            if (p.oracle_seed && oracle_h) {
                seedcand.clear();
                seedcand.push_back({oracle_h, (float)(p.seed_disc * 100.0), p_now});
            }
            // seed words enter highest-surprisal first, capped at seed_words, and never
            // displacing a request word
            std::sort(seedcand.begin(), seedcand.end(),
                      [](const inf_qword & a, const inf_qword & b) { return a.surp > b.surp; });
            const size_t scap = p.seed_words > 0 ? std::min<size_t>(seedcand.size(), (size_t) p.seed_words)
                                                 : seedcand.size();
            for (size_t i = 0; i < scap; ++i)
                if (q.emplace(seedcand[i].h, seedcand[i].surp).second) { nseed++; seed_only.insert(seedcand[i].h); }
            if (p.evt_log && p.seed_words > 0 && !seedcand.empty()) {
                printf("  SEEDPICK span %d cap %d: took", p.seed_span, p.seed_words);
                for (size_t i = 0; i < scap; ++i)
                    printf(" hash %016llx surp %.1f t=%d", (unsigned long long) seedcand[i].h,
                           seedcand[i].surp, seedcand[i].t);
                printf("  (of %zu candidates)\n", seedcand.size());
                fflush(stdout);
            }
            n_req_words += nreq; n_seed_words += nseed; n_seed_gathers++;
            if (n_up > 0) {
                std::string cw; double cs = 0.0;
                auto close = [&]() {
                    std::string w2;
                    for (unsigned char c : cw) if (isalnum(c) || c == '-' || c == '_') w2 += (char) c;
                    if (p.win_log && w2.size() >= 2 && !stopw.count(w2))
                        printf("  QUP '%s' surp %.1f %s\n", w2.c_str(), cs, cs > p.trig_surp ? "-> query term" : "(below trig_surp)");
                    if (w2.size() >= 2 && !stopw.count(w2) && cs > p.trig_surp)
                        q[inf_word_hash(w2)] = (float) cs;
                    cw.clear(); cs = 0.0;
                };
                for (int32_t i = 0; i < n_up; ++i) {
                    const std::string & pc = piece[up[i]];
                    if (inf_word_start(pc) && !cw.empty()) close();
                    cw += pc; cs += -freq.logp(up[i]);
                }
                close();
            }
            if (p.win_log) {
                last_q.assign(q.begin(), q.end());
                for (const auto & kv : q) {
                    auto io = wocc.find(kv.first); auto iu = winv.find(kv.first);
                    int nof = 0, nuf = 0;
                    if (io != wocc.end()) for (int32_t ws : io->second) if (ws < hi) nof++;
                    if (iu != winv.end()) for (int32_t u : iu->second) if (units[u].t1 <= hi) nuf++;
                    auto it2 = wtext.find(kv.first);
                    printf("  QTERM '%s' w %.1f  far occurrences posted %d (of %zu)  far units posted %d (of %zu)\n",
                           it2 == wtext.end() ? "?" : it2->second.c_str(), kv.second, nof,
                           io == wocc.end() ? (size_t) 0 : io->second.size(), nuf, iu == winv.end() ? (size_t) 0 : iu->second.size());
                }
            }
            if (p.win_k > 0) {
                // 09-24 unit=win<K> (RESULTS Q7.5.4).  Windows [ws-K, ws+K) around every posted
                // occurrence of a query term that lies in the far region, clipped to
                // [n_pin+1, far_hi); merged when they overlap (next start < current end) across
                // all terms; score = the summed weight of the DISTINCT terms of a merged window,
                // added in hash order so equal term sets score bit-identically; admitted whole in
                // score order (ties: later end, then later start) until the budget.
                win_done = true;
                struct IV { int32_t lo, hi; uint64_t h; };
                std::vector<IV> iv;
                for (const auto & kv : q) {
                    if (p.seed_sent && seed_only.count(kv.first)) continue;   // 09-24 seedsent: seed terms fetch sentence units below
                    auto it = wocc.find(kv.first);
                    if (it == wocc.end()) continue;
                    for (int32_t ws : it->second) {
                        if (ws >= hi) continue;          // a trunk posting: it used a cap slot, dropped here
                        const int32_t kk = (p.win_k_seed > 0 && seed_only.count(kv.first)) ? p.win_k_seed : p.win_k;
                        const int32_t a = std::max(p.n_pin + 1, ws - kk);
                        const int32_t b = std::min(hi, ws + kk);
                        if (a >= b) continue;
                        iv.push_back({a, b, kv.first});
                    }
                }
                std::sort(iv.begin(), iv.end(), [](const IV & x, const IV & y) {
                    return x.lo != y.lo ? x.lo < y.lo : (x.hi != y.hi ? x.hi < y.hi : x.h < y.h); });
                last_win.clear();
                for (const auto & v : iv) {
                    if (!last_win.empty() && v.lo < last_win.back().hi) {
                        auto & m = last_win.back();
                        m.hi = std::max(m.hi, v.hi);
                        if (std::find(m.terms.begin(), m.terms.end(), v.h) == m.terms.end()) m.terms.push_back(v.h);
                    } else {
                        last_win.push_back({v.lo, v.hi, 0.0, {v.h}});
                    }
                }
                for (auto & m : last_win) {
                    std::sort(m.terms.begin(), m.terms.end());
                    double sc = 0.0;
                    for (uint64_t h2 : m.terms) sc += (double) q[h2];
                    m.score = sc;
                }
                last_win_order.resize(last_win.size());
                for (size_t i = 0; i < last_win.size(); ++i) last_win_order[i] = (int32_t) i;
                std::sort(last_win_order.begin(), last_win_order.end(), [&](int32_t x, int32_t y) {
                    const auto & A = last_win[x]; const auto & B = last_win[y];
                    if (A.score != B.score) return A.score > B.score;
                    if (A.hi != B.hi) return A.hi > B.hi;
                    return A.lo > B.lo; });
                for (const auto & m : last_win) n_trig_want += m.hi - m.lo;
                const int32_t budget = (int32_t) (p.trig_frac * p.S);
                last_hit.clear();
                last_hit_taken = 0;
                last_win_taken = 0;
                if (!p.seed_sent) {
                for (int32_t i : last_win_order) {
                    if ((int32_t) forced.size() >= budget) break;
                    n_trig_units++;
                    last_win_taken++;
                    for (int32_t t = last_win[i].lo; t < last_win[i].hi; ++t) push_forced(t);
                }
                } else {
                    // 09-24 seedsent (per-source units, coordinator): terms that entered the query ONLY
                    // through the trunk seed fetch the SENTENCE units that hold them (the unit index,
                    // built by the sentence rule: seedsent sets unit_stride 0), scored as the unit path
                    // scores a unit (the summed weight of its distinct query terms; seed terms carry
                    // seed_disc x surprisal); request and event terms keep their +-win_k windows.
                    // Windows and seed units are admitted whole from ONE list in score order (ties:
                    // later end, then later start) until the budget.
                    struct SU { int32_t lo, hi; double score; int32_t win; };
                    std::vector<SU> cands;
                    for (size_t i = 0; i < last_win.size(); ++i)
                        cands.push_back({last_win[i].lo, last_win[i].hi, last_win[i].score, (int32_t) i});
                    std::vector<int32_t> uh;
                    for (uint64_t h2 : seed_only) {
                        auto it = winv.find(h2);
                        if (it == winv.end()) continue;
                        const float w = q[h2];
                        for (int32_t u : it->second) {
                            if (units[u].t1 > hi) continue;
                            if (u_stamp[u] != stamp_cur) { u_stamp[u] = stamp_cur; u_score[u] = 0.0f; uh.push_back(u); }
                            u_score[u] += w;
                        }
                    }
                    for (int32_t u : uh) { cands.push_back({units[u].t0, units[u].t1, (double) u_score[u], -1}); n_trig_want += units[u].t1 - units[u].t0; }
                    std::sort(cands.begin(), cands.end(), [](const SU & A, const SU & B) {
                        if (A.score != B.score) return A.score > B.score;
                        if (A.hi != B.hi) return A.hi > B.hi;
                        return A.lo > B.lo; });
                    for (const auto & c : cands) {
                        if ((int32_t) forced.size() >= budget) break;
                        n_trig_units++;
                        const size_t f0 = forced.size();
                        for (int32_t t = c.lo; t < c.hi; ++t) push_forced(t);
                        if (c.win >= 0) last_win_taken++;
                        else { n_seedu_units++; n_seedu_tok += (int64_t) (forced.size() - f0); }
                    }
                    n_seedu_cand += (int64_t) uh.size();
                }
                if (p.win_log) {
                    printf("  WINSEL %zu merged windows, %d admitted, %zu tokens forced (budget %d); top by score:\n",
                           last_win.size(), last_win_taken, forced.size(), budget);
                    for (size_t r = 0; r < last_win_order.size() && r < 12; ++r) {
                        const auto & m = last_win[last_win_order[r]];
                        printf("    #%zu [%d,%d) len %d score %.1f terms %zu%s\n", r, m.lo, m.hi, m.hi - m.lo, m.score,
                               m.terms.size(), (int32_t) r < last_win_taken ? "" : "  (not admitted)");
                    }
                }
            } else
            for (const auto & kv : q) {
                auto it = winv.find(kv.first);
                if (it == winv.end()) continue;
                for (int32_t u : it->second) {
                    if (units[u].t1 > hi) continue;
                    if (u_stamp[u] != stamp_cur) { u_stamp[u] = stamp_cur; u_score[u] = 0.0f; hit.push_back(u); }
                    u_score[u] += kv.second;
                }
            }
        } else {
        std::unordered_set<llama_token> q;
        for (int32_t t = std::max(0, p_now - p.B); t < p_now; ++t) q.insert(toks[t]);
        for (int32_t i = 0; i < n_up; ++i) q.insert(up[i]);
        const int32_t cap = std::min<int32_t>(p.max_post, std::max<int32_t>(16, (int32_t) units.size()/4));
        for (llama_token id : q) {
            if (tok_stop[id]) continue;
            auto it = inv.find(id);
            if (it == inv.end()) continue;
            if ((int32_t) it->second.size() > cap) continue;
            const float w = (float) -freq.logp(id);
            for (int32_t u : it->second) {
                if (units[u].t1 > hi) break;
                if (u_stamp[u] != stamp_cur) { u_stamp[u] = stamp_cur; u_score[u] = 0.0f; hit.push_back(u); }
                u_score[u] += w;
            }
        }
        }
        if (!win_done) {
        // 09-25 v2bld28 sentfill (coordinator, LOG 137f): the neighbouring sentences of every hit unit (+-1, then
        // +-2) enter as secondary units at a discounted score (the hit's score x sfd1, x sfd2), so the admission
        // below keeps taking units in score order until the band budget S is reached.  A unit's score is the max of
        // its own trigger score and the best neighbour score.  Triggers, the seed and events are unchanged.
        if (p.sent_fill > 0 && p.trig_word && !hit.empty()) {
            std::vector<std::pair<int32_t,float>> add;
            for (int32_t u : hit) {
                const float s0 = u_score[u];
                for (int32_t d = 1; d <= p.sent_fill; ++d) {
                    const float w = s0 * (float) (d == 1 ? p.sf_d1 : (d == 2 ? p.sf_d2 : p.sf_d2 * std::pow(p.sf_d1, d - 2)));
                    for (int32_t nb : { u - d, u + d }) {
                        if (nb < 0 || nb >= (int32_t) units.size()) continue;
                        if (units[nb].t1 > hi) continue;
                        add.push_back({nb, w});
                    }
                }
            }
            for (const auto & a : add) {
                const int32_t nb = a.first;
                if (u_stamp[nb] != stamp_cur) { u_stamp[nb] = stamp_cur; u_score[nb] = a.second; hit.push_back(nb); n_sf_cand++; }
                else if (a.second > u_score[nb]) u_score[nb] = a.second;
            }
        }
        std::sort(hit.begin(), hit.end(), [&](int32_t x, int32_t y) { return u_score[x] > u_score[y]; });
        // how many tokens the trigger WANTED, before the budget cut: reach coverage is what fits
        // divided by this, and it is the number that says whether a smaller band would truncate.
        for (int32_t u : hit) n_trig_want += units[u].t1 - units[u].t0;
        const int32_t budget = (int32_t) (p.trig_frac * p.S);
        last_hit = hit;
        last_hit_taken = 0;
        for (int32_t u : hit) {
            if ((int32_t) forced.size() >= budget) break;
            n_trig_units++;
            last_hit_taken++;
            for (int32_t t = units[u].t0; t < units[u].t1; ++t) push_forced(t);
        }
        }
    }
    n_trig_tok += (int64_t) forced.size();
    note_use_tokens(forced);   // 09-25 v2bld32: usage at exit (admissions per unit)

    // 3. the top-`topc` clusters per (layer, KV head), scored against the block's mean query
    const int32_t nch = std::min(n_chunk_done, hi / p.chunk);
    if (p.use_clusters && have_q && nch > 0) {
        const double ts = inf_now();
        const int32_t ncl = nch * p.n_clu;
        std::vector<int32_t> picks((size_t) n_lh * p.topc, -1);
        inf_parallel(p.nthread, n_lh, [&](int32_t lh) {
            const float * q  = qmean.data() + (size_t) lh * P;
            const float * MU = clu_mu[lh].data();
            const float * LN = clu_logn[lh].data();
            const int32_t nsh = std::min(ncl, p.shortlist);
            // bounded min-heap over the linear part: O(ncl) with O(k log n) insertions, where
            // an nth_element over every cluster would dominate the block at 1M tokens
            std::vector<std::pair<float,int32_t>> H;
            H.reserve(nsh);
            auto gt = [](const std::pair<float,int32_t> & x, const std::pair<float,int32_t> & y) { return x.first > y.first; };
            float thr = -1e30f;
            for (int32_t c = 0; c < ncl; ++c) {
                const float * mu = MU + (size_t) c*P;
                float d = LN[c];
                for (int32_t e = 0; e < P; ++e) d += q[e]*mu[e];
                if ((int32_t) H.size() < nsh) {
                    H.push_back({d, c}); std::push_heap(H.begin(), H.end(), gt);
                    if ((int32_t) H.size() == nsh) thr = H.front().first;
                } else if (d > thr) {
                    std::pop_heap(H.begin(), H.end(), gt);
                    H.back() = {d, c};
                    std::push_heap(H.begin(), H.end(), gt);
                    thr = H.front().first;
                }
            }
            // full quadratic term on the shortlist
            std::vector<float> Q2((size_t) P2);
            { int32_t o = 0;
              for (int32_t i = 0; i < P; ++i)
                  for (int32_t j = i; j < P; ++j) Q2[o++] = (i == j ? 0.5f : 1.0f) * q[i]*q[j]; }
            const float * tab = inf_f16tab();
            const uint16_t * SG = clu_sig[lh].data();
            std::vector<std::pair<float,int32_t>> sc((size_t) H.size());
            for (size_t z = 0; z < H.size(); ++z) {
                const int32_t c = H[z].second;
                const uint16_t * sg = SG + (size_t) c*P2;
                float d = 0;
                for (int32_t e = 0; e < P2; ++e) d += Q2[e]*tab[sg[e]];
                sc[z] = { H[z].first + d, c };
            }
            const int32_t nt = std::min<int32_t>(p.topc, (int32_t) sc.size());
            std::partial_sort(sc.begin(), sc.begin() + nt, sc.end(),
                              [](const std::pair<float,int32_t> & x, const std::pair<float,int32_t> & y) { return x.first > y.first; });
            for (int32_t z = 0; z < nt; ++z) picks[(size_t) lh*p.topc + z] = sc[z].second;
        });
        t_score += inf_now() - ts;
        t_score_blocks += 1.0;

        for (int32_t lh = 0; lh < n_lh; ++lh) {
            const auto & OF = clu_off[lh];
            const auto & ME = clu_mem[lh];
            for (int32_t z = 0; z < p.topc; ++z) {
                const int32_t c = picks[(size_t) lh*p.topc + z];
                if (c < 0) continue;
                for (uint32_t s = OF[c]; s < OF[c+1]; ++s) {
                    const int32_t t = ME[s];
                    if (t <= p.n_pin || t >= hi) continue;
                    if (stamp[t] != stamp_cur) { stamp[t] = stamp_cur; votes[t] = 0; cand.push_back(t); }
                    if (votes[t] < (1 << 20)) votes[t]++;
                }
            }
        }
    }

    // 4. union, capped at S: forced first, then the cluster candidates by vote count
    sel = forced;
    if ((int32_t) sel.size() > p.S) sel.resize(p.S);
    if ((int32_t) sel.size() < p.S && !cand.empty()) {
        const int32_t room = p.S - (int32_t) sel.size();
        // drop the tokens that are already forced
        std::vector<int32_t> c2;
        c2.reserve(cand.size());
        for (int32_t t : cand) if (votes[t] < (1 << 20)) c2.push_back(t);
        if ((int32_t) c2.size() > room) {
            std::nth_element(c2.begin(), c2.begin() + room, c2.end(),
                             [&](int32_t x, int32_t y) { return votes[x] > votes[y]; });
            c2.resize(room);
        }
        n_clu_tok += (int64_t) c2.size();
        sel.insert(sel.end(), c2.begin(), c2.end());
    }
    if (p.win_log) last_sel_trig = sel;   // 09-24 T27: the admitted set before any fill
    last_hi = hi;
    // 5. the band is a fixed S cells and must stay full (see place_band): back-fill the rest with
    //    the most recent far tokens.  n_band_sel / (S * gathers) reports how much of the band the
    //    index and the sentinels actually claimed.
    n_sel_real += (int64_t) sel.size();

    // the tokens the INDEX asked for this gather (before any fill), so the band's provenance and
    // composition are decidable rather than inferred
    smark_hit.clear();
    for (int32_t t : sel) smark_hit.insert(t);

    // 5a. fill_keep: top up with what the band ALREADY holds, newest first, before falling back to
    //     recency.  Everything still resident is then protected from the ring for another gather,
    //     which is what gives a triggered unit a residency longer than one selection.
    if (p.fill_keep && (int32_t) sel.size() < p.S) {
        std::vector<int32_t> held;
        held.reserve(p.S);
        for (int32_t c = 0; c < p.S; ++c) {
            const int32_t t = band_tok[c];
            if (t <= p.n_pin || t >= hi) continue;
            if (stamp[t] == stamp_cur) continue;
            held.push_back(t);
        }
        std::sort(held.begin(), held.end(), std::greater<int32_t>());
        for (int32_t t : held) {
            if ((int32_t) sel.size() >= p.S) break;
            if (stamp[t] == stamp_cur) continue;
            stamp[t] = stamp_cur; votes[t] = 0;
            sel.push_back(t);
        }
    }

    for (int32_t t = hi - 1; t > p.n_pin && (int32_t) sel.size() < p.S; --t) {
        if (stamp[t] == stamp_cur) continue;
        if (is_evicted(t)) continue;   // 09-25 v2bld32 store budget
        stamp[t] = stamp_cur; votes[t] = 0;
        sel.push_back(t);
    }
    std::sort(sel.begin(), sel.end(), std::greater<int32_t>());
    if (smark.size() != stamp.size()) smark.resize(stamp.size(), 0);
    for (int32_t t : sel) smark[t] = stamp_cur;
}

// ---------------------------------------------------------------------------------------------
// place the selected tokens in the band.
//
// The band is always FULL: S cells, every one of them holding a real far token at a real slot
// position.  llama_batch_allocr requires the cache's maximum position to be exactly (query - 1),
// so a cell can be neither empty (find_slot would take it and break the contiguous band) nor
// parked above the query.  Cells the selection does not ask for are therefore overwritten in ring
// order by the ones it does; whatever survives keeps a slot.  Ranks are assigned by recency over
// the band's ACTUAL contents, so the placement invariant (newer = nearer) always holds.
// ---------------------------------------------------------------------------------------------

// =============================================================================================
// Event-driven refresh.
//
// The block-boundary gather can only react to a rare word at the NEXT multiple of B, and during
// generation it cannot react to a word the model itself just produced until up to B-1 tokens
// later.  The event path closes that: the policy assembles whole words as their pieces arrive,
// and the moment a word turns out to be rare it pulls that word's posting-list units into the
// band, in prefill and in decode alike.
//
// The write reuses the VICTIM CELL'S OWN SLOT POSITION rather than re-ranking the band, so an
// event costs one contiguous run of kv_cells_write and no kv_cells_place at all.  The band's set
// of slot positions is therefore invariant between block gathers, which is what keeps
// llama_batch_allocr happy (max position == query - 1); only the contents move.
// =============================================================================================

// a word starts at a piece that begins with a space, and at the first token of the stream
static inline bool inf_word_start(const std::string & pc) {
    return pc.empty() || pc[0] == ' ' || pc[0] == '\n' || pc[0] == '\r' || pc[0] == '\t';
}

static inline uint64_t inf_word_hash(const std::string & w) {
    uint64_t h = 1469598103934665603ull;                 // FNV-1a
    for (unsigned char c : w) {
        if (c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');   // case-fold
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
        h ^= c; h *= 1099511628211ull;
    }
    return h;
}

void inf_ctx::word_flush(int32_t t_end) {
    if (wcur.empty()) { wsurp = 0.0; wt0 = t_end; return; }

    // strip surrounding punctuation so "cactus." and "cactus" are the same word
    std::string w;
    for (unsigned char c : wcur) if (isalnum(c) || c == '-' || c == '_') w += (char) c;

    if (w.size() >= 2 && !stopw.count(w)) {
        const uint64_t h = inf_word_hash(w);

        // the query side: every word of the recent past, with its surprisal, so select() can
        // apply the same threshold to the query as the index applied to the document
        wlog.push_back({h, (float) wsurp, t_end});
        if (wlog.size() > 4096) wlog.erase(wlog.begin(), wlog.begin() + 2048);

        // the index side: only words above the threshold are worth a posting at all
        if (!p.trig_word || wsurp > p.trig_surp) { wpend.push_back(h); wpend_t0.push_back(wt0); }

        // 09-24 unit=win<K>: one posting per occurrence (its first token), the last post_cap
        if (p.win_log) wtext[h] = w;   // 09-24 diagnostic: hash -> text
        if (p.win_k > 0) {
            if (p.win_log) wocc_log.push_back({h, wt0, wsurp, w});
            if (wsurp > p.trig_surp) {
                auto & v = wocc[h];
                v.push_back(wt0);
                if (p.post_cap > 0 && (int32_t) v.size() > p.post_cap)
                    v.erase(v.begin(), v.begin() + (v.size() - p.post_cap));
            }
        }

        // RARE?  Summed surprisal over the word's pieces, in nats.  A word the tokeniser splits
        // into several pieces accumulates their surprisal, which is the point of keying on words:
        // no single piece of "cactus" need be rare for "cactus" to be.
        // 09-24 reqfetch=0: events only on GENERATED words (not on the request while it is encoded)
        if (p.evt && two_tier && wsurp >= p.evt_surp && (p.reqfetch || in_decode)) {
            // The fourth corner of the crossing: the FIRST event of a generation fires the SEED's
            // gather rather than this word's postings.  select() + place_band() are used directly,
            // so it is the same ranking, the same cap and the same placement as the hand-off seed
            // and only the MOMENT differs.  Once per generation; the normal event path resumes.
            if (p.evt_seed_span > 0 && in_decode && !evt_seed_done) {
                evt_seed_done = true;
                const int32_t save = p.seed_span;
                p.seed_span = p.evt_seed_span;
                select(n_past, nullptr, 0);
                place_band(n_past);
                p_place = n_past;
                p.seed_span = save;
                n_evt_seed_gathers++;
                if (p.evreenc && in_decode) {   // 09-24: an event-time gather too
                    if (p.evr_surp > 0.0 && wsurp < p.evr_surp) n_reenc_surp_skip++;
                    else reenc_want = true;
                }
                if (p.evt_log)
                    printf("  EVTSEED t=%d word '%s': fired the full seed gather (span %d) at the event\n",
                           t_end, w.c_str(), p.evt_seed_span);
                wcur.clear(); wsurp = 0.0; wt0 = t_end;
                return;
            }
            n_evt_try++;
            auto it = winv.find(h);
            if (p.win_k > 0) {
                // 09-24 unit=win<K>: the windows of the word's last evt_cap occurrences
                auto io = wocc.find(h);
                if (io == wocc.end() || io->second.empty()) {
                    n_evt_nohash++;
                    if (p.evt_log)
                        printf("  EVTLOG t=%d word '%s' hash %016llx surp %.1f: NOT IN INDEX (win)\n",
                               t_end, w.c_str(), (unsigned long long) h, wsurp);
                } else {
                    const std::vector<int32_t> & lst = io->second;
                    std::vector<std::array<int32_t,3>> sp;
                    const int32_t take = std::min<int32_t>(p.evt_cap, (int32_t) lst.size());
                    for (int32_t i = 0; i < take; ++i) {
                        const int32_t ws = lst[lst.size() - 1 - i];
                        sp.push_back({ws - p.win_k, ws + p.win_k, ws + 1});
                    }
                    const int64_t f0 = n_evt_fetch, t0c = n_evt_trunk, r0 = n_evt_resident, c0 = n_event_cells;
                    event_gather_spans(sp, t_end);
                    if (p.evreenc && in_decode && n_evt_fetch > f0) {
                        if (p.evr_surp > 0.0 && wsurp < p.evr_surp) n_reenc_surp_skip++;   // 09-24 evr_surp
                        else reenc_want = true;
                    }
                    if (p.evt_log) {
                        printf("  EVTLOG t=%d word '%s' hash %016llx surp %.1f: %zu postings, windows", t_end, w.c_str(),
                               (unsigned long long) h, wsurp, lst.size());
                        for (auto & s : sp) printf(" [%d,%d)", s[0], s[1]);
                        printf("  -> %lld cells  [%s]\n", (long long) (n_event_cells - c0),
                               n_evt_fetch > f0 ? "fetched" : (n_evt_trunk > t0c ? "all in trunk" :
                               (n_evt_resident > r0 ? "already resident" : "nothing")));
                    }
                }
            } else
            if (it == winv.end() || it->second.empty()) {
                n_evt_nohash++;
                if (p.evt_log)
                    printf("  EVTLOG t=%d word '%s' hash %016llx surp %.1f: NOT IN INDEX\n",
                           t_end, w.c_str(), (unsigned long long) h, wsurp);
            } else {
                const std::vector<int32_t> & lst = it->second;
                std::vector<int32_t> us;
                const int32_t take = std::min<int32_t>(p.evt_cap, (int32_t) lst.size());
                for (int32_t i = 0; i < take; ++i) us.push_back(lst[lst.size() - 1 - i]);
                const int64_t f0 = n_evt_fetch, t0c = n_evt_trunk, r0 = n_evt_resident;
                event_gather(us, t_end);
                // 09-24 evreenc: something new entered the band during decode -> re-encode the
                // tail once, at the end of this eval() (after n_past has advanced)
                if (p.evreenc && in_decode && n_evt_fetch > f0) {
                    if (p.evr_surp > 0.0 && wsurp < p.evr_surp) n_reenc_surp_skip++;   // 09-24 evr_surp
                    else reenc_want = true;
                }
                if (p.evt_log) {
                    // how many of the pulled units' far tokens are resident NOW, i.e. just before
                    // the next generated token, which for a needle is the first digit
                    int32_t res = 0, tot = 0;
                    const int32_t hi = p_place - p.W;
                    for (int32_t u : us) {
                        if (u < 0 || u >= (int32_t) units.size()) continue;
                        for (int32_t t = units[u].t0; t < units[u].t1; ++t) {
                            if (t <= p.n_pin || t >= hi) continue;
                            tot++; if (band_of.count(t)) res++;
                        }
                    }
                    int32_t rmin = -1, rmax = -1, hres = 0, htot = 0;
                    for (int32_t u : us) {
                        if (u < 0 || u >= (int32_t) units.size()) continue;
                        for (int32_t t = units[u].t0; t < units[u].t1; ++t) {
                            auto bi = band_of.find(t);
                            if (bi == band_of.end()) continue;
                            const int32_t rk = (int32_t) last_rank.size() > bi->second ? last_rank[bi->second] : -1;
                            if (rk >= 0) { if (rmin < 0 || rk < rmin) rmin = rk; if (rk > rmax) rmax = rk; }
                        }
                        for (int32_t t = units[u].t0 - 96; t < units[u].t1 + 96; ++t) {
                            if (t <= p.n_pin || t >= hi) continue;
                            if (t >= units[u].t0 && t < units[u].t1) continue;
                            htot++; if (band_of.count(t)) hres++;
                        }
                    }
                    printf("  EVTLOG t=%d word '%s' hash %016llx surp %.1f: %zu postings, units",
                           t_end, w.c_str(), (unsigned long long) h, wsurp, lst.size());
                    for (int32_t u : us) printf(" %d[%d,%d)", u, units[u].t0, units[u].t1);
                    printf("  -> resident %d/%d  slot rank [%d,%d] dist [%d,%d]  halo +/-96 resident %d/%d  [%s]\n",
                           res, tot, rmin, rmax,
                           rmin >= 0 ? p.W + 1 + rmin : -1, rmax >= 0 ? p.W + 1 + rmax : -1, hres, htot,
                           n_evt_fetch > f0 ? "fetched" : (n_evt_trunk > t0c ? "all in trunk" :
                           (n_evt_resident > r0 ? "already resident" : "nothing")));
                }
            }
        }
    }
    wcur.clear(); wsurp = 0.0; wt0 = t_end;
}

void inf_ctx::word_feed(int32_t t, llama_token id) {
    const std::string & pc = piece[id];
    if (inf_word_start(pc) && !wcur.empty()) word_flush(t);
    if (wcur.empty()) wt0 = t;
    wcur += pc;
    wsurp += -freq.logp(id);
}

void inf_ctx::event_gather(const std::vector<int32_t> & us, int32_t t_now) {
    // 09-24: units become spans (t0 - halo, t1 + halo, anchor t1); the fetch itself is unchanged
    std::vector<std::array<int32_t,3>> sp;
    for (int32_t u : us) {
        if (u < 0 || u >= (int32_t) units.size()) continue;
        sp.push_back({units[u].t0 - p.evt_halo, units[u].t1 + p.evt_halo, units[u].t1});
    }
    event_gather_spans(sp, t_now);
}

void inf_ctx::event_gather_spans(const std::vector<std::array<int32_t,3>> & sp, int32_t t_now) {
    (void) t_now;
    flush_capture();   // 09-24 PERF capblk
    if (!two_tier || sp.empty()) return;
    const int32_t hi = p_place - p.W;             // the far region as of the last placement
    if (hi <= p.n_pin + 1) return;

    std::vector<int32_t> need;
    std::unordered_set<int32_t> in_need;   // 09-23 (T8): overlapping units/halos must not push a token twice
    int32_t n_in_trunk = 0, n_considered = 0, n_already = 0;
    for (const auto & s : sp) {
        n_considered++;
        if (s[2] > hi) { n_in_trunk++; continue; }   // still in the trunk: nothing to fetch
        n_event_units++;
        for (int32_t t = s[0]; t < s[1]; ++t) {
            if (t <= p.n_pin || t >= hi) continue;
            if (band_of.count(t)) { n_already++; continue; }   // already resident
            if (is_evicted(t)) { n_evict_blocked++; continue; }  // 09-25 v2bld32 store budget
            if (!in_need.insert(t).second) continue;           // already queued by an overlapping unit
            need.push_back(t);
            if ((int32_t) need.size() >= p.evt_max) break;
        }
        if ((int32_t) need.size() >= p.evt_max) break;
    }
    if (need.empty()) {
        n_event_empty++;
        if (n_considered && n_in_trunk == n_considered) n_evt_trunk++;
        else if (n_already)                             n_evt_resident++;
        return;
    }
    if (last_rank.empty()) { n_event_empty++; return; }   // no placement yet: nothing to inherit
    note_use_tokens(need);   // 09-25 v2bld32: usage at exit

    const double t0 = inf_now();
    size_t wi = 0;
    while (wi < need.size()) {
        const int32_t c0 = ring;
        int32_t m = (int32_t) std::min<size_t>(need.size() - wi,
                        (size_t) std::min(p.S - c0, INF_WRITE_MAX));
        if (m <= 0) { ring = 0; continue; }

        // a victim must already carry a slot position for the new token to inherit
        while (m > 0 && last_rank[c0 + m - 1] < 0) m--;
        if (m <= 0) { ring = (c0 + 1) % p.S; continue; }

        for (int32_t j = 0; j < m; ++j) {
            const int32_t old = band_tok[c0 + j];
            if (old >= 0) band_of.erase(old);
        }
        src_pos.resize(m); dst_pos.resize(m);
        std::vector<llama_token> tokv(m);
        for (int32_t j = 0; j < m; ++j) {
            const int32_t t = need[wi + j];
            tokv[j]    = t;
            dst_pos[j] = p_place - (p.W + 1 + last_rank[c0 + j]);        // inherit the victim's slot
            src_pos[j] = p.prerope ? dst_pos[j] : t;   // 09-23: prerope key is rotated once, below
            band_tok[c0 + j] = t;
            band_of[t] = c0 + j;
        }
        inf_parallel(p.nthread, n_layer, [&](int32_t li) {
            uint8_t * dk = pin_k + (size_t) li*m*row;
            uint8_t * dv = pin_v + (size_t) li*m*row;
            for (int32_t j = 0; j < m; ++j) {
                const int32_t t = need[wi + j];
                memcpy(dk + (size_t) j*row, host.k(t) + (size_t) li*row, row);
                memcpy(dv + (size_t) j*row, host.v(t) + (size_t) li*row, row);
                if (p.prerope) rope_row_f16(dk + (size_t) j*row, dst_pos[j]);
            }
        });
        llama_memory_kv_cells_write(mem, seq, band_c0 + (uint32_t) c0, (size_t) m,
                                    src_pos.data(), dst_pos.data(), pin_k, pin_v, tokv.data());
        const int64_t nb = 2ll * m * n_layer * (int64_t) row;
        bytes_h2d += nb;
        if (in_decode) { bytes_h2d_decode += nb; n_event_cells_decode += m; }
        n_new_cells   += m;
        n_event_cells += m;
        n_event_runs++;
        wi  += m;
        ring = (c0 + m) % p.S;
    }
    n_events++;
    n_evt_fetch++;
    if (in_decode) n_events_decode++;
    t_place += inf_now() - t0;
    if (p.bandchk) band_check("event", p_place);   // 09-24: event cells inherit slots of the last placement
}

// re-place the band without selecting anything: used when the event path owns the contents and
// the block boundary is only there to keep ranks and the trunk edge honest.
void inf_ctx::rank_only(int32_t p_now) {
    cell_idx.clear(); dst_pos.clear();
    std::vector<std::pair<int32_t,int32_t>> live;
    live.reserve(p.S);
    for (int32_t c = 0; c < p.S; ++c) if (band_tok[c] >= 0) live.push_back({band_tok[c], c});
    std::sort(live.begin(), live.end(), std::greater<std::pair<int32_t,int32_t>>());
    cell_idx.resize(live.size()); dst_pos.resize(live.size());
    if ((int32_t) last_rank.size() != p.S) last_rank.assign(p.S, -1);
    std::fill(last_rank.begin(), last_rank.end(), -1);
    for (size_t r = 0; r < live.size(); ++r) {
        cell_idx[r] = band_c0 + (uint32_t) live[r].second;
        dst_pos[r]  = p_now - (p.W + 1 + (int32_t) r);
        last_rank[live[r].second] = (int32_t) r;
    }
    if (!cell_idx.empty()) {
        if (p.prerope) place_rotated();   // one rotation, from the stored pre-RoPE key
        else llama_memory_kv_cells_place(mem, seq, cell_idx.data(), dst_pos.data(), cell_idx.size());
    }
    n_band_live += (int64_t) live.size();
    n_gather++;
    if (p.bandchk) band_check("rank", p_now);   // 09-24
}

void inf_ctx::place_band(int32_t p_now) {
    flush_capture();   // 09-24 PERF capblk (the event-time seed gather calls this directly)
    const double t0 = inf_now();
    wrote_cell.clear(); wrote_tok.clear();
    if (p.syncall) llama_synchronize(lctx);

    // what the selection asks for that is not already resident
    std::vector<int32_t> need;
    need.reserve(sel.size());
    int32_t n_hit = 0;
    for (int32_t t : sel) {
        if (band_of.count(t)) n_hit++;
        else                  need.push_back(t);
    }

    // Append at a RING head, so a gather is one or two CONTIGUOUS runs and therefore one or two
    // ggml_backend_tensor_set() calls per layer.  Writing into the (scattered) cells the selection
    // dropped moves ~1.5x fewer bytes, but it costs ~900 runs per gather, and a tensor_set
    // synchronises the stream every call (~11 us): measured 645 ms per gather against 60 ms here.
    // A cell the ring overwrites while it was still wanted is fetched again on a later block;
    // that is the 1.5x.
    size_t wi = 0;
    while (wi < need.size()) {
        const int32_t c0 = ring;
        int32_t m = (int32_t) std::min<size_t>(need.size() - wi, (size_t) std::min(p.S - c0, INF_WRITE_MAX));
        if (m <= 0) { ring = 0; continue; }
        for (int32_t j = 0; j < m; ++j) {
            const int32_t old = band_tok[c0 + j];
            if (old >= 0) band_of.erase(old);
        }
        inf_parallel(p.nthread, n_layer, [&](int32_t li) {
            uint8_t * dk = pin_k + (size_t) li*m*row;
            uint8_t * dv = pin_v + (size_t) li*m*row;
            for (int32_t j = 0; j < m; ++j) {
                const int32_t t = need[wi + j];
                memcpy(dk + (size_t) j*row, host.k(t) + (size_t) li*row, row);
                memcpy(dv + (size_t) j*row, host.v(t) + (size_t) li*row, row);
            }
        });
        src_pos.resize(m); dst_pos.resize(m);
        for (int32_t j = 0; j < m; ++j) {
            const int32_t t = need[wi + j];
            src_pos[j] = t;    // the position the stored key was RoPEd at
            dst_pos[j] = t;    // no shift here: the slot comes from the ranking pass below
            band_tok[c0 + j] = t;
            band_of[t] = c0 + j;
            if ((int32_t) band_src.size() == p.S) band_src[c0 + j] = (smark_hit.count(t) ? 1 : 0);
            if ((int32_t) band_shifts.size() == p.S) band_shifts[c0 + j] = 0;
        }
        llama_memory_kv_cells_write(mem, seq, band_c0 + (uint32_t) c0, (size_t) m,
                                    src_pos.data(), dst_pos.data(), pin_k, pin_v, src_pos.data());
        for (int32_t j = 0; j < m; ++j) { wrote_cell.push_back(c0 + j); wrote_tok.push_back(need[wi + j]); }
        const int64_t nb = 2ll * m * n_layer * (int64_t) row;
        bytes_h2d += nb;
        if (in_decode) bytes_h2d_decode += nb;
        n_new_cells += m;
        n_runs++;
        wi  += m;
        ring = (c0 + m) % p.S;
    }

    // rank the band's actual contents by recency and place every cell
    cell_idx.clear(); dst_pos.clear();
    std::vector<std::pair<int32_t,int32_t>> live;   // (token, cell)
    live.reserve(p.S);
    for (int32_t c = 0; c < p.S; ++c) if (band_tok[c] >= 0) live.push_back({band_tok[c], c});
    std::sort(live.begin(), live.end(), std::greater<std::pair<int32_t,int32_t>>());
    cell_idx.resize(live.size()); dst_pos.resize(live.size());
    if ((int32_t) last_rank.size() != p.S) last_rank.assign(p.S, -1);
    std::fill(last_rank.begin(), last_rank.end(), -1);
    if ((int32_t) band_src.size()    != p.S) band_src.assign(p.S, 0);
    if ((int32_t) band_shifts.size() != p.S) band_shifts.assign(p.S, 0);
    int64_t cur = 0, kept = 0, fill = 0;
    for (size_t r = 0; r < live.size(); ++r) {
        const int32_t c = live[r].second;
        cell_idx[r] = band_c0 + (uint32_t) c;
        dst_pos[r]  = p_now - (p.W + 1 + (int32_t) r);
        last_rank[c] = (int32_t) r;
        band_shifts[c]++;                       // every re-placement re-rotates this cell's key
        if (band_shifts[c] > n_shift_max) n_shift_max = band_shifts[c];
        if (!band_src[c])                 fill++;
        else if (smark_hit.count(band_tok[c])) cur++;
        else                              kept++;
    }
    if (!cell_idx.empty()) {
        if (p.prerope) place_rotated();   // one rotation, from the stored pre-RoPE key
        else llama_memory_kv_cells_place(mem, seq, cell_idx.data(), dst_pos.data(), cell_idx.size());
    }
    n_comp_cur += cur; n_comp_kept += kept; n_comp_fill += fill; n_comp_gathers++;

    n_kept_cells += n_hit;
    n_band_live  += (int64_t) live.size();
    n_gather++;

    if (p.syncall) llama_synchronize(lctx);
    if (p.verify) verify_band(p_now);

    t_place += inf_now() - t0;
    if (p.bandchk) band_check("gather", p_now);   // 09-24, outside t_place
}

// 09-24 band check (throughput agent).  Reads every cell of the cache and checks its contents
// against the host store under the cell's own provenance.  Band cells [band_c0, band_c0 + S):
// provenance = the cell's ext tok (kv_cells_write stores the token POSITION there), slot = the
// cell's position.  Other cells at positions >= p_place - W are the trunk: provenance = the true
// position.  Pins (below the trunk, outside the band) are T25's.  prerope: expected K = host K
// rotated by the slot; otherwise by slot - t, which is only valid once the queued shift has been
// applied, so the band K check is skipped under prerope=0.
void inf_ctx::band_check(const char * where, int32_t p_now, bool trunk_only) {
    if (!two_tier && !trunk_only) return;
    const bool bandmode = two_tier;   // single tier: every cell at a true position is "trunk"
    flush_capture();   // 09-24 PERF capblk: the trunk check reads the host store
    const double t0 = inf_now();
    llama_synchronize(lctx);
    const int32_t NC = n_ctx;
    std::vector<llama_pos>   cpos(NC);
    std::vector<llama_token> ctok(NC);
    llama_memory_kv_cells_state(mem, seq, 0, (size_t) NC, cpos.data(), ctok.data());

    const int32_t tr_lo = p_now - p.W;   // p_now = the placement position (p_place is set after place_band returns)
    std::unordered_map<int32_t, int32_t> seen;   // band provenance -> count
    int64_t live = 0, meta = 0, slot = 0, dup = 0, vbad = 0, kbad = 0;
    int64_t tlive = 0, tvbad = 0, tkbad = 0;
    double  kmin = 2.0, tkmin = 2.0;
    for (int32_t c = 0; bandmode && c < p.S; ++c) {
        const int32_t cc = (int32_t) band_c0 + c;
        const int32_t got = cpos[cc] < 0 ? -1 : (int32_t) ctok[cc];
        if (got != band_tok[c]) meta++;
        if (got >= 0 && ++seen[got] == 2) dup++;
        if (got >= 0 && (int32_t) last_rank.size() == p.S && last_rank[c] >= 0 &&
            cpos[cc] != p_now - (p.W + 1 + last_rank[c])) slot++;
    }

    const int32_t CH = 1024;
    std::vector<uint8_t> kb((size_t) CH * n_layer * row), vb(kb.size());
    std::vector<uint32_t> cl(CH);
    const float * f16 = inf_f16tab();
    const int32_t np = (int32_t) theta.size();
    std::vector<int32_t> bad_here;
    for (int32_t c0 = 0; c0 < NC; c0 += CH) {
        const int32_t m = std::min(CH, NC - c0);
        // skip chunks with no cell to check
        bool any = false;
        for (int32_t j = 0; j < m && !any; ++j) {
            const int32_t cc = c0 + j;
            if (cpos[cc] < 0) continue;
            if ((bandmode && cc >= (int32_t) band_c0) || cpos[cc] >= tr_lo) any = true;
        }
        if (!any) continue;
        for (int32_t j = 0; j < m; ++j) cl[j] = (uint32_t) (c0 + j);
        llama_memory_kv_cells_read(mem, seq, cl.data(), (size_t) m, kb.data(), vb.data());
        // per cell: kind (0 skip, 1 band, 2 trunk), provenance, rotation
        std::vector<int32_t> kind(m, 0), prov(m, -1);
        std::vector<double>  cs((size_t) m * np), sn((size_t) m * np);
        for (int32_t j = 0; j < m; ++j) {
            const int32_t cc = c0 + j;
            if (cpos[cc] < 0) continue;
            int32_t t = -1, k = 0;
            if (bandmode && cc >= (int32_t) band_c0) { t = (int32_t) ctok[cc]; k = 1; }
            else if (cpos[cc] >= tr_lo)  { t = cpos[cc];           k = 2; }
            if (k == 0 || t < 0 || t >= host.n_tok) continue;
            kind[j] = k; prov[j] = t;
            const double delta = p.prerope ? (double) cpos[cc] : (double) (cpos[cc] - t);
            for (int32_t pl = 0; pl < np; ++pl) {
                const double a = (double) theta[pl] * delta;
                cs[(size_t) j*np + pl] = cos(a); sn[(size_t) j*np + pl] = sin(a);
            }
        }
        std::vector<char>   vb_bad(m, 0), kb_bad(m, 0);
        std::vector<double> kmin_c(m, 2.0);
        std::mutex mu;
        inf_parallel(p.nthread, n_layer, [&](int32_t li) {
            std::vector<double> ex(row / 2);
            std::vector<char>   lv(m, 0), lk(m, 0);
            std::vector<double> lmin(m, 2.0);
            for (int32_t j = 0; j < m; ++j) {
                if (!kind[j]) continue;
                const int32_t t = prov[j];
                if (memcmp(vb.data() + ((size_t) li*m + j)*row, host.v(t) + (size_t) li*row, row) != 0) lv[j] = 1;
                if (kind[j] == 1 && !p.prerope) continue;   // queued shift not applied yet
                const uint16_t * hk = (const uint16_t *) (host.k(t) + (size_t) li*row);
                for (size_t e = 0; e < row/2; ++e) ex[e] = f16[hk[e]];
                for (int32_t hh = 0; hh < n_head_kv; ++hh) {
                    double * q = ex.data() + (size_t) hh*head_dim;
                    for (int32_t pl = 0; pl < np; ++pl) {
                        const int32_t i0 = rope_neox ? pl : 2*pl, i1 = rope_neox ? pl + np : 2*pl + 1;
                        if (i0 >= head_dim || i1 >= head_dim) continue;
                        const double c_ = cs[(size_t) j*np + pl], s_ = sn[(size_t) j*np + pl];
                        const double u = q[i0], w = q[i1];
                        q[i0] = u*c_ - w*s_; q[i1] = u*s_ + w*c_;
                    }
                }
                const uint16_t * ck = (const uint16_t *) (kb.data() + ((size_t) li*m + j)*row);
                double d = 0, na = 0, nb = 0;
                for (size_t e = 0; e < row/2; ++e) { const double x = f16[ck[e]], y = ex[e]; d += x*y; na += x*x; nb += y*y; }
                const double co = (na > 0 && nb > 0) ? d / sqrt(na*nb) : 1.0;
                lmin[j] = std::min(lmin[j], co);
                if (co < 0.999) lk[j] = 1;
            }
            std::lock_guard<std::mutex> lg(mu);
            for (int32_t j = 0; j < m; ++j) { vb_bad[j] |= lv[j]; kb_bad[j] |= lk[j]; kmin_c[j] = std::min(kmin_c[j], lmin[j]); }
        });
        for (int32_t j = 0; j < m; ++j) {
            if (!kind[j]) continue;
            const bool bad = vb_bad[j] || kb_bad[j];
            if (kind[j] == 1) { live++;  vbad += vb_bad[j]; kbad += kb_bad[j]; if (kmin_c[j] <= 1.5) kmin  = std::min(kmin,  kmin_c[j]); }
            else              { tlive++; tvbad += vb_bad[j]; tkbad += kb_bad[j]; if (kmin_c[j] <= 1.5) tkmin = std::min(tkmin, kmin_c[j]); }
            if (bad) bad_here.push_back(prov[j]);
        }
    }
    n_bchk++;
    n_bchk_live += live; n_bchk_meta += meta; n_bchk_slot += slot; n_bchk_dup += dup;
    n_bchk_v += vbad; n_bchk_k += kbad;
    n_bchk_tr_live += tlive; n_bchk_tr_v += tvbad; n_bchk_tr_k += tkbad;
    bchk_kmin = std::min(bchk_kmin, kmin); bchk_tr_kmin = std::min(bchk_tr_kmin, tkmin);
    for (int32_t t : bad_here) if (bchk_bad_t.size() < 1000000) bchk_bad_t.push_back(t);
    printf("  BCHK %-6s p %d  band live %lld  meta %lld  slot %lld  dup %lld  V %lld  K %lld  kcos_min %.6f  |  "
           "trunk live %lld  V %lld  K %lld  kcos_min %.6f  (%.2fs)\n",
           where, p_now, (long long) live, (long long) meta, (long long) slot, (long long) dup,
           (long long) vbad, (long long) kbad, kmin > 1.5 ? -1.0 : kmin,
           (long long) tlive, (long long) tvbad, (long long) tkbad, tkmin > 1.5 ? -1.0 : tkmin, inf_now() - t0);
    fflush(stdout);
}

// Read the cache's own (position, token) state for every band cell and check it against what the
// policy thinks it placed; then REBUILD band_of from the cache, so residency is derived rather
// than believed.  The byte check covers the cells written by this gather: their K is still RoPEd
// at the token's true position (the shift to the slot is queued, not yet applied), so the cache
// row must equal the host row exactly.
void inf_ctx::verify_band(int32_t p_now) {
    chk_pos.resize(p.S);
    chk_tok.resize(p.S);
    llama_memory_kv_cells_state(mem, seq, band_c0, (size_t) p.S, chk_pos.data(), chk_tok.data());

    int32_t pos_bad = 0, tok_bad = 0, fixed = 0;
    for (int32_t c = 0; c < p.S; ++c) {
        const int32_t want_tok = band_tok[c];
        const int32_t got_tok  = (int32_t) chk_tok[c];
        if (want_tok != got_tok) tok_bad++;

        if (last_rank[c] >= 0) {
            const llama_pos want = p_now - (p.W + 1 + last_rank[c]);
            if (chk_pos[c] != want) pos_bad++;
        }
    }

    // rebuild band_of from the cache
    band_of.clear();
    for (int32_t c = 0; c < p.S; ++c) {
        const int32_t t = (int32_t) chk_tok[c];
        band_tok[c] = (chk_pos[c] < 0 || t == LLAMA_TOKEN_NULL) ? -1 : t;
        if (band_tok[c] >= 0) {
            auto it = band_of.find(band_tok[c]);
            if (it != band_of.end()) fixed++;   // the same token in two cells
            band_of[band_tok[c]] = c;
        }
    }

    // byte check on this gather's writes
    int32_t byte_bad = 0;
    const int32_t nw = (int32_t) wrote_cell.size();
    const int32_t step = nw > 64 ? nw/64 : 1;
    for (int32_t j = 0; j < nw; j += step) {
        const uint32_t c = band_c0 + (uint32_t) wrote_cell[j];
        const int32_t  t = wrote_tok[j];
        llama_memory_kv_cells_read(mem, seq, &c, 1, pin_k, pin_v);
        for (int32_t il = 0; il < n_layer; ++il) {
            if (memcmp(pin_k + (size_t) il*row, host.k(t) + (size_t) il*row, row) != 0 ||
                memcmp(pin_v + (size_t) il*row, host.v(t) + (size_t) il*row, row) != 0) { byte_bad++; break; }
        }
        n_chk_cells++;
    }

    n_chk++;
    n_chk_pos_bad  += pos_bad;
    n_chk_tok_bad  += tok_bad;
    n_chk_byte_bad += byte_bad;
    n_bandof_fixed += fixed;
}

// ---------------------------------------------------------------------------------------------
// the one-off switch from dense attention to the two-tier layout
// ---------------------------------------------------------------------------------------------
// 09-23 pin fix (ported from v1).  Move the pinned prefix to the slots np[0..n_pin].  pinhost = 0:
// the in-place llama_memory_seq_remap() (one more f16 K-shift of the live cells per call; the
// error accumulates, ~8,000 shifts at 1M).  pinhost = N: every N-th move rewrites K from the host
// copy rotated ONCE (prerope: exact rotation to the slot, no shift; else src = true position and
// one queued shift).  V is never roped, so only K is written.  Pins must sit in one contiguous
// cell run for kv_cells_write; otherwise fall back to the remap and count it.
void inf_ctx::pin_move(const llama_pos * np) {
    const int32_t np1 = p.n_pin + 1;
    n_pin_moves++;
    bool host_ok = p.pinhost > 0 && (n_pin_moves % p.pinhost) == 0 && (int32_t) pin_pos.size() >= np1;
    if (host_ok) {
        pin_cell.assign(np1, UINT32_MAX);
        if (!llama_memory_kv_cells_of(mem, seq, pin_pos[0], pin_pos[0] + np1, pin_cell.data())) host_ok = false;
        for (int32_t i = 1; host_ok && i < np1; ++i) if (pin_cell[i] != pin_cell[0] + (uint32_t) i) host_ok = false;
        if (!host_ok) n_pin_host_fallback++;
    }
    if (!host_ok) {
        map_buf.assign(np1, -1);
        const llama_pos lo = pin_pos[0];
        for (int32_t i = 0; i < np1; ++i) map_buf[pin_pos[i] - lo] = np[i];
        llama_memory_seq_remap(mem, seq, lo, lo + np1, map_buf.data(), (size_t) np1);
        for (int32_t i = 0; i < np1; ++i) pin_pos[i] = np[i];
        return;
    }
    src_pos.resize(np1); dst_pos.resize(np1);
    std::vector<llama_token> tokv(np1);
    for (int32_t i = 0; i < np1; ++i) {
        tokv[i] = i; dst_pos[i] = np[i]; src_pos[i] = p.prerope ? np[i] : i;
    }
    inf_parallel(p.nthread, n_layer, [&](int32_t li) {
        for (int32_t i = 0; i < np1; ++i) {
            uint8_t * dk = pin_k + ((size_t) li*np1 + i)*row;
            memcpy(dk, host.k(i) + (size_t) li*row, row);
            if (p.prerope) rope_row_f16(dk, np[i]);
        }
    });
    llama_memory_kv_cells_write(mem, seq, pin_cell[0], (size_t) np1, src_pos.data(), dst_pos.data(),
                                pin_k, nullptr, tokv.data());
    bytes_h2d += (int64_t) np1 * n_layer * (int64_t) row;
    if (in_decode) bytes_h2d_decode += (int64_t) np1 * n_layer * (int64_t) row;
    for (int32_t i = 0; i < np1; ++i) pin_pos[i] = np[i];
    n_pin_host++;
}

// 09-24 T25 (port of v1 pin_drift): cosine of every pinned (pin, layer) K row in the cache against
// the host copy rotated once, in double precision, to the pin's current slot.  Under prerope the
// host holds the pre-RoPE key (rotate by the slot); otherwise the post-RoPE key at its true
// position i (rotate by slot - i).  Call only when no K-shift is pending, i.e. after a decode:
// block_update() runs before each block's llama_decode(), which applies the shift.
bool inf_ctx::pin_host_cos(double & cmin, double & cmean, double & smin, double & dmax, int64_t & nrows) {
    llama_synchronize(lctx);
    const int32_t np1 = p.n_pin + 1;
    cmin = 2.0; cmean = 0.0; smin = 2.0; dmax = 0.0; nrows = 0;
    if ((int32_t) pin_pos.size() < np1 || host.n_tok < np1) return false;
    std::vector<uint32_t> cl(np1);
    for (int32_t i = 0; i < np1; ++i) {
        uint32_t cc = UINT32_MAX;
        if (!llama_memory_kv_cells_of(mem, seq, pin_pos[i], pin_pos[i] + 1, &cc) || cc == UINT32_MAX) return false;
        cl[i] = cc;
    }
    std::vector<uint8_t> kb((size_t) np1 * n_layer * row), vb(kb.size());
    llama_memory_kv_cells_read(mem, seq, cl.data(), (size_t) np1, kb.data(), vb.data());
    const float * f16 = inf_f16tab();
    const int32_t np = (int32_t) theta.size();
    double csum = 0.0;
    std::vector<double> ex((size_t) row / 2);
    for (int32_t i = 0; i < np1; ++i) {
        const double delta = p.prerope ? (double) pin_pos[i] : (double) (pin_pos[i] - i);
        for (int32_t li = 0; li < n_layer; ++li) {
            const uint16_t * hk = (const uint16_t *) (host.k(i) + (size_t) li*row);
            for (size_t e = 0; e < row/2; ++e) ex[e] = f16[hk[e]];
            for (int32_t hh = 0; hh < n_head_kv; ++hh) {
                double * q = ex.data() + (size_t) hh*head_dim;
                for (int32_t pl = 0; pl < np; ++pl) {
                    const int32_t i0 = rope_neox ? pl : 2*pl, i1 = rope_neox ? pl + np : 2*pl + 1;
                    if (i0 >= head_dim || i1 >= head_dim) continue;
                    const double a = (double) theta[pl] * delta, cs = cos(a), sn = sin(a);
                    const double u = q[i0], w = q[i1];
                    q[i0] = u*cs - w*sn; q[i1] = u*sn + w*cs;
                }
            }
            const uint16_t * ck = (const uint16_t *) (kb.data() + ((size_t) li*np1 + i)*row);
            double d = 0, na = 0, nb = 0;
            for (size_t e = 0; e < row/2; ++e) {
                const double x = f16[ck[e]], y = ex[e];
                d += x*y; na += x*x; nb += y*y; dmax = std::max(dmax, fabs(x - y));
            }
            const double cc = (na > 0 && nb > 0) ? d / sqrt(na*nb) : 1.0;
            cmin = std::min(cmin, cc); csum += cc; nrows++;
            if (i == 0) smin = std::min(smin, cc);
        }
    }
    cmean = nrows ? csum / nrows : 0.0;
    return true;
}

void inf_ctx::transition(int32_t p_now) {
    // everything is on the host, so: drop the whole cache, park the band, and re-write the
    // trunk and the pinned prefix at their true positions.
    llama_memory_clear(mem, false);
    band_of.clear();
    std::fill(band_tok.begin(), band_tok.end(), -1);
    ring = 0;

    // the band cells are empty after the clear; the gather that follows this call fills all S of
    // them (select() back-fills to exactly S), so nothing is left for find_slot() to take.

    // trunk: tokens [p_now - W, p_now) at their true positions, cells [0, W)
    auto write_run = [&](int32_t t0, int32_t nt, uint32_t c0, bool pinned, int32_t pin_first) {
        int32_t done = 0;
        while (done < nt) {
            const int32_t m = std::min(nt - done, INF_WRITE_MAX);
            src_pos.resize(m); dst_pos.resize(m);
            std::vector<llama_token> tokv(m);
            for (int32_t j = 0; j < m; ++j) {
                const int32_t t = t0 + done + j;
                tokv[j]    = t;
                dst_pos[j] = pinned ? (p_now - (p.W + p.S + 1 + (pin_first - t))) : t;
                // 09-23: under prerope the host holds the PRE-RoPE key, so it is rotated here to
                // the slot it takes and claimed there (no queued shift).  Before this fix the
                // trunk (and the pins) were written back UN-rotated at every two-tier transition.
                src_pos[j] = p.prerope ? dst_pos[j] : t;
                if (pinned) pin_pos[t] = dst_pos[j];
            }
            inf_parallel(p.nthread, n_layer, [&](int32_t li) {
                uint8_t * dk = pin_k + (size_t) li*m*row;
                uint8_t * dv = pin_v + (size_t) li*m*row;
                for (int32_t j = 0; j < m; ++j) {
                    memcpy(dk + (size_t) j*row, host.k(t0 + done + j) + (size_t) li*row, row);
                    memcpy(dv + (size_t) j*row, host.v(t0 + done + j) + (size_t) li*row, row);
                    if (p.prerope) rope_row_f16(dk + (size_t) j*row, dst_pos[j]);
                }
            });
            llama_memory_kv_cells_write(mem, seq, c0 + (uint32_t) done, (size_t) m,
                                        src_pos.data(), dst_pos.data(), pin_k, pin_v, tokv.data());
            bytes_h2d += 2ll * m * n_layer * (int64_t) row;
            done += m;
        }
    };

    write_run(p_now - p.W, p.W, 0, false, 0);
    write_run(0, p.n_pin + 1, (uint32_t) p.W, true, p.n_pin);

    two_tier = true;
    p_place  = p_now;
    p_shift  = p_now;
    if (p.debug) fprintf(stderr, "infinite: two-tier layout on at p = %d (trunk %d + pinned %d + band %d)\n",
                         p_now, p.W, p.n_pin + 1, p.S);
}

// ---------------------------------------------------------------------------------------------

// Window-only prefill: keep the pinned prefix at the slots immediately below the trunk and the
// last W tokens at their true positions, and drop everything else.  No band exists, so a block
// costs one seq_rm and one seq_remap of 129 cells -- no selection, no sentinel scan, no gather.
// Every token's K/V still reaches the host through capture(), and the units, the word counts and
// both indexes are still built, so the band can be assembled the moment it is wanted.
void inf_ctx::window_update(int32_t p_now) {
    const int32_t rm_hi = p_now - p.W;
    if (rm_hi <= p.n_pin + 1) { p_place = p_now; return; }        // still inside the window

    // Evict first, then move the pins up into the space that just freed.  The pins sit strictly
    // below the previous trunk edge, so the eviction range never contains them, and every slot
    // they move to was either just evicted or is a pin's own old slot.
    const int32_t rm_lo = std::max(p.n_pin + 1, win_edge);
    if (rm_hi > rm_lo) llama_memory_seq_rm(mem, seq, rm_lo, rm_hi);
    win_edge = rm_hi;

    {   // 09-23: through pin_move() (pinhost = 0 is the same remap as before)
        std::vector<llama_pos> np(p.n_pin + 1);
        for (int32_t i = 0; i <= p.n_pin; ++i) np[i] = p_now - (p.W + 1 + (p.n_pin - i));
        pin_move(np.data());
    }

    p_place = p_now;
}

void inf_ctx::block_update(int32_t p_now, const llama_token * up, int32_t n_up) {
    flush_capture();   // 09-24 PERF capblk: the transition and the gather read the host store
    if (p.syncall) llama_synchronize(lctx);
    n_blocks++;
    freq.refresh();
    rare_rank = freq.rank_for_tail(p.rare_tail);

    const int32_t P_on = p.W + p.S + p.n_pin + 2;

    // 09-24 FLOOR (measurement arm): the trunk and the pins only, in prefill and in decode alike.
    // The same window roll as window-only prefill; the two-tier layout is never built.
    if (p.floor_only) {
        if (!p.t12fix || p_now >= P_on) window_update(p_now);
        return;
    }

    // Window-only prefill: no band until the hand-off asks for one.  pf_depth opens a region at
    // the END of the prefill in which the full machinery runs, so the last N tokens are encoded
    // with the band in place; pf_blk sets how often the gather fires inside it.
    const int32_t blk   = p.pf_blk > p.B ? p.pf_blk : p.B;
    const bool    pf_on = p_now >= pf_from &&
                          (p.pf_blk < 0 ? units.size() > pf_units
                                        : ((p_now - pf_from) % blk) == 0);
    if (pf_on && p.pf_blk < 0) pf_units = units.size();
    // Before the region opens the layout is single-tier, so the window roll is the whole job.
    // Once two_tier is on, window_update() would tear the layout down, so never take it again.
    // 09-24 T12 fix: below P_on the whole context still fits the cache, so it stays dense; the
    // first window_update() at p >= P_on evicts [n_pin+1, p - W) at once and moves the pins from
    // their true positions.  Before this, every context of ~8.3K-24.7K was read through an 8K
    // window plus the pins with no band (T12 under CORE_PH: 32768/32768 logits differ).
    if (p.store_budget > 0) store_evict(p_now);   // 09-25 v2bld32 store budget
    if (!p.pf_gather && !in_decode && !two_tier && !pf_on) {
        // 09-25 v2bld32 feedwin: inside a batched run the roll happened at the batch start (eval_fast)
        if ((!p.t12fix || p_now >= P_on) && !fast_batch_active) window_update(p_now);
        return;
    }

    bool just_transitioned = false;
    if (!two_tier) {
        if (p_now < P_on) return;      // still dense: the whole context is inside the cache
        transition(p_now);
        just_transitioned = true;      // this block IS the hand-off gather; it must not be skipped
    } else {
        // 1. re-place the pinned prefix (contiguous run of positions)
        {   // 09-23: through pin_move() (pinhost = 0 is the same remap as before)
            std::vector<llama_pos> npv(p.n_pin + 1);
            for (int32_t i = 0; i <= p.n_pin; ++i) npv[i] = p_now - (p.W + p.S + 1 + (p.n_pin - i));
            pin_move(npv.data());
        }

        // 2. evict whatever left the trunk since the LAST placement (its K/V is already on the
        //    host).  The range starts at the previous placement's trunk edge, not at p - B - W:
        //    the band's positions all sit below p_place - W, so this can never touch a band cell,
        //    and a second block_update() at the same p is a no-op instead of eating the band.
        const int32_t rm_hi = p_now - p.W;
        const int32_t rm_lo = std::max(p.n_pin + 1, p_place - p.W);
        if (rm_hi > rm_lo) llama_memory_seq_rm(mem, seq, rm_lo, rm_hi);
    }

    // 3. gather.  With the block gather OFF, the band is written only by the hand-off gather and
    //    by rare-word events; nothing moves in between.  All this block boundary still does is
    //    let the trunk slide (already done in step 2) and, once the band's drift would push its
    //    far edge past the trained range, re-place it.  That is roughly one placement per
    //    shift_every tokens instead of one per 128.
    // No gather at this boundary: slide the trunk, and re-place only when the band's drift
    // demands it.  pre_hold holds this off across the request (and the held-back tail), whose
    // gather already ran before any of it was encoded -- but the trunk must still roll, which is
    // why this is a branch inside block_update and not a skip of the whole call.
    if ((!p.evt_block || pre_hold > 0) && two_tier && !just_transitioned && !pf_on) {
        if (p_now - p_shift >= p.shift_every) {
            rank_only(p_now);
            p_shift = p_now;
            p_place = p_now;
        }
        return;
    }
    const double tg = inf_now();
    select(p_now, up, n_up);
    t_gather += inf_now() - tg;
    place_band(p_now);
    p_place = p_now;
}

// selftest: drop the whole cache and rewrite every token from the HOST store at its true
// position (optionally shifted by a constant, which RoPE-relative attention must not notice).
// If this changes the model's continuation, the capture or the write-back is wrong and nothing
// else in the design can be trusted.
void inf_ctx::rewrite_all(int32_t shift, bool do_clear, bool do_data) {
    flush_capture();   // 09-24 PERF capblk
    if (do_clear) llama_memory_clear(mem, false);
    band_of.clear();
    std::fill(band_tok.begin(), band_tok.end(), -1);
    two_tier = false;

    int32_t done = 0;
    while (done < n_past) {
        const int32_t m = std::min(n_past - done, INF_WRITE_MAX);
        for (int32_t li = 0; li < n_layer; ++li) {
            uint8_t * dk = pin_k + (size_t) li*m*row;
            uint8_t * dv = pin_v + (size_t) li*m*row;
            for (int32_t j = 0; j < m; ++j) {
                memcpy(dk + (size_t) j*row, host.k(done + j) + (size_t) li*row, row);
                memcpy(dv + (size_t) j*row, host.v(done + j) + (size_t) li*row, row);
            }
        }
        src_pos.resize(m); dst_pos.resize(m);
        for (int32_t j = 0; j < m; ++j) { src_pos[j] = done + j; dst_pos[j] = done + j - shift; }
        llama_memory_kv_cells_write(mem, seq, (uint32_t) done, (size_t) m,
                                    src_pos.data(), dst_pos.data(),
                                    do_data ? pin_k : nullptr, do_data ? pin_v : nullptr, src_pos.data());
        done += m;
    }
    n_past -= shift;
}

// Where did a token range end up at the last gather?  The decision tree the needle diagnosis
// needs: was it far at all -> did the selection ask for it -> did it survive into the band ->
// at what slot rank; and if the selection did not ask for it, did the lexical trigger hit its
// unit and get cut by the budget, or did it never hit.

void inf_ctx::qk_mass(int32_t t0, int32_t t1, int32_t d0, int32_t d1, const char * tag, int32_t qtok) {
    flush_capture();   // 09-24 PERF capblk
    if (!two_tier || band_tok.empty()) return;
    llama_synchronize(lctx);

    // the live band, and which of its cells hold the range we are asking about
    std::vector<uint32_t> cl;
    std::vector<uint8_t>  istgt;
    cl.reserve(p.S); istgt.reserve(p.S);
    for (int32_t c = 0; c < p.S; ++c) {
        if (band_tok[c] < 0) continue;
        cl.push_back(band_c0 + (uint32_t) c);
        istgt.push_back(band_tok[c] >= t0 && band_tok[c] < t1 ? 1 : 0);
    }
    const size_t nc = cl.size();
    if (!nc) return;
    int32_t ntgt = 0; for (uint8_t b : istgt) ntgt += b;

    std::vector<uint8_t> kb((size_t) nc * row * n_layer), vb((size_t) nc * row * n_layer);
    llama_memory_kv_cells_read(mem, seq, cl.data(), nc, kb.data(), vb.data());

    // VALUES ARE NEVER ROPED.  A band cell's V must therefore be BYTE-IDENTICAL to host.v(t); any
    // difference at all is a bug rather than a rotation.  K is rotated to its slot, so it is
    // reported as a cosine against the host copy instead of a byte comparison.
    {
        size_t vdiff = 0, vtot = 0; double kcos_min = 2.0;
        const float * f16 = inf_f16tab();
        for (size_t j = 0; j < nc; ++j) {
            if (!istgt[j]) continue;
            const int32_t t = band_tok[(cl[j] - band_c0)];
            for (int32_t li = 0; li < n_layer; ++li) {
                const uint8_t * cv = vb.data() + ((size_t) li*nc + j)*row;
                const uint8_t * hv = host.v(t) + (size_t) li*row;
                for (size_t b = 0; b < row; ++b) { vtot++; if (cv[b] != hv[b]) vdiff++; }
                const uint16_t * ck = (const uint16_t *)(kb.data() + ((size_t) li*nc + j)*row);
                const uint16_t * hk = (const uint16_t *)(host.k(t) + (size_t) li*row);
                double d = 0, na = 0, nb2 = 0;
                for (size_t e = 0; e < row/2; ++e) { const double a = f16[ck[e]], b2 = f16[hk[e]];
                    d += a*b2; na += a*a; nb2 += b2*b2; }
                const double c = (na > 0 && nb2 > 0) ? d/sqrt(na*nb2) : 1.0;
                if (c < kcos_min) kcos_min = c;
            }
        }
        if (vtot) printf("  VCHECK %s  V bytes differing from host %zu of %zu  |  K vs host (rotated) min cosine %.4f\n",
                         tag, vdiff, vtot, kcos_min);
        // Per layer, because the needle-mass peak moves from L16 at 128K to L12 at 1M with the
        // total mass intact - which is what a LAYER-DEPENDENT fault in the band path would look
        // like.  If V is byte-perfect at every layer and K's cosine is flat, the band path is not
        // where the layer shift comes from.
        if (vtot) {
            printf("  VCHECKL %s  per-layer  V diff / K cosine:", tag);
            for (int32_t li = 0; li < n_layer; ++li) {
                size_t ld = 0, lt = 0; double lc = 2.0;
                for (size_t j = 0; j < nc; ++j) {
                    if (!istgt[j]) continue;
                    const int32_t t = band_tok[(cl[j] - band_c0)];
                    const uint8_t * cv = vb.data() + ((size_t) li*nc + j)*row;
                    const uint8_t * hv = host.v(t) + (size_t) li*row;
                    for (size_t b3 = 0; b3 < row; ++b3) { lt++; if (cv[b3] != hv[b3]) ld++; }
                    const uint16_t * ck = (const uint16_t *)(kb.data() + ((size_t) li*nc + j)*row);
                    const uint16_t * hk = (const uint16_t *)(host.k(t) + (size_t) li*row);
                    double d = 0, na = 0, nb3 = 0;
                    for (size_t e = 0; e < row/2; ++e) { const double a3 = f16[ck[e]], b4 = f16[hk[e]];
                        d += a3*b4; na += a3*a3; nb3 += b4*b4; }
                    const double cc = (na > 0 && nb3 > 0) ? d/sqrt(na*nb3) : 1.0;
                    if (cc < lc) lc = cc;
                }
                if (lt) printf(" L%d=%zu/%.4f", li, ld, lc);
            }
            printf("\n");
        }
    }

    const int32_t qrow = n_head * head_dim;
    bufQ.resize((size_t) qrow);
    const float * f16t = inf_f16tab();

    double best_layer = 0.0; int32_t best_il = -1;
    double sum_layer = 0.0, sum_dig = 0.0;
    printf("  QKMASS %s tgt [%d,%d) cells %d/%zu  per-layer share of far mass:", tag, t0, t1, ntgt, nc);
    for (int32_t il = 0; il < n_layer; ++il) {
        if (llama_get_qkv(lctx, il, 0, qtok, 1, bufQ.data()) < 0) { printf(" (no Q)"); break; }
        double lmass = 0.0, lmassd = 0.0;
        for (int32_t h = 0; h < n_head; ++h) {
            const float * q = bufQ.data() + (size_t) h * head_dim;
            const int32_t kh = h / gqa;
            // logits over the live band for this query head
            double mx = -1e300;
            std::vector<double> z(nc);
            for (size_t j = 0; j < nc; ++j) {
                const uint16_t * k = (const uint16_t *) (kb.data() + ((size_t) il*nc + j)*row)
                                   + (size_t) kh * head_dim;
                double d = 0.0;
                for (int32_t e = 0; e < head_dim; ++e) d += (double) q[e] * f16t[k[e]];
                z[j] = d * kq_scale;
                if (z[j] > mx) mx = z[j];
            }
            double den = 0.0, num = 0.0, numd = 0.0;
            for (size_t j = 0; j < nc; ++j) {
                const double w = exp(z[j] - mx); den += w;
                if (istgt[j]) {
                    num += w;
                    const int32_t t = band_tok[(cl[j] - band_c0)];
                    if (d1 > d0 && t >= d0 && t < d1) numd += w;   // the DIGIT cells
                }
            }
            lmass  += den > 0 ? num/den  : 0.0;
            lmassd += den > 0 ? numd/den : 0.0;
        }
        lmass /= n_head; lmassd /= n_head;
        sum_layer += lmass; sum_dig += lmassd;
        if (lmass > best_layer) { best_layer = lmass; best_il = il; }
        if ((il % 4) == 0) printf(" L%d=%.4f", il, lmass);
    }
    printf("  | mean %.4f  max %.4f at L%d  |  of which on the DIGIT cells: mean %.4f (%.0f%% of the unit's mass)\n",
           sum_layer/n_layer, best_layer, best_il, sum_dig/n_layer,
           sum_layer > 0 ? 100.0*sum_dig/sum_layer : 0.0);
    fflush(stdout);
}

inf_probe inf_ctx::probe(int32_t t0, int32_t t1) const {
    inf_probe r;
    r.n_tok   = t1 - t0;
    r.p_place = p_place;
    r.far_hi  = p_place - p.W;

    for (int32_t t = t0; t < t1; ++t) {
        if (t > p.n_pin && t < r.far_hi) r.n_far++;
        if (t >= 0 && t < (int32_t) smark.size() && smark[t] == stamp_cur) {
            r.n_sel++;
            if (t < (int32_t) votes.size()) {
                if (votes[t] >= (1 << 20)) r.n_forced++;
                else if (votes[t] > r.votes_max) r.votes_max = votes[t];
            }
        }
        auto it = band_of.find(t);
        if (it != band_of.end()) {
            r.n_band++;
            const int32_t rk = (int32_t) last_rank.size() > it->second ? last_rank[it->second] : -1;
            if (rk >= 0) {
                if (r.rank_min < 0 || rk < r.rank_min) r.rank_min = rk;
                if (rk > r.rank_max) r.rank_max = rk;
            }
        }
    }
    if (r.rank_min >= 0) { r.dist_min = p.W + 1 + r.rank_min; r.dist_max = p.W + 1 + r.rank_max; }

    // the unit holding the first token of the range
    int32_t lo = 0, hiu = (int32_t) units.size();
    while (lo < hiu) { const int32_t m = (lo + hiu)/2; if (units[m].t1 <= t0) lo = m + 1; else hiu = m; }
    if (lo < (int32_t) units.size() && units[lo].t0 <= t0) {
        r.unit = lo; r.unit_t0 = units[lo].t0; r.unit_t1 = units[lo].t1;
        if (lo < (int32_t) u_stamp.size() && u_stamp[lo] == stamp_cur) r.unit_score = u_score[lo];
    }
    // the needle's own unit, and the 96 far tokens either side of it: if the needle's tokens are
    // absent but these are present, the content can reach the answer through THEIR keys and values,
    // which is the only mechanism left once the trunk is excluded.
    if (r.unit >= 0) {
        for (int32_t t = r.unit_t0; t < r.unit_t1; ++t) {
            if (t <= p.n_pin || t >= r.far_hi) continue;
            r.unit_n++;
            if (band_of.find(t) != band_of.end()) r.unit_band++;
        }
    }
    {
        const int32_t lo = t0 - 96 > 0 ? t0 - 96 : 0;
        const int32_t hi = t1 + 96;
        for (int32_t t = lo; t < hi; ++t) {
            if (t >= t0 && t < t1) continue;          // the needle itself is counted separately
            if (t <= p.n_pin || t >= r.far_hi) continue;
            r.ctx_n++;
            if (band_of.find(t) != band_of.end()) r.ctx_band++;
        }
    }
    for (int32_t t = t0; t < t1; ++t) if (t >= r.far_hi) r.trunk_n++;

    r.unit_hits  = (int32_t) last_hit.size();
    r.unit_taken = last_hit_taken;
    for (size_t i = 0; i < last_hit.size(); ++i) if (last_hit[i] == r.unit) { r.unit_rank = (int32_t) i; break; }

    for (int32_t c = 0; c < p.S; ++c) if (band_tok[c] >= 0 && (r.band_oldest < 0 || band_tok[c] < r.band_oldest)) r.band_oldest = band_tok[c];

    // splice: invert rank -> token over the whole band, then read off the needle's neighbourhood
    if (r.rank_min >= 0) {
        std::vector<int32_t> by_rank(p.S, -1);
        for (int32_t c = 0; c < p.S; ++c) {
            if (band_tok[c] < 0) continue;
            const int32_t rk = ((int32_t) last_rank.size() > c) ? last_rank[c] : -1;
            if (rk >= 0 && rk < p.S) by_rank[rk] = band_tok[c];
        }
        const int32_t u0 = r.unit_t0, u1 = r.unit_t1;
        const int32_t nb[2] = { r.rank_min - 1, r.rank_max + 1 };
        for (int32_t i = 0; i < 2; ++i) {
            const int32_t rk = nb[i];
            if (rk < 0 || rk >= p.S) continue;
            const int32_t t = by_rank[rk];
            if (t < 0) continue;
            r.adj_n++;
            if (t >= u0 && t < u1) r.adj_same++;
            if (t >= u0 - 96 && t < u1 + 96) r.adj_near++;
        }
        for (int32_t rk = r.rank_min; rk <= r.rank_max && rk < p.S; ++rk) {
            const int32_t t = by_rank[rk];
            r.span_n++;
            if (t >= u0 && t < u1) r.span_own++;
        }
        const int32_t lo2 = (r.rank_min - 1 > 0) ? r.rank_min - 1 : 0;
        const int32_t hi2 = (r.rank_max + 1 < p.S - 1) ? r.rank_max + 1 : p.S - 1;
        for (int32_t rk = lo2; rk < hi2; ++rk) {
            const int32_t a = by_rank[rk], b = by_rank[rk + 1];
            if (a < 0 || b < 0) { r.disc++; continue; }
            if (b - a != 1) r.disc++;
        }
    }

    return r;
}

// byte-level round trip: read m cells out of the cache, write the same bytes back into the same
// cells at the same positions (which must be a no-op), read them again and compare.
void inf_ctx::wtest(int32_t c0, int32_t m) {
    std::vector<uint32_t> cl(m);
    for (int32_t j = 0; j < m; ++j) cl[j] = (uint32_t) (c0 + j);

    const size_t per = (size_t) m * row * n_layer;
    std::vector<uint8_t> kA(per), vA(per), kB(per), vB(per);

    llama_memory_kv_cells_read(mem, seq, cl.data(), (size_t) m, kA.data(), vA.data());

    std::vector<llama_pos> sp(m), dp(m);
    for (int32_t j = 0; j < m; ++j) { sp[j] = c0 + j; dp[j] = c0 + j; }   // dense phase: cell == pos
    llama_memory_kv_cells_write(mem, seq, (uint32_t) c0, (size_t) m, sp.data(), dp.data(), kA.data(), vA.data(), sp.data());

    llama_memory_kv_cells_read(mem, seq, cl.data(), (size_t) m, kB.data(), vB.data());

    size_t dk = 0, dv = 0, firstk = per, firstv = per;
    for (size_t i = 0; i < per; ++i) {
        if (kA[i] != kB[i]) { dk++; if (firstk == per) firstk = i; }
        if (vA[i] != vB[i]) { dv++; if (firstv == per) firstv = i; }
    }
    printf("WTEST\tcells=[%d,%d)\tbytes=%zu\tK_diff=%zu (first at %zu)\tV_diff=%zu (first at %zu)\n",
           c0, c0 + m, per, dk, firstk, dv, firstv);

    // and: does the HOST store hold the same bytes the cache holds for those tokens?
    // (kv_cells_of maps position -> cell, and in the dense phase cell == position == token.)
    size_t hk = 0, hv = 0, firsth = per;
    for (int32_t li = 0; li < n_layer; ++li) {
        for (int32_t j = 0; j < m; ++j) {
            const uint8_t * ck = kA.data() + ((size_t) li*m + j)*row;
            const uint8_t * cv = vA.data() + ((size_t) li*m + j)*row;
            const uint8_t * hkk = host.k(c0 + j) + (size_t) li*row;
            const uint8_t * hvv = host.v(c0 + j) + (size_t) li*row;
            for (size_t b = 0; b < row; ++b) {
                if (ck[b] != hkk[b]) { hk++; if (firsth == per) firsth = ((size_t) li*m + j)*row + b; }
                if (cv[b] != hvv[b]) { hv++; }
            }
        }
    }
    printf("WTEST\thost-vs-cache\tK_diff=%zu\tV_diff=%zu\t(first K diff at %zu of %zu)\n", hk, hv, firsth, per);
    fflush(stdout);
}

// The hand-off, run BEFORE the request's forward pass.  `up` is the request's tokens: they are
// not in the stream yet, and block_update already accepts an upcoming-token query, which is the
// same channel the per-block path uses.  Nothing is re-encoded; the history prefill is unchanged.
void inf_ctx::pre_handoff(const llama_token * up, int32_t n_up) {
    if (!p.prehand) return;
    if (p.floor_only) return;       // 09-24 FLOOR: no hand-off gather, no band
    if (p.tailq < 0) p.tailq = std::min(4096, p.W / 2);   // the derived default
    evt_seed_done = false;
    turn_t0 = n_past;               // 09-24 evreenc: the request starts here
    reenc_turn_tok = 0;             // 09-24 evr_budget: a new turn
    if (n_past >= p.W + p.S + p.n_pin + 2 || two_tier) {
        // THIS IS THE HAND-OFF GATHER, even though the request has not been fed yet.  Under
        // window-only prefill block_update() returns early while in_decode is false - it is the
        // flag that says "a band is wanted now" - so without this the pre-encode hand-off silently
        // did nothing: no transition, no gather, an empty band, and regather() skipped afterwards
        // because pre_done had been set.  Caught by band 0 / sel 0 in the needle PROBE.
        const bool was_decode = in_decode;
        in_decode = true;
        // 09-24 reqfetch=0: the request's words stay out of the query (events-only measurement)
        block_update(n_past, p.reqfetch ? up : nullptr, p.reqfetch ? n_up : 0);
        in_decode = was_decode;
        pre_done = true;
        pre_hold = n_up;            // the request must not trigger another gather as it is fed
        n_prehand++;
    }
}

// Park the band: the cells stay allocated and the host copy is untouched, but they leave the
// sequence, so attention no longer sees them.  The positions the band occupies are exactly
// p_place - (W + 1 + rank) for rank in [0, S), which is strictly below the trunk edge and strictly
// above the pins, so the range can be removed without touching either.
void inf_ctx::park_band() {
    if (!two_tier) return;
    int32_t live = 0;
    for (int32_t c = 0; c < p.S; ++c) if (band_tok[c] >= 0) live++;
    if (!live) return;
    const int32_t hi = p_place - (p.W + 1) + 1;
    const int32_t lo = p_place - (p.W + 1 + (p.S - 1));
    if (hi > lo) llama_memory_seq_rm(mem, seq, std::max(lo, p.n_pin + 1), hi);
    for (int32_t c = 0; c < p.S; ++c) { band_tok[c] = -1; last_rank[c] = -1; }
    n_parked += live;
    printf("  PARK  %d band cells released after the request was encoded (pos [%d,%d))\n",
           live, std::max(lo, p.n_pin + 1), hi);
    fflush(stdout);
}

void inf_ctx::regather() {
    if (p.floor_only) return;       // 09-24 FLOOR: no transition, no band at the prompt -> decode hand-off
    if (p.prehand && pre_done) {
        pre_done = false;
        if (p.evictafter) park_band();      // the answer must live on the residuals alone
        return;
    }
    turn_t0 = n_past;               // 09-24 evreenc: no pre-encode hand-off -> the generation start
    reenc_turn_tok = 0;             // 09-24 evr_budget: a new turn
    evt_seed_done = false;          // one event-time seed gather per generation
    if (!p.pf_gather) {
        // window-only prefill: THIS is the first gather.  Build the two-tier layout from the host
        // and select against the question block, which is the last B tokens of the prompt.
        if (n_past >= p.W + p.S + p.n_pin + 2) block_update(n_past, nullptr, 0);
        return;
    }
    if (!two_tier) return;
    // eval() already gathered at n_past if it ended on a block boundary; gathering a second
    // time at the same position re-runs the selection and advances the ring head over cells
    // the first gather installed, which the prompt logits were computed against.
    if (!p.regather && n_past % p.B == 0) return;
    block_update(n_past, nullptr, 0);
}

// 09-25 v2bld32: usage at exit -- one admission per unit touched by this admission (the unit index of the token)
void inf_ctx::note_use_tokens(const std::vector<int32_t> & ts) {
    if (p.store_budget <= 0 || ts.empty() || units.empty()) return;   // inert without a budget
    if (u_use.size() < units.size()) u_use.resize(units.size() + 4096, 0);
    int32_t last_u = -1;
    std::vector<int32_t> seen;
    for (int32_t t : ts) {
        int32_t lo = 0, hi2 = (int32_t) units.size();
        while (lo < hi2) { const int32_t md = (lo + hi2)/2; if (units[md].t1 <= t) lo = md + 1; else hi2 = md; }
        if (lo >= (int32_t) units.size() || units[lo].t0 > t || lo == last_u) continue;
        last_u = lo;
        if (std::find(seen.begin(), seen.end(), lo) != seen.end()) continue;
        seen.push_back(lo);
        u_use[lo]++;
    }
}

// 09-25 v2bld32 store budget (Mark's question).  When the host store holds more than store_budget tokens, evict whole
// FAR units (closed below the trunk edge, not pinned, no token resident in the band) until it holds 90% of the budget:
// never-used units first, then by fewest admissions; oldest first within a score.  An evicted unit's tokens are marked
// (every read path skips them: select's push_forced and recency fill, the event path) and their host K/V pages are
// released with madvise(MADV_DONTNEED).  One log line per round.
void inf_ctx::store_evict(int32_t p_now) {
    const int64_t live = host.n_tok - n_evict_tok;
    if (p.store_budget <= 0 || live <= p.store_budget) return;
    const int32_t far_hi = p_now - p.W;
    if (u_use.size() < units.size()) u_use.resize(units.size() + 4096, 0);
    if (u_evicted.size() < units.size()) u_evicted.resize(units.size() + 4096, 0);
    std::vector<int32_t> cand;
    for (int32_t u = 0; u < (int32_t) units.size(); ++u) {
        if (u_evicted[u] || units[u].t1 > far_hi || units[u].t0 <= p.n_pin) continue;
        bool res = false;
        for (int32_t t = units[u].t0; t < units[u].t1 && !res; ++t) if (band_of.count(t)) res = true;
        if (!res) cand.push_back(u);
    }
    std::sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) {
        if (u_use[a] != u_use[b]) return u_use[a] < u_use[b];
        return units[a].t0 < units[b].t0; });
    const int64_t target = (int64_t) (0.9 * (double) p.store_budget);
    int64_t cur = live, ev_u = 0, ev_t = 0, ev_unused = 0;
    if ((int64_t) tok_evicted.size() < host.n_tok) tok_evicted.resize((size_t) host.n_tok + 65536, 0);
    const long pg = sysconf(_SC_PAGESIZE);
    auto release = [&](uint8_t * a, size_t n) {
        uintptr_t s0 = ((uintptr_t) a + pg - 1) & ~(uintptr_t) (pg - 1);
        uintptr_t s1 = ((uintptr_t) a + n) & ~(uintptr_t) (pg - 1);
        if (s1 > s0) madvise((void *) s0, s1 - s0, MADV_DONTNEED);
    };
    for (int32_t u : cand) {
        if (cur <= target) break;
        u_evicted[u] = 1;
        ev_u++; if (u_use[u] == 0) ev_unused++;
        for (int32_t t = units[u].t0; t < units[u].t1; ++t) {
            if (tok_evicted[t]) continue;   // overlapping units (fixed chunks) share tokens
            tok_evicted[t] = 1; ev_t++; cur--;
            release(host.k(t), host.tok_bytes);
            release(host.v(t), host.tok_bytes);
        }
    }
    n_evict_rounds++; n_evict_units += ev_u; n_evict_tok += ev_t; n_evict_units_unused += ev_unused;
    printf("  STORE evict round %lld at p %d: live %lld > budget %lld -> evicted %lld units (%lld never used), %lld tokens; live now %lld\n",
           (long long) n_evict_rounds, p_now, (long long) live, (long long) p.store_budget, (long long) ev_u, (long long) ev_unused,
           (long long) ev_t, (long long) cur);
    fflush(stdout);
}

// 09-24 v2bld24-perf: the per-token bookkeeping of one fed run [n_past, n_past + take), moved
// verbatim out of eval() so that the batched feeding path runs exactly the same code.
void inf_ctx::post_stream(int32_t take) {
    // units + inverted index over the tokens just seen
    for (int32_t i = 0; i < take; ++i) {
        const int32_t t = n_past + i;
        const llama_token id = toks[t];

        // feed the word assembler first: a piece that starts a new word closes the previous
        // one, and closing a RARE word is what fires an event (prefill and decode alike).
        // Gated on p.evt so that every measurement taken without the event path is running
        // exactly the code it was running before this feature existed.
        if (p.evt || p.trig_word) word_feed(t, id);

        // v2 default: fixed chunks.  A unit closes whenever the next stride boundary is
        // reached, so units are unit_len long and start every unit_stride tokens.
        const bool brk = p.unit_stride > 0
                       ? (t - unit_open + 1) >= p.unit_len
                       : (inf_is_break(piece[id]) || (t - unit_open + 1) >= p.unit_max);
        if (brk) {
            // the word ending at t belongs to the unit that is closing, so close it first.
            // 09-24 chunkcut: a FIXED-chunk boundary falls every unit_stride tokens whatever the
            // text, so this flush cut words in two ("jolly" | "-anchor"): the handle was never
            // indexed (the chunk rows' residency misses, e.g. niah1 Mistral 128K ex 1 and 10).
            // chunkcut=0 leaves the word open; it closes at the next word start and posts to the
            // next chunk, which holds it whole (chunks overlap by unit_len - unit_stride).  Under
            // unit=win<K> units do not decide the band at all, so the word is never cut there.
            const bool cut = p.unit_stride > 0 ? (p.chunk_cut && p.win_k == 0) : true;
            if ((p.evt || p.trig_word) && cut) word_flush(t + 1);

            inf_unit u; u.t0 = unit_open; u.t1 = t + 1;
            units.push_back(u);
            const int32_t ui = (int32_t) units.size() - 1;
            for (int32_t x = u.t0; x < u.t1; ++x) {
                const llama_token xi = toks[x];
                if (tok_stop[xi]) continue;
                auto & v = inv[xi];
                if (v.empty() || v.back() != ui) v.push_back(ui);
            }
            // and the same for the WHOLE WORDS the unit contained
            if (p.evt || p.trig_word) for (uint64_t h : wpend) {
                auto & v = winv[h];
                if (v.empty() || v.back() != ui) v.push_back(ui);
                // keep only the LAST post_cap postings: a term that fires on ten thousand
                // units is not a handle, and at 1M that is what the demand is made of
                if (p.post_cap > 0 && (int32_t) v.size() > p.post_cap)
                    v.erase(v.begin(), v.begin() + (v.size() - p.post_cap));
            }
            // fixed chunks advance by the STRIDE, not to the end of the unit that just
            // closed, so consecutive units overlap by unit_len - unit_stride tokens
            unit_open = p.unit_stride > 0 ? unit_open + p.unit_stride : t + 1;
            // 09-24 wordall: a word wholly inside the overlap also belongs to the next chunk,
            // so it stays pending and is posted again when that chunk closes
            if (p.wordall && p.unit_stride > 0) {
                size_t k2 = 0;
                for (size_t i2 = 0; i2 < wpend.size(); ++i2)
                    if (i2 < wpend_t0.size() && wpend_t0[i2] >= unit_open) { wpend[k2] = wpend[i2]; wpend_t0[k2] = wpend_t0[i2]; k2++; }
                wpend.resize(k2); wpend_t0.resize(k2);
            } else { wpend.clear(); wpend_t0.clear(); }
        }
    }
    n_past += take;
    if (in_decode) n_decode_tok += take;
    close_chunks();
}

// 09-24 v2bld24-perf: append one run to the stream (tokens + the unigram counts), verbatim from eval()
void inf_ctx::feed_stream(const llama_token * tk, int32_t take) {
    for (int32_t i = 0; i < take; ++i) {
        const int32_t t = n_past + i;
        const llama_token id = tk[i];
        if ((int32_t) toks.size() <= t) toks.resize(t + 1);
        toks[t] = id;
        freq.add(id);
    }
}

// 09-24 v2bld24-perf: the batch the context accepts, a whole number of blocks, within the staging
int32_t inf_ctx::feed_cap() const {
    int32_t f = std::min<int32_t>(p.feed, (int32_t) llama_n_batch(lctx));
    f = std::min<int32_t>(f, INF_WRITE_MAX);
    return (f / p.B) * p.B;
}

// 09-24 v2bld24-perf: tokens the batched path may take now (0 = use the 128-token block path).
// Whole blocks, every block start < P_on (below it block_update() is a unigram refresh and never
// touches the cache) and < pf_from; prefill only; the final config's capture path only (no
// pre-RoPE K from the graph, which holds the last ubatch only; no cluster query); t12fix on.
int32_t inf_ctx::fast_take(int32_t rem) const {
    const int32_t fb = feed_cap();
    if (fb <= p.B || rem <= 0) return 0;
    if (in_decode || two_tier || p.prerope || p.use_clusters || !p.t12fix) return 0;
    if (n_past % p.B != 0) return 0;
    const int32_t P_on = p.W + p.S + p.n_pin + 2;
    // 09-25 v2bld32 feedwin: a batch may start at or above P_on (window-only prefill, trunk rolled at the batch start),
    // but never straddles P_on: below it the whole context is resident and the first roll must precede the first
    // above-range block, exactly as in the per-block path
    const bool above0 = n_past >= P_on;
    if (above0 && !(p.feed_win && !p.pf_gather && !p.floor_only)) return 0;
    int32_t t = 0;
    while (t < rem && t < fb) {
        const int32_t bs = n_past + t;
        if (bs >= pf_from) break;
        if (!above0 && bs >= P_on) break;
        t += std::min(p.B, rem - t);
    }
    return t;
}

// 09-24 v2bld24-perf: K/V of [p0, p0 + n) to the host WITHOUT llama_synchronize().  Only called for
// a batch whose compute is known complete (eval_fast() synchronises at the end of every batch)
// while the NEXT batch computes: kv_cells_read() copies on cudaStreamPerThread, which the backend's
// compute stream does not order against, and the next batch writes other cells.  Same bytes, same
// layout, same scatter as capture() on the final config's path (prerope 0, no clusters).
void inf_ctx::capture_nosync(int32_t p0, int32_t n) {
    const double t0 = inf_now();
    host.ensure((int64_t) p0 + n - 1);
    cellbuf.resize(n);
    if (!llama_memory_kv_cells_of(mem, seq, p0, p0 + n, cellbuf.data())) {
        fprintf(stderr, "infinite: capture_nosync: cells for [%d, %d) not all present\n", p0, p0 + n);
        exit(1);
    }
    llama_memory_kv_cells_read(mem, seq, cellbuf.data(), (size_t) n, pin_k, pin_v);
    bytes_d2h += 2ll * n * n_layer * (int64_t) row;
    inf_parallel(p.nthread, n_layer, [&](int32_t il) {
        for (int32_t j = 0; j < n; ++j) {
            memcpy(host.k(p0 + j) + (size_t) il*row, pin_k + ((size_t) il*n + j)*row, row);
            memcpy(host.v(p0 + j) + (size_t) il*row, pin_v + ((size_t) il*n + j)*row, row);
        }
    });
    n_cap_async++;
    t_capture += inf_now() - t0;
}

void inf_ctx::cap_pending_flush() {
    if (cap_pend_n <= 0) return;
    llama_synchronize(lctx);               // already synchronised at the end of the batch; kept as a guard
    capture_nosync(cap_pend_p0, cap_pend_n);
    cap_pend_n = 0;
}

// 09-24 v2bld24-perf: one batch of m tokens (fast_take()) in ONE llama_decode(): (1) launch it --
// llama_decode() returns before the graph has run; (2) while it computes, the previous batch's K/V
// go to the host (capasync); (3) the per-block bookkeeping, in the original order: for each block,
// stream append + unigram counts, block_update() (below P_on: the unigram refresh only; asserted),
// then the per-token words / units / index.  None of it reads the cache.  (4) this batch's K/V: kept
// pending for the next batch (capasync, after a synchronise), or captured now.
bool inf_ctx::eval_fast(llama_batch & batch, const llama_token * tok, int32_t m, bool want_logits) {
    const int32_t p0 = n_past;
    // 09-25 v2bld32 feedwin: above P_on, roll the trunk ONCE, here, before the batch (= after the previous batch): the
    // batch's queries see [p0 - W, p0 + i) plus the pins, i.e. W to W + m - 1 recent tokens.  The previous batch is
    // synchronised (below), so the pin re-place (pinhost) cannot race a computing graph.
    const int32_t P_on_b = p.W + p.S + p.n_pin + 2;
    if (p0 >= P_on_b) { window_update(p0); n_fast_above += m; }
    const int32_t win_edge0 = win_edge;
    const double td = inf_now();
    batch.n_tokens = m;
    for (int32_t i = 0; i < m; ++i) {
        batch.token[i]     = tok[i];
        batch.pos[i]       = p0 + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i]    = 0;
    }
    if (want_logits) batch.logits[m - 1] = 1;
    const int rc = llama_decode(lctx, batch);
    t_decode += inf_now() - td;
    if (rc != 0) { fprintf(stderr, "infinite: llama_decode failed (%d) at pos %d (batched feeding)\n", rc, p0); return false; }
    n_fast_batch++; n_fast_tok += m;

    if (cap_pend_n > 0) { capture_nosync(cap_pend_p0, cap_pend_n); cap_pend_n = 0; }

    int32_t s = 0;
    fast_batch_active = true;
    while (s < m) {
        const int32_t take = std::min(p.B - (n_past % p.B), m - s);
        feed_stream(tok + s, take);
        if (n_past % p.B == 0) block_update(n_past, tok + s, take);
        if (two_tier || win_edge != win_edge0) {
            fprintf(stderr, "infinite: batched feeding: the layout moved at %d (two_tier %d, win_edge %d -> %d)\n",
                    n_past, (int) two_tier, win_edge0, win_edge);
            exit(1);
        }
        if (pre_hold > 0) pre_hold -= take;
        post_stream(take);
        s += take;
    }

    fast_batch_active = false;
    if (p.capasync) { cap_pend_p0 = p0; cap_pend_n = m; llama_synchronize(lctx); }
    else capture(p0, m, true);
    return true;
}

// ---------------------------------------------------------------------------------------------
bool inf_ctx::eval(const llama_token * tok, int32_t n, bool logits_last) {
    llama_batch batch = llama_batch_init(std::max(p.B, feed_cap()), 0, 1);
    bool ok = true;
    int32_t done = 0;

    while (done < n && ok) {
        // 09-24 v2bld24-perf: batched feeding below range (feed=N), when every block start of the run
        // lies below P_on, where block_update() only refreshes the unigram model and moves nothing
        const int32_t fb = fast_take(n - done);
        if (fb > 0) {
            ok = eval_fast(batch, tok + done, fb, logits_last && done + fb == n);
            done += fb;
            continue;
        }
        cap_pending_flush();   // leaving the batched region: the last batch's K/V reaches the host first
        int32_t take = p.B - (n_past % p.B);
        take = std::min(take, n - done);

        // append to the stream (units + inverted index + rarity) (09-24 v2bld24-perf: feed_stream(), verbatim)
        feed_stream(tok + done, take);

        // gather once per block, before the block is decoded.  Under prehand the request's own
        // blocks are held off: the gather for this turn already ran, before the request was fed.
        if (n_past % p.B == 0) block_update(n_past, tok + done, take);
        if (pre_hold > 0) pre_hold -= take;

        const double td = inf_now();
        batch.n_tokens = take;
        for (int32_t i = 0; i < take; ++i) {
            batch.token[i]     = tok[done + i];
            batch.pos[i]       = n_past + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = seq;
            batch.logits[i]    = 0;
        }
        if (logits_last && done + take == n) batch.logits[take - 1] = 1;
        const int rc = llama_decode(lctx, batch);
        t_decode += inf_now() - td;
        if (rc != 0) { fprintf(stderr, "infinite: llama_decode failed (%d) at pos %d\n", rc, n_past); ok = false; break; }

        capture(n_past, take, ((n_past + take) % p.B) == 0 || (done + take == n));

        // units + inverted index over the tokens just seen (09-24 v2bld24-perf: post_stream(), verbatim)
        post_stream(take);
        done   += take;
    }

    cap_pending_flush();   // 09-24 v2bld24-perf: nothing captured asynchronously outlives an eval() call
    llama_batch_free(batch);
    // 09-24 evreenc: a decode-time event fetched far units during this eval -> re-encode the
    // request and the generation so far with them resident; the logits of the last token are
    // then the re-encoded ones
    if (ok && reenc_want) {
        reenc_want = false;
        if (!reencode_tail()) ok = false;
    }
    return ok;
}

// 09-24 evreenc.  The event path makes a fetched unit visible only to LATER queries (the strict
// rule): the request and the tokens generated before the event keep the K/V they were computed
// with, without the unit.  Mistral does not read such a late arrival (conceptval P.12: events-only
// twin 1/81, re-encoded 81/81).  This re-runs the tail at the SAME positions with the band as it
// now is, which is the pre-encode hand-off applied at the event: remove the tail's cells, decode
// the same tokens again in blocks of <= B (one ubatch each, so the graph's pre-RoPE K is the
// whole chunk), and re-capture them so the host store holds the re-encoded K/V.  Nothing else
// changes: the stream (toks, freq, units, index, word assembler) already holds these tokens.
bool inf_ctx::reencode_tail() {
    if (!two_tier || turn_t0 < 0 || turn_t0 >= n_past) return true;
    const double t0 = inf_now();
    flush_capture();                               // the pending decode V reads cells we remove
    int32_t lo = turn_t0;
    const int32_t cap = std::max(1, std::min(p.evr_max, p.W - 2*p.B));   // stays inside the trunk
    if (n_past - lo > cap) { lo = n_past - cap; n_reenc_clamp++; }
    // 09-24 evr_budget: a re-encode that does not fit what is left of the turn's budget is not run;
    // the fetched unit stays in the band under the strict rule (visible to later queries only)
    if (p.evr_budget > 0 && reenc_turn_tok + (n_past - lo) > p.evr_budget) {
        n_reenc_budget_skip++;
        if (p.evt_log)
            printf("  REENC t=%d SKIPPED: tail %d tokens, turn budget %d with %lld used -> strict path\n",
                   n_past, n_past - lo, p.evr_budget, (long long) reenc_turn_tok);
        return true;
    }
    if (!llama_memory_seq_rm(mem, seq, lo, n_past)) {
        fprintf(stderr, "infinite: reencode: seq_rm [%d, %d) refused\n", lo, n_past);
        n_reenc_fail++;
        return false;
    }
    llama_batch batch = llama_batch_init(p.B, 0, 1);
    bool ok = true;
    for (int32_t a = lo; a < n_past && ok; a += p.B) {
        const int32_t m = std::min(p.B, n_past - a);
        batch.n_tokens = m;
        for (int32_t j = 0; j < m; ++j) {
            batch.token[j]     = toks[a + j];
            batch.pos[j]       = a + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = seq;
            batch.logits[j]    = (a + j == n_past - 1) ? 1 : 0;
        }
        const int rc = llama_decode(lctx, batch);
        if (rc != 0) {
            fprintf(stderr, "infinite: reencode: llama_decode failed (%d) at pos %d\n", rc, a);
            n_reenc_fail++; ok = false; break;
        }
        capture(a, m, false);                      // host K/V := the re-encoded rows
    }
    llama_batch_free(batch);
    if (!ok) return false;
    n_reenc++;
    n_reenc_tok += n_past - lo;
    reenc_turn_tok += n_past - lo;
    llama_synchronize(lctx);
    t_reenc += inf_now() - t0;
    if (p.evt_log)
        printf("  REENC t=%d tail [%d,%d) %d tokens re-encoded with the band resident (%.1f ms)\n",
               n_past, lo, n_past, n_past - lo, 1000.0*(inf_now() - t0));
    if (p.bandchk) band_check("reenc", p_place);
    return true;
}

size_t inf_ctx::host_bytes() const {
    size_t b = host.bytes();
    for (int32_t i = 0; i < n_lh; ++i) {
        b += clu_mu[i].size()*sizeof(float) + clu_sig[i].size()*sizeof(uint16_t)
           + clu_logn[i].size()*sizeof(float) + clu_mem[i].size()*sizeof(int32_t)
           + clu_off[i].size()*sizeof(uint32_t);
    }
    b += units.size()*sizeof(inf_unit);
    for (const auto & kv : inv) b += kv.second.size()*sizeof(int32_t) + 32;
    b += toks.size()*sizeof(llama_token);
    return b;
}

int32_t inf_ctx::band_live() const {
    int32_t n = 0;
    for (int32_t c = 0; c < p.S; ++c) if (band_tok[c] >= 0) n++;
    return n;
}

void inf_ctx::report(const char * tag) const {
    const double gb = 1.0/1073741824.0;
    printf("  [infinite %s] n_past %d  blocks %lld  gathers %lld  two_tier %d\n",
           tag, n_past, (long long) n_blocks, (long long) n_gather, (int) two_tier);
    printf("      band: live %d/%d (mean index+cluster occupancy %.3f)  new cells/gather %.0f (%.1f runs)  kept %.0f  "
           "trigger units/gather %.1f  trigger tokens/gather %.0f  cluster tokens/gather %.0f  "
           "trigger tokens WANTED/gather %.0f  reach coverage %.3f\n",
           band_live(), p.S, n_gather ? (double) n_sel_real/n_gather/p.S : 0.0,
           n_gather ? (double) n_new_cells/n_gather : 0.0,
           n_gather ? (double) n_runs/n_gather : 0.0,
           n_gather ? (double) n_kept_cells/n_gather : 0.0,
           n_gather ? (double) n_trig_units/n_gather : 0.0,
           n_gather ? (double) n_trig_tok/n_gather : 0.0,
           n_gather ? (double) n_clu_tok/n_gather : 0.0,
           n_gather ? (double) n_trig_want/n_gather : 0.0,
           n_trig_want ? (double) n_trig_tok/n_trig_want : 1.0);

    if (n_seed_gathers) {
        printf("      query words/gather: request %.1f  trunk-seed %.1f (span %d, discount %.2f)\n",
               (double) n_req_words/n_seed_gathers, (double) n_seed_words/n_seed_gathers,
               p.seed_span, p.seed_disc);
        fflush(stdout);
    }
    if (n_comp_gathers) {
        const double g = (double) n_comp_gathers, S = (double) p.S;
        printf("      band composition: current hits %.3f  kept hits %.3f  recency fill %.3f  "
               "(max re-placements on a live cell %lld)\n",
               n_comp_cur/g/S, n_comp_kept/g/S, n_comp_fill/g/S, (long long) n_shift_max);
        fflush(stdout);
    }
    if (p.evt) {
        printf("      event attempts %lld: %lld no index entry (HASH MISS)  %lld all in trunk  "
               "%lld already resident  %lld fetched\n",
               (long long) n_evt_try, (long long) n_evt_nohash, (long long) n_evt_trunk,
               (long long) n_evt_resident, (long long) n_evt_fetch);
        printf("      event: %lld events (%lld in decode)  %lld fired but found nothing  "
               "%.1f cells/event  %.2f events/1K tokens  "
               "%lld units pulled  %lld runs  %.1f MB h2d from events\n",
               (long long) n_events, (long long) n_events_decode, (long long) n_event_empty,
               n_events ? (double) n_event_cells/n_events : 0.0,
               n_past ? 1000.0*(double) n_events/n_past : 0.0,
               (long long) n_event_units, (long long) n_event_runs,
               2.0*(double) n_event_cells*n_layer*row/1048576.0);
        fflush(stdout);
    }
    printf("      host: far KV %.2f GiB  index+sentinels %.2f GiB  total %.2f GiB  (%.0f B/token)  units %zu  inv keys %zu  chunks %d\n",
           host.bytes()*gb, (host_bytes() - host.bytes())*gb, host_bytes()*gb,
           n_past ? (double) host_bytes()/n_past : 0.0, units.size(), inv.size(), n_chunk_done);
    printf("      cost: capture %.2fs  cluster %.2fs (%lld chunks, %.1f ms/chunk)  score %.2fs (%.2f ms/block)  "
           "select %.2fs  place %.2fs  decode %.2fs\n",
           t_capture, t_cluster, (long long) n_cluster_chunks,
           n_cluster_chunks ? 1000.0*t_cluster/n_cluster_chunks : 0.0,
           t_score, t_score_blocks ? 1000.0*t_score/t_score_blocks : 0.0,
           t_gather, t_place, t_decode);
    if (n_chk) printf("      bandchk: %lld placements checked  pos mismatches %lld  token mismatches %lld  "
                      "byte mismatches %lld of %lld cells sampled  duplicate-token cells %lld\n",
                      (long long) n_chk, (long long) n_chk_pos_bad, (long long) n_chk_tok_bad,
                      (long long) n_chk_byte_bad, (long long) n_chk_cells, (long long) n_bandof_fixed);
    if (p.pinhost > 0 || n_pin_moves > 0)   // 09-24: also printed for pinhost=0 (the non-default path)
        printf("      pins: %lld moves, %lld re-placed from the host (every %d), %lld fell back to the remap\n",
               (long long) n_pin_moves, (long long) n_pin_host, p.pinhost, (long long) n_pin_host_fallback);
    if (p.evreenc || n_reenc > 0)   // 09-24
        printf("      evreenc: %lld re-encodes, %lld tokens (%.1f per re-encode), %.1f ms total (%.2f ms each), "
               "%lld clamped to evr_max %d, %lld failed, %lld skipped by the per-turn budget %d, %lld below evr_surp %.1f\n",
               (long long) n_reenc, (long long) n_reenc_tok, n_reenc ? (double) n_reenc_tok/n_reenc : 0.0,
               1000.0*t_reenc, n_reenc ? 1000.0*t_reenc/n_reenc : 0.0, (long long) n_reenc_clamp, p.evr_max,
               (long long) n_reenc_fail, (long long) n_reenc_budget_skip, p.evr_budget, (long long) n_reenc_surp_skip, p.evr_surp);
    printf("      traffic: H2D %.2f GiB (%.2f MiB/gather)  D2H %.2f GiB  decode H2D %.0f B/token\n",
           bytes_h2d*gb, n_gather ? bytes_h2d/(double) n_gather/1048576.0 : 0.0, bytes_d2h*gb,
           n_decode_tok ? (double) bytes_h2d_decode/n_decode_tok : 0.0);
}

// ---------------------------------------------------------------------------------------------
bool inf_parse_args(inf_params & p, const std::string & k, const std::string & v) {
    if      (k == "prehand")  p.prehand = std::stoi(v);
    else if (k == "evictafter") p.evictafter = std::stoi(v);
    else if (k == "tailq")    p.tailq = std::stoi(v);
    else if (k == "pfdepth")  p.pf_depth = std::stoi(v);
    else if (k == "pfblk")    p.pf_blk = std::stoi(v);
    else if (k == "oracleseed") p.oracle_seed = true;
    else if (k == "prerope")  p.prerope = v.empty() ? true : (std::stoi(v) != 0);   // 09-23: "prerope=0" used to set it TRUE
    else if (k == "pinhost")  p.pinhost = std::stoi(v);        // 09-23 pin fix
    else if (k == "bandchk")  p.bandchk = v.empty() ? true : (std::stoi(v) != 0);   // 09-24
    else if (k == "t12fix")   p.t12fix  = v.empty() ? true : (std::stoi(v) != 0);   // 09-24
    else if (k == "wordall")  p.wordall = v.empty() ? true : (std::stoi(v) != 0);   // 09-24
    else if (k == "capblk")   p.capblk  = v.empty() ? true : (std::stoi(v) != 0);   // 09-24 PERF
    else if (k == "capnok")   p.capnok  = v.empty() ? true : (std::stoi(v) != 0);   // 09-24 PERF
    else if (k == "evreenc")  p.evreenc  = v.empty() ? true : (std::stoi(v) != 0);  // 09-24
    else if (k == "evrmax")   p.evr_max  = std::stoi(v);                             // 09-24
    else if (k == "evrbudget") p.evr_budget = std::stoi(v);                          // 09-24
    else if (k == "evrsurp")  p.evr_surp  = std::stod(v);                            // 09-24
    else if (k == "winkseed") p.win_k_seed = std::stoi(v);                           // 09-24
    else if (k == "reqfetch") p.reqfetch = v.empty() ? true : (std::stoi(v) != 0);  // 09-24
    else if (k == "reqwin")   p.reqwin   = v.empty() ? true : (std::stoi(v) != 0);  // 09-24
    else if (k == "chunkcut") p.chunk_cut = v.empty() ? true : (std::stoi(v) != 0);  // 09-24
    else if (k == "floor")    p.floor_only = v.empty() ? true : (std::stoi(v) != 0);  // 09-24 window-only floor arm
    else if (k == "feed")     p.feed = std::stoi(v);                                   // 09-24 v2bld24-perf
    else if (k == "capasync") p.capasync = v.empty() ? true : (std::stoi(v) != 0);   // 09-24 v2bld24-perf
    else if (k == "kvcap")    p.kvcap = v.empty() ? true : (std::stoi(v) != 0);      // 09-24 v2bld24-perf
    else if (k == "feedwin")  p.feed_win = v.empty() ? true : (std::stoi(v) != 0);   // 09-25 v2bld32
    else if (k == "storebudget" || k == "store_budget") p.store_budget = std::stoll(v);   // 09-25 v2bld32 (tokens; 0 = unbounded)
    else if (k == "seedfar")  p.seed_far = v.empty() ? true : (std::stoi(v) != 0);   // 09-25 v2bld29
    else if (k == "sentfill") p.sent_fill = v.empty() ? 2 : std::stoi(v);             // 09-25 v2bld28 (0 = off; N = neighbours +-1..N)
    else if (k == "sfd1")     p.sf_d1 = std::stod(v);                                  // 09-25 v2bld28
    else if (k == "sfd2")     p.sf_d2 = std::stod(v);                                  // 09-25 v2bld28
    else if (k == "seedsent") { p.seed_sent = v.empty() ? true : (std::stoi(v) != 0); p.seed_sent_user = true; if (p.seed_sent) p.unit_stride = 0; }   // 09-24 per-source units (v2bld25); 09-25 v2bld31: an explicit seedsent wins over unit=win<K>
    else if (k == "W")        p.W = std::stoi(v);
    else if (k == "S")        p.S = std::stoi(v);
    else if (k == "B")        p.B = std::stoi(v);
    else if (k == "pin")      p.n_pin = std::stoi(v);
    else if (k == "roll")     p.n_roll = std::stoi(v);
    else if (k == "unit" && v.rfind("win", 0) == 0) {   // 09-24 unit=win<K>; 09-25 v2bld31 (Mark): request/event windows, the seed stays SENTENCE
        p.win_k = std::stoi(v.substr(3));                // units unless the user sets seedsent (seedsent=0 = seed windows too); unit=win0 = sentence units
        if (p.win_k > 0 && !p.seed_sent_user) { p.seed_sent = true; p.unit_stride = 0; }
        if (p.win_k == 0 && !p.seed_sent_user) p.seed_sent = false;
    }
    else if (k == "unit")     p.unit_max = std::stoi(v);
    else if (k == "winlog")   p.win_log = v.empty() ? true : (std::stoi(v) != 0);
    else if (k == "unitlen")  p.unit_len = std::stoi(v);
    else if (k == "unitstride") p.unit_stride = std::stoi(v);
    else if (k == "sentence") p.unit_stride = 0;   // the v1 sentence rule
    else if (k == "chunk")    p.chunk = std::stoi(v);
    else if (k == "nclu")     p.n_clu = std::stoi(v);
    else if (k == "kmiter")   p.km_iter = std::stoi(v);
    else if (k == "planes")   p.n_planes = std::stoi(v);
    else if (k == "topc")     p.topc = std::stoi(v);
    else if (k == "qtok")     p.qtok = std::stoi(v);
    else if (k == "shortlist")p.shortlist = std::stoi(v);
    else if (k == "rare")     p.rare_tail = std::stod(v);
    else if (k == "maxpost")  p.max_post = std::stoi(v);
    else if (k == "trigfrac") p.trig_frac = std::stod(v);
    else if (k == "nthread")  p.nthread = std::stoi(v);
    else if (k == "noclu")    p.use_clusters = !(v.empty() || std::stoi(v) != 0) ;   // 09-24: value-aware (noclu=0 turns clusters back on)
    else if (k == "notrig")   p.use_triggers = false;
    else if (k == "noroll")   p.use_roll = false;
    else if (k == "syncall")  p.syncall = true;
    else if (k == "verify")   p.verify = true;
    else if (k == "noregather") p.regather = false;
    else if (k == "evt")      p.evt = std::stoi(v) != 0;
    else if (k == "evtnoblock") p.evt_block = !(v.empty() || std::stoi(v) != 0);   // 09-24: value-aware
    else if (k == "evtcap")   p.evt_cap = std::stoi(v);
    else if (k == "evtsurp")  p.evt_surp = std::stod(v);
    else if (k == "evtmax")   p.evt_max = std::stoi(v);
    else if (k == "evtlog")   p.evt_log = true;
    else if (k == "evthalo")  p.evt_halo = std::stoi(v);
    else if (k == "evtseedspan") p.evt_seed_span = std::stoi(v);
    else if (k == "seedspan") p.seed_span = std::stoi(v);
    else if (k == "seeddisc") p.seed_disc = std::stod(v);
    else if (k == "seedwords") p.seed_words = std::stoi(v);
    else if (k == "seedindexed") p.seed_indexed = true;
    else if (k == "nopfgather") p.pf_gather = !(v.empty() || std::stoi(v) != 0);   // 09-24: value-aware
    else if (k == "shiftevery") p.shift_every = std::stoi(v);
    else if (k == "fillkeep")   p.fill_keep = v.empty() || std::stoi(v) != 0;       // 09-24: value-aware
    else if (k == "trigword")   p.trig_word = v.empty() || std::stoi(v) != 0;       // 09-24: value-aware
    else if (k == "trigsurp")   p.trig_surp = std::stod(v);
    else if (k == "postcap")    p.post_cap = std::stoi(v);
    else if (k == "debug")    p.debug = std::stoi(v);
    else return false;
    return true;
}
