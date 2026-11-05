# Unbounded context for stock language models

A patch to [llama.cpp](https://github.com/ggml-org/llama.cpp) that retrofits unbounded context onto an unmodified
rotary decoder. The model is not fine-tuned and no weight changes. Tested on Mistral-7B-v0.3 and Llama-3.1-8B
from 32K tokens to one million tokens on one GPU.

A paper describing the design and its measurements is forthcoming; the citation will be added here.

## What it does

A rotary model reads a far key through its slow rotary planes and by the words the key carries, and tolerates any
position inside its trained range for that key. The order of the tokens inside a span is already written into the
span's keys and values by the time the span leaves the recent window. The design uses those facts to keep the GPU
cache bounded while the history grows without bound:

- **Near window.** The last 8,192 tokens stay in the cache at their true positions, with dense attention.
- **Host store.** Every key and value that leaves the near window is kept exact in host memory, 131 KB per token
  at f16 for a 32-layer model. Nothing is evicted.
- **Word index.** The store is indexed by whole words. A word enters the index when its summed surprisal under a
  running unigram model of the document exceeds 15 nats, and the index keeps the word's last 16 occurrences, each
  with the sentence around it.
- **Far window.** A band of 16,384 cache cells just beyond the near window, inside the trained range, is filled
  for each request before the request is encoded: the sentences holding the request's rare words (and the rare
  words of the recent window that have far occurrences), then whatever the band already held, then the most
  recent far tokens. Fetched keys are re-rotated from their true position to their slot.
- **Events.** When the model generates a rare word during decoding, the word's sentences are fetched into the far
  window and the request and the generated tail are re-encoded once, exactly, over them.
- **Anchors.** The attention sink and the first 128 tokens of the document are kept at slots inside the trained
  range at every length.
- **Window-only prefill.** The document is prefilled through the near window alone; retrieval begins at the
  request.

Every setting is derived from the model card at start-up. The GPU cache holds about 25K cells per layer at any
document length; the host store grows with the document.

Measured on one H200 with the Q8_0 models: planted single needles are read at 99 to 100 of 100 and three-key
discriminations at 88 to 100 of 100 from 32K to one million tokens; RULER scores 70.8 against 69.9 for dense
attention on Llama-3.1-8B at 128K and 67.8 against 75.5 on Mistral-7B at 32K; at 128K the engine prefills at 1.85
times and decodes at 1.65 to 1.68 times the speed of stock llama.cpp, because every query attends over about 25K
cells; a million tokens prefill at 8,222 tokens per second and decode at 149 tokens per second on one H200, with
the store at about 130 GiB of host memory.

## What is in this repository

| path | contents |
| --- | --- |
| `llama.cpp.patch` | the library change: cell-indexed read, write and move of KV cache cells with re-rotation of the key by the position difference, an accessor for the rotary base, and the build hooks (about 600 lines across `include/llama.h`, `src/llama-kv-cache.*`, `src/llama-context.*`, `src/llama-graph.*`, `src/llama-model.cpp`, `src/models/llama.cpp`, `common/CMakeLists.txt`, `tools/CMakeLists.txt`) |
| `files/common/infinite.cpp`, `infinite.h` | the policy module: host store, word index, far window, hand-off, events, re-encode |
| `files/common/infinite-freq.cpp`, `infinite-freq.h` | the online unigram surprisal model that admits words to the index |
| `files/tools/inf-bench/` | `llama-inf-bench`, the command-line tool that drives the policy: prompts, planted needles, throughput, generation |
| `apply.sh` | clones llama.cpp at the pinned commit (or uses your checkout), applies the patch, copies the sources, builds |
| `scripts/get_books.sh` | fetches the six Project Gutenberg books the needle and bench modes use as a haystack |
| `UPSTREAM_COMMIT` | the llama.cpp commit the patch was last built and tested against |

With the policy off, the patched library is the stock engine: the patch adds entry points and changes no existing
code path.

## Requirements

- A CUDA GPU with room for the model and about 25K cache cells per layer. The measurements were made on one
  H200 (141 GB); a 7 to 8B model at Q8_0 with the full design used under 11 GiB of GPU memory at one million
  tokens. Metal and CPU builds compile and run but were not measured.
- Host memory for the store: 131 KB per token at f16. 128K tokens need about 17 GiB, one million tokens about
  130 GiB.
- Models in GGUF. The results used the Q8_0 files of Mistral-7B-v0.3 and Llama-3.1-8B. A Llama-3.1 GGUF must
  carry its rotary frequency table; convert from the official weights with llama.cpp's `convert_hf_to_gguf.py`
  rather than using a third-party file that lacks it.
- Rotary models only. The slot placement re-rotates keys plane by plane and has no meaning for multi-head latent
  attention or linear-bias position schemes.

## Build

```sh
git clone https://github.com/mark-oskin/unbounded-attention.git
cd unbounded-attention
LLAMA_CMAKE_ARGS="-DGGML_CUDA=ON" ./apply.sh         # clones llama.cpp at UPSTREAM_COMMIT, patches, builds
```

To patch a checkout you already have, pass its path: `./apply.sh /path/to/llama.cpp`. The patch is a plain
`git apply` against `include/`, `src/`, `common/CMakeLists.txt` and `tools/CMakeLists.txt`; the new files are
copied in from `files/`. The binary is `llama.cpp/build/bin/llama-inf-bench`.

## Use

`--inf` turns the design on. Without it the tool runs the stock engine with dense attention over the whole
cache, which is the dense reference in every comparison. A bare `--inf` is the configuration of the paper.

### Your own documents

Prompts mode reads JSON lines with a long `history`, a `request`, and optional reference `outputs` for scoring,
and writes one prediction per line:

```sh
printf '%s\n' '{"task":"demo","index":0,"history":"<the document>","request":"<the question>","max_new":64,"outputs":["<expected answer>"],"metric":"all"}' > in.jsonl
build/bin/llama-inf-bench -m model.gguf --mode prompts --prompts in.jsonl --preds out.jsonl --out results.txt --inf
```

The history is prefilled through the near window, the far window is filled for the request's words before the
request is encoded, and generation is greedy for up to `max_new` tokens. `metric` is `all` (the fraction of the
reference strings that appear in the output) or `part` (any of them). Without `--inf` the prompt must fit the
context size set with `-c`.

### Planted needles

Needle mode plants seven-digit codes in a haystack of Gutenberg prose and asks for them by their handle word:

```sh
scripts/get_books.sh data/books
build/bin/llama-inf-bench -m model.gguf --mode needle -T 131072 --nex 100 --seed 100 \
    --tasks niah1,key3r,hop2 --books data/books --inf --tag demo --out results/demo.txt
```

`-T` is the context length in tokens, `--nex` the number of documents, `--seed` the item seed. `niah1` is the
single needle, `key3r` three-key discrimination (three same-format codes, one asked for) and `hop2` the two-hop
question whose bridge sentence sits inside the window. Per-example rows and a summary line per task go to
`--out`; `--resume` continues a partial run.

### Throughput

Bench mode prefills `-T` tokens of the haystack and decodes `-n` tokens, reporting prefill and decode rates,
GPU and host memory, and the re-encodes per generated token:

```sh
build/bin/llama-inf-bench -m model.gguf --mode bench -T 1048576 -n 128 --repeats 2 --bench-q 48 --books data/books --inf
```

`--bench-q N` feeds the last N tokens as a request, so that the hand-off and the re-encodes are exercised as in a
real turn. Compare against stock `llama-bench` from the same build.

### Generation

```sh
build/bin/llama-inf-bench -m model.gguf --mode gen -p "..." -n 64 --inf
```

### Settings

Settings are passed as `--inf-set key=value,key=value`. The ones a user might change:

| key | default | meaning |
| --- | --- | --- |
| `W` | 8192 | the near window, in tokens; a lower bound, since prefill above the window runs in 1,024-token batches |
| `S` | 16384 | the far window, in cache cells |
| `pin` | 128 | document tokens pinned after the sink |
| `trigsurp` | 15 | summed surprisal in nats, on the running unigram model, above which a whole word enters the index and counts as a query term |
| `evtsurp` | 11 | surprisal above which a word the model generates fetches its units into the far window |
| `evrsurp` | 15 | surprisal above which such an event also re-encodes the request and the generated tail (`evrsurp=0`: every event re-encodes) |
| `evrbudget` | 2048 | tokens one turn may spend on re-encodes |
| `seedspan`, `seedwords` | 4096, 32 | the rare words of the recent window's last `seedspan` tokens that join the request's words at the hand-off |
| `unit` | `win0` | `win256` or `win512` replaces sentence units with windows of that many tokens around each occurrence, for code and other volume-heavy text |
| `feed` | 1024 | prefill batch size above the window |
| `store_budget` | 0 | bounds the host store to about this many tokens (0 = unbounded); whole units are evicted by usage, which the paper shows loses needles on long documents |
| `floor` | 0 | `floor=1` runs the window-only floor: near window and pinned prefix, no retrieval |

Other keys in `common/infinite.cpp` are measurement controls and diagnostics; the defaults are the measured
configuration.

## Models used

- Mistral-7B-v0.3, Q8_0, trained range 32,768 tokens
- Llama-3.1-8B, Q8_0 converted from the official weights, trained range 131,072 tokens

## License

MIT, as llama.cpp itself. See `LICENSE`.
