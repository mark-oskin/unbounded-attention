#!/usr/bin/env bash
# Applies the unbounded-attention patch to a llama.cpp checkout and builds llama-inf-bench.
#
#   usage: ./apply.sh [PATH_TO_LLAMA_CPP]
#
# If PATH_TO_LLAMA_CPP (default: ./llama.cpp) is not a git checkout, ggml-org/llama.cpp is cloned
# there and checked out at the commit in UPSTREAM_COMMIT, the commit the patch was last tested
# against. Extra CMake options go in LLAMA_CMAKE_ARGS, for example
#   LLAMA_CMAKE_ARGS="-DGGML_CUDA=ON" ./apply.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DST=${1:-$HERE/llama.cpp}
PIN=$(head -1 "$HERE/UPSTREAM_COMMIT")

if [ ! -d "$DST/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$DST"
    git -C "$DST" checkout -q "$PIN"
fi

cd "$DST"
git apply --check "$HERE/llama.cpp.patch"
git apply "$HERE/llama.cpp.patch"
cp -R "$HERE/files/common/." common/
mkdir -p tools/inf-bench
cp -R "$HERE/files/tools/inf-bench/." tools/inf-bench/

# shellcheck disable=SC2086
cmake -B build -DCMAKE_BUILD_TYPE=Release ${LLAMA_CMAKE_ARGS:-}
cmake --build build --target llama-inf-bench -j
echo "built: $DST/build/bin/llama-inf-bench"
