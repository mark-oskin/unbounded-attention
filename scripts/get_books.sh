#!/usr/bin/env bash
# Fetches the six Project Gutenberg books that the needle and bench modes of llama-inf-bench
# concatenate into a haystack (War and Peace, Les Miserables, The Count of Monte Cristo,
# Don Quixote, Middlemarch, Moby Dick). The tool strips the Gutenberg header and footer itself.
#   usage: scripts/get_books.sh [DIR]      (default DIR = data/books)
set -euo pipefail
OUT=${1:-data/books}
mkdir -p "$OUT"
for id in 2600 135 1184 996 145 2701; do
    if [ ! -s "$OUT/$id.txt" ]; then
        echo "fetching $id"
        curl -fsSL "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" -o "$OUT/$id.txt"
    fi
done
ls -la "$OUT"
