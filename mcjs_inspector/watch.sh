#!/bin/sh

export RUST_LOG=debug

filename="${1:-/tmp/mcjs-inspector-0.case}"
echo "case filename: $filename"

systemfd --no-pid -s http::10001 -- cargo watch -s "mold -run cargo run --release --bin mcjs_inspector ${filename}"

