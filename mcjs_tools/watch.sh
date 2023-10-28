#!/bin/sh

export RUST_LOG=debug

systemfd --no-pid -s http::10001 -- cargo watch -s "mold -run cargo run --bin mcjs_tools -- ../mcjs_vm/test-resources/test-scripts/json5/test_parse.mjs"


