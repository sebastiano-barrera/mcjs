#!/bin/sh

export RUST_LOG=debug

systemfd --no-pid -s http::10001 -- cargo watch -s 'cargo run --release --bin mcjs_inspector /tmp/mcjs-inspector-0.case'

