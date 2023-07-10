#!/bin/sh

systemfd --no-pid -s http::10001 -- cargo watch -s 'cargo run --bin mcjs_inspector /tmp/mcjs-inspector-0.case'

