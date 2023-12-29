#!/bin/bash

here="$(dirname "$0")"
cd "$here"

mkdir -p ../out
cargo run --release -- tests.json > ../out/runs.json

