#!/bin/bash

here="$(dirname "$0")"
cd "$here"

mkdir -p ../out
cargo run -- tests.json > ../out/runs.json

