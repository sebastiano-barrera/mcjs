#!/bin/bash

here="$(dirname "$0")"
cd "$here"

cargo run --release -- tests.json > out.json
