#!/bin/bash

here="$(dirname "$0")"
cd "$here"

cargo run --release -- tests.json | tee out.json | python3 summary.py

