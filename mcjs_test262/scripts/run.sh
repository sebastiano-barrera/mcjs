#!/bin/bash

here="$(dirname "$0")"
cd "$here"

mkdir -p ../out

if [ "$VERBOSE" = "1" ]
then cargo run -- tests.json >../out/runs.json
else cargo run -- tests.json 2>/dev/null >../out/runs.json
fi





