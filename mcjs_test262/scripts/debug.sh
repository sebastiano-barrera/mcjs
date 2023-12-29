#!/bin/bash

set -e

TEST262_ROOT="$(realpath "${TEST262_ROOT:-$HOME/src/test262/}")"

here="$(dirname "$0")"
DBG_ROOT="${here}/../../mcjs_tools/"
cd "$DBG_ROOT"

if [[ "$#" -lt 1 ]]
then
	echo "usage: $0 <path to test262 test case.js>"
	exit 1
fi

test_case="$1"

cargo run -- \
	"$TEST262_ROOT/harness/assert.js" \
	"$TEST262_ROOT/harness/sta.js" \
	"$test_case"

