#!/bin/bash

set -e

TEST262_ROOT="$(realpath "${TEST262_ROOT:-$HOME/src/test262/}")"
test_case="$(realpath "$1")"

if [[ "$#" -lt 1 ]]
then
	echo "usage: $0 <path to test262 test case.js>"
	exit 1
fi

if [ "$WATCH" = "1" ]
then
	cargo watch -- cargo run -p mcjs_tools -- \
		"$TEST262_ROOT/harness/assert.js" \
		"$TEST262_ROOT/harness/sta.js" \
		"$test_case"
else
	cargo run -p mcjs_tools -- \
		"$TEST262_ROOT/harness/assert.js" \
		"$TEST262_ROOT/harness/sta.js" \
		"$test_case"
fi

