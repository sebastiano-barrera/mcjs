#!/bin/bash

set -e

if [[ "$#" -lt 1 ]]
then
	echo "usage: $0 <path to test262 test case.js>"
	exit 1
fi

here="$(dirname "$0")"

TEST262_ROOT="$(realpath "${TEST262_ROOT:-$HOME/src/test262/}")"
DBG_ROOT="${here}/../../mcjs_tools/"
cd "$DBG_ROOT"

test_case="$(realpath "$TEST262_ROOT/$1")"
if [ ! -f "$test_case" ]
then
	echo "no such file: $test_case"
	exit 1
fi

cargo run -- \
	"$TEST262_ROOT/harness/assert.js" \
	"$TEST262_ROOT/harness/sta.js" \
	"$test_case"


