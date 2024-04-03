#!/bin/bash

awk '$1 == "now" {print $2}' \
	| jq -Rs '{test262Root: "/home/sebastiano/src/test262/", testFiles: (split("\n") | map(select(. != "")) | map("test/language/" + .))}'

