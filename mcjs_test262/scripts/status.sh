#!/bin/bash

cd "$(dirname "$0")"

suffix=""
[ -z "$(git status --porcelain)" ] || suffix="-dirty"
version="$(git rev-parse HEAD)$suffix"

sqlite3 -table ../out/tests.db "select * from status where version like '${version}%'" >../status.txt

