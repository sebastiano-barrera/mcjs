#!/bin/bash

cd "$(dirname "$0")"

version="$(git rev-parse HEAD)"
[ -z "$(git status --porcelain)" ] || version="dirty"

sqlite3 -table ../out/tests.db "select * from status where version like '${version}%'" >../out/status.txt

