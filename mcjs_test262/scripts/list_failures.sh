#!/bin/bash

here="$(dirname "$0")"
cd "$here"

suffix=""
[ -z "$(git status --porcelain)" ] || suffix="-dirty"
version="$(git rev-parse HEAD)$suffix"

sqlite3 ../out/tests.db "select path from general where not success and version like '${version}%'"

