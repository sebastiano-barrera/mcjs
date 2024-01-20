#!/bin/bash

here="$(dirname "$0")"
cd "$here"

version="$(git rev-parse HEAD)"
[ -z "$(git status --porcelain)" ] || version="dirty"

sqlite3 ../out/tests.db "select path from general where not success and version = '${version}'"

