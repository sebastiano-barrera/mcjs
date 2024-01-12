#!/bin/bash

here="$(dirname "$0")"
cd "$here"

version="$(git rev-parse HEAD)"

sqlite3 ../out/tests.db "select path from general where not success and version like '${version}%'"

