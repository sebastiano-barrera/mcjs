#!/bin/bash

here="$(dirname "$0")"
cd "$here"

sqlite3 ../out/tests.db "select path from general where not success"
