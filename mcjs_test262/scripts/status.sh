#!/bin/bash

cd "$(dirname "$0")"

sqlite3 -table ../out/tests.db  'select * from status' >../status.txt

