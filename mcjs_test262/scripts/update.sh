#!/bin/bash

here="$(dirname "$0")"
cd "$here"

rm tests.db

jq -c < out.json  | python3 import_data.py --attr outcome

sqlite3 tests.db 'select file_path from outcome' \
	| perl -Mstrict -MJSON -MFile::Basename -ne 'chomp; print encode_json({file_path => $_, dir => dirname($_)}) . "\n"' \
	| python3 import_data.py --attr dir

rg --files-with-matches -wF eval ~/src/test262/test/ \
	| xargs realpath --relative-to="$HOME/src/test262" \
	| jq -Rc '{"file_path": .}' \
	| python3 import_data.py --attr uses_eval

sqlite3 tests.db '
create view general as 
select o.file_path
, o.error is null as success
, dir
, (o.file_path in (select file_path from uses_eval)) as uses_eval 
from outcome o, dir
where o.file_path = dir.file_path
'

sqlite3 -box tests.db "$(cat <<-EOF
with q as (
	select dir
	, success
	, count(*) as count
	from general
	group by dir, success 
	order by dir, success
)
, q2 as (
	select dir
	, sum(count) filter (where success = 1) as ok
	, sum(count) filter (where success = 0) as fail
	from q
	group by dir
)
select *
, cast(ok as real) * 100 / (ok + fail) as progress
from q2
EOF
)" | tee status.txt


