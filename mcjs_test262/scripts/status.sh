#!/bin/bash

cd "$(dirname "$0")"

version="$(git rev-parse HEAD)"
[ -z "$(git status --porcelain)" ] || version="dirty"

{
	echo "# version: $version"
	sqlite3 -box ../out/tests.db <<-EOF 
	with q as (
		select dirname
		, success
		, count(*) as count
		from general
		where version = '${version}'
		group by dirname, success 
		order by dirname, success
	)
	, q2 as (
		select dirname
		, ifnull(sum(count) filter (where success = 1), 0) as ok
		, ifnull(sum(count) filter (where success = 0), 0) as fail
		from q
		group by dirname
	)
	select *
	, printf('%6.2f', cast(ok as real) * 100 / (ok + fail)) as progress
	from q2
	;
EOF
} | tee ../out/status.txt

