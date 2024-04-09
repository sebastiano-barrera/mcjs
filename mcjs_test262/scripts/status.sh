#!/bin/bash

cd "$(dirname "$0")"

if [ -z "$1" ]
then
	version="$(git rev-parse HEAD)"
	[ -z "$(git status --porcelain)" ] || version="dirty"
else
	version="$(git rev-parse "$1")"
fi

{
	echo "# version: $version"
	sqlite3 -readonly -box ../out/tests.db <<-EOF 
	with q as (
		select sg.string as dirname
		, (error_message_hash is null) as success
		, count(*) as count
		from runs, groups g, strings sg
		where version = '${version}'
			and g.path_hash = runs.path_hash
			and g.group_hash = sg.hash
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

