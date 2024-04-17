#!/bin/bash

cd "$(dirname "$0")"

if [ "$#" -lt 2 ]
then
	echo "usage: $0 version_pre version_post"
	echo "(versions are parsed as git commits)"
	exit 1
fi

version_pre="$(git rev-parse "$1")"
version_post="$(git rev-parse "$2")"

echo "# comparing $version_pre -> $version_post"
sqlite3 -readonly -box ../data/tests.db <<EOF 
	with a as (
		select path_hash
		, is_strict
		, (error_message_hash is null) as success
		from runs
		where version = '${version_pre}'
	)
	, b as (
		select path_hash
		, is_strict
		, (error_message_hash is null) as success
		from runs
		where version = '${version_post}'
	)
	select s.string, a.is_strict, iif(b.success, 'NEW SUCCESS!', 'new fail :(') as news
	from a, b, strings s
	where a.path_hash = b.path_hash
		and a.is_strict = b.is_strict
		and a.success <> b.success
		and a.path_hash = s.hash
	order by s.string, a.is_strict, b.success
	;
EOF

