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
select is_strict, path.string as path 
from runs, strings path 
where path_hash = path.hash
	and version = '${version}'
	and error_message_hash is not null
order by path;
EOF
}



