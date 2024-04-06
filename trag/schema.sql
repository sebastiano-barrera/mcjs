CREATE TABLE IF NOT EXISTS runs
	( path_hash varchar
	, is_strict boolean
	, error_category varchar
	, version varchar
	, error_message_hash varchar
	);

CREATE TABLE IF NOT EXISTS testcases
	( path_hash varchar
	, expected_error varchar
	);

CREATE TABLE IF NOT EXISTS strings
	( string varchar
	, hash varchar
	, unique (hash)
	);

CREATE INDEX IF NOT EXISTS testcases__path_hash on testcases (path_hash);
CREATE INDEX IF NOT EXISTS runs__version on runs (version);
CREATE INDEX IF NOT EXISTS runs__path_hash on runs (path_hash);

