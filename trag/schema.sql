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

CREATE TABLE IF NOT EXISTS groups
	( path_hash varchar
	, group_hash varchar
	, unique (path_hash)
	);

DROP INDEX IF EXISTS testcases__path_hash; 
CREATE INDEX testcases__path_hash on testcases (path_hash);

DROP INDEX IF EXISTS runs__version;
CREATE INDEX runs__version on runs (version, path_hash);

DROP INDEX IF EXISTS runs__path_hash;
CREATE INDEX runs__path_hash on runs (path_hash, version);

DROP INDEX IF EXISTS groups__path_hash;
CREATE INDEX groups__path_hash on groups (path_hash);

