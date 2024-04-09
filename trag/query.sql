
-- name: InsertString :exec
insert into strings (string, hash)
	values (?, ?);

-- name: InsertTestCase :exec
insert into 
	testcases (path_hash, expected_error)
	values (?, ?);

-- name: ClearTestCases :exec
delete from testcases;

-- name: ListTestCases :many
select s.string as path
from testcases tc
	left join strings s
	on (tc.path_hash = s.hash)
order by path asc;

-- name: InsertRun :exec
insert or ignore into runs (path_hash, version, is_strict, error_category, error_message_hash)
values (?, ?, ?, ?, ?);

-- name: AssignGroup :exec
-- Assign a group naem to a path (both represented as a hash). Overrides any
-- previous assignments.
insert or replace into groups (path_hash, group_hash)
values (?, ?);

-- name: DeleteRunsForVersion :exec
delete from runs where version = ?;

-- name: CountSuccessesByVersion :many
select version, count(*) from runs
where error_message_hash is not null
group by version;

-- name: CountFailuresByVersion :many
select version, count(*) from runs
where error_message_hash is null
group by version;






