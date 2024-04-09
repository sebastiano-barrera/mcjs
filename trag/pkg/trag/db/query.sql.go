// Code generated by sqlc. DO NOT EDIT.
// versions:
//   sqlc v1.26.0
// source: query.sql

package db

import (
	"context"
	"database/sql"
)

const assignGroup = `-- name: AssignGroup :exec
insert or replace into groups (path_hash, group_hash)
values (?, ?)
`

type AssignGroupParams struct {
	PathHash  string
	GroupHash string
}

// Assign a group naem to a path (both represented as a hash). Overrides any
// previous assignments.
func (q *Queries) AssignGroup(ctx context.Context, arg AssignGroupParams) error {
	_, err := q.db.ExecContext(ctx, assignGroup, arg.PathHash, arg.GroupHash)
	return err
}

const clearTestCases = `-- name: ClearTestCases :exec
delete from testcases
`

func (q *Queries) ClearTestCases(ctx context.Context) error {
	_, err := q.db.ExecContext(ctx, clearTestCases)
	return err
}

const countFailuresByVersion = `-- name: CountFailuresByVersion :many
select version, count(*) from runs
where error_message_hash is null
group by version
`

type CountFailuresByVersionRow struct {
	Version string
	Count   int64
}

func (q *Queries) CountFailuresByVersion(ctx context.Context) ([]CountFailuresByVersionRow, error) {
	rows, err := q.db.QueryContext(ctx, countFailuresByVersion)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var items []CountFailuresByVersionRow
	for rows.Next() {
		var i CountFailuresByVersionRow
		if err := rows.Scan(&i.Version, &i.Count); err != nil {
			return nil, err
		}
		items = append(items, i)
	}
	if err := rows.Close(); err != nil {
		return nil, err
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return items, nil
}

const countSuccessesByVersion = `-- name: CountSuccessesByVersion :many
select version, count(*) from runs
where error_message_hash is not null
group by version
`

type CountSuccessesByVersionRow struct {
	Version string
	Count   int64
}

func (q *Queries) CountSuccessesByVersion(ctx context.Context) ([]CountSuccessesByVersionRow, error) {
	rows, err := q.db.QueryContext(ctx, countSuccessesByVersion)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var items []CountSuccessesByVersionRow
	for rows.Next() {
		var i CountSuccessesByVersionRow
		if err := rows.Scan(&i.Version, &i.Count); err != nil {
			return nil, err
		}
		items = append(items, i)
	}
	if err := rows.Close(); err != nil {
		return nil, err
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return items, nil
}

const deleteRunsForVersion = `-- name: DeleteRunsForVersion :exec
delete from runs where version = ?
`

func (q *Queries) DeleteRunsForVersion(ctx context.Context, version string) error {
	_, err := q.db.ExecContext(ctx, deleteRunsForVersion, version)
	return err
}

const insertRun = `-- name: InsertRun :exec
insert or ignore into runs (path_hash, version, is_strict, error_category, error_message_hash)
values (?, ?, ?, ?, ?)
`

type InsertRunParams struct {
	PathHash         string
	Version          string
	IsStrict         bool
	ErrorCategory    sql.NullString
	ErrorMessageHash sql.NullString
}

func (q *Queries) InsertRun(ctx context.Context, arg InsertRunParams) error {
	_, err := q.db.ExecContext(ctx, insertRun,
		arg.PathHash,
		arg.Version,
		arg.IsStrict,
		arg.ErrorCategory,
		arg.ErrorMessageHash,
	)
	return err
}

const insertString = `-- name: InsertString :exec
insert into strings (string, hash)
	values (?, ?)
`

type InsertStringParams struct {
	String string
	Hash   string
}

func (q *Queries) InsertString(ctx context.Context, arg InsertStringParams) error {
	_, err := q.db.ExecContext(ctx, insertString, arg.String, arg.Hash)
	return err
}

const insertTestCase = `-- name: InsertTestCase :exec
insert or replace into 
	testcases (path_hash, expected_error)
	values (?, ?)
`

type InsertTestCaseParams struct {
	PathHash      string
	ExpectedError sql.NullString
}

func (q *Queries) InsertTestCase(ctx context.Context, arg InsertTestCaseParams) error {
	_, err := q.db.ExecContext(ctx, insertTestCase, arg.PathHash, arg.ExpectedError)
	return err
}

const listTestCases = `-- name: ListTestCases :many
select s.string as path
from testcases tc
	left join strings s
	on (tc.path_hash = s.hash)
order by path asc
`

func (q *Queries) ListTestCases(ctx context.Context) ([]sql.NullString, error) {
	rows, err := q.db.QueryContext(ctx, listTestCases)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var items []sql.NullString
	for rows.Next() {
		var path sql.NullString
		if err := rows.Scan(&path); err != nil {
			return nil, err
		}
		items = append(items, path)
	}
	if err := rows.Close(); err != nil {
		return nil, err
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return items, nil
}
