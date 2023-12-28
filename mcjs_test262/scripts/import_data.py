import sqlite3
import json
import sys
import re
import argparse

IDENT_PATTERN = re.compile(r'^[a-z_][a-z_0-9]*$')

def parse_identifier(s):
    if IDENT_PATTERN.match(s):
        return s
    raise Exception()

def schema_from_record(name, keys):
    keys = set(keys)
    keys.add('file_path')

    for key in keys:
        if not IDENT_PATTERN.match(key):
            raise ValueError('invalid key: ' + key)

    keys_lines = []
    for ndx, key in enumerate(keys):
        if ndx == 0:
            prefix = '( '
        else:
            prefix = ', '
        keys_lines.append(prefix + key + ' varchar')

    create_table_stmt = [
        f'create table {name}'
    ] + keys_lines + [');']

    return [
        f'drop table if exists {name}',
        '\n'.join(create_table_stmt)
    ]

def value_to_sql(v):
    if isinstance(v, str) or v is None:
        return v
    return json.dumps(v)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attr', help='Name of the attribute to import (causes a table of the same name to be created)', required=True, type=parse_identifier)
    args = parser.parse_args()

    db = sqlite3.connect('tests.db', autocommit=False)
    cur = db.cursor()

    schema_created = False

    for line in iter(sys.stdin.readline, ''):
        record = json.loads(line)

        if not schema_created:
            schema_created = True
            schema_stmts = schema_from_record(
                name=args.attr,
                keys=record.keys(),
            )
            for stmt in schema_stmts:
                cur.execute(stmt)

        keys = list(record.keys())
        values = [
            value_to_sql(v)
            for v in record.values()
        ]

        prepared_stmt = 'insert into {} ({}) values ({})'.format(
            args.attr,
            ', '.join(keys),
            ', '.join('?' for _ in values),
        )
        cur.execute(prepared_stmt, values)

    db.commit()

if __name__ == '__main__':
    main()

