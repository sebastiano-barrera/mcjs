#!/usr/bin/env python3

from collections import namedtuple
import argparse
import sqlite3
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number',
                        help='Limit summary to this number of commits (starting from the last, backwards).',
                        default=30)
    parser.add_argument('DATABASE', help='Path to the database', default='../data/tests.db')
    args = parser.parse_args()

    db_filename = args.DATABASE
    limit_count = args.number

    db = sqlite3.connect(db_filename)

    res = db.execute('''
    select version
    , sum(iif(error_message_hash is null, 1, 0)) as success_count
    , count(*) as total_count
    from runs
    group by version
    ''')

    stats_by_version = {
        version: dict(
            success_count=success_count,
            total_count=total_count,
        )
        for version, success_count, total_count in res.fetchall()
    }

    output = subprocess.check_output(
        ['git', 'log', '--format=%H|%s', '-n', str(limit_count)],
        encoding='utf8',
    )
    for line in output.splitlines():
        commit_id, summary = line.split('|', 2)

        stats = stats_by_version.get(commit_id)
        if stats:
            success_frac = stats['success_count'] / stats['total_count']
            stats_s = '{}/{} {:4.1f}%'.format(
                stats['success_count'],
                stats['total_count'],
                success_frac * 100,
            )
        else:
            stats_s = '---'
            
        print('{:>20}  {:10}  {}'.format(
            stats_s,
            commit_id[:10], summary,
        ))


if __name__ == '__main__':
    main()


