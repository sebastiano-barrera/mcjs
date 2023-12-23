import yaml
import argparse
import os
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename', help='')
    args = parser.parse_args()

    input_file = open(args.input_filename)
    report = yaml.full_load(input_file)

    for case in report:
        js_filename = case['params']['chunk_paths'][-1]
        case['group'] = os.path.dirname(js_filename)

    successful_count = sum(case['error'] is None for case in report)
    print('Success (general): {}/{} ({:.1f}%)'.format(
        successful_count,
        len(report),
        100.0 * successful_count / len(report)))

    group_total = Counter()
    group_successful = Counter()
    for case in report:
        g = case['group']
        if case['error'] is None:
            group_successful[g] += 1
        group_total[g] += 1

    for group, total_count in group_total.items():
        total_successful = group_successful.get(group, 0)
        frac = total_successful / total_count 
        print('{:6}/{:6} {:5.1f}% {}'.format(total_successful, total_count, frac * 100, group))


if __name__ == '__main__':
    main()

