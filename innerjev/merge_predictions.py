"""Merge replica outputs while preserving original input order."""
import argparse
import json
from pathlib import Path
from .data import read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--parts', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output file.')
    rows = read_rows(args.input)
    lookup = {}
    for part in sorted(args.parts.glob('part-*.jsonl')):
        for line in part.open():
            pred = json.loads(line)
            if pred['id'] in lookup:
                raise ValueError('Duplicate prediction ID.')
            lookup[pred['id']] = pred
    if set(lookup) != {row['id'] for row in rows}:
        raise ValueError('The replica outputs do not cover exactly the input IDs.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(lookup[row['id']], ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
