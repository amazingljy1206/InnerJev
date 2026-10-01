"""JSONL input validation and Hugging Face dataset resolution."""
import json
import math
from pathlib import Path


def read_rows(path, training=False, evaluation=False):
    rows = []
    seen = set()
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            ident = row.get('id')
            if not isinstance(ident, str) or not ident or ident in seen:
                raise ValueError(f'Line {line_number}: IDs must be unique nonempty strings.')
            seen.add(ident)
            if row.get('type') not in ('choice', 'noul', 'score') or 'state' not in row:
                raise ValueError(f'{ident}: state and a supported type are required.')
            if not isinstance(row.get('question'), str):
                raise ValueError(f'{ident}: question must be text.')
            options = row.get('options', [])
            if not 2 <= len(options) <= 255:
                raise ValueError(f'{ident}: requires 2–255 options.')
            keys = [o.get('key') for o in options]
            if len(set(keys)) != len(keys) or any(not isinstance(k, str) or not k for k in keys):
                raise ValueError(f'{ident}: option keys must be unique strings.')
            if any(not isinstance(o.get('desc'), str) for o in options):
                raise ValueError(f'{ident}: option descriptions must be text.')
            if row['type'] == 'noul' and len(options) != 2:
                raise ValueError(f'{ident}: noul requires exactly two options.')
            required = ['gold_target'] if evaluation or training else []
            if training:
                required += ['train_target', 'p0']
            for field in required:
                values = row.get(field, [])
                if len(values) != len(options) or any(not math.isfinite(v) or v < 0 for v in values):
                    raise ValueError(f'{ident}: invalid {field}.')
                if abs(sum(values)-1) > 1e-5:
                    raise ValueError(f'{ident}: {field} must sum to one.')
            rows.append(row)
    if not rows:
        raise ValueError('The input is empty.')
    return rows


def resolve_data(dataset, local=None):
    if local:
        root = Path(local)
        return root / 'train.jsonl', root / 'validation.jsonl'
    from huggingface_hub import hf_hub_download
    return tuple(Path(hf_hub_download(dataset, filename, repo_type='dataset'))
                 for filename in ['train.jsonl', 'validation.jsonl'])


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
