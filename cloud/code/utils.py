"""Small file/log helpers shared by the two training stages."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        if keys:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_deadline(config):
    seconds = config.get('per_seed_wall_seconds', 0)
    return time.perf_counter() + seconds if seconds else float('inf')


def remaining(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError('Configured wall-time limit reached; partial output is preserved')


def manifest(out, config, data_dir, **extra):
    package = Path(__file__).resolve().parent
    files = sorted(package.glob('*.py')) + sorted(Path(data_dir).rglob('*.json'))
    save(Path(out) / 'MANIFEST.json', dict(
        created_utc=utc(), config=config, data_dir=str(Path(data_dir).resolve()),
        sha256={str(path.resolve()): digest(path) for path in files}, **extra))
