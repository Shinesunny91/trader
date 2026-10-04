"""Atomic file I/O utilities to prevent corruption from concurrent writes.

Systemd timers, the Streamlit app, and scanner daemon all write to shared
JSON state files. Without atomic writes, concurrent access can produce
truncated or corrupted files. This module provides:

- atomic_write_json: Write JSON atomically via temp file + os.replace
- atomic_read_json: Read JSON with retry on transient decode errors
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path | str, data: Any, *, indent: int = 2,
                      default=str) -> None:
    """Write JSON atomically: write to temp file, then os.replace.
    
    os.replace is atomic on POSIX (same filesystem). The temp file is
    created in the same directory to guarantee same-filesystem rename.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp',
                               prefix=f'.{path.stem}_')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=indent, default=default)
            f.write('\n')  # trailing newline for POSIX compliance
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_read_json(path: Path | str, *, default: Any = None,
                     retries: int = 2, delay: float = 0.05) -> Any:
    """Read JSON with retry on transient decode errors.
    
    A concurrent atomic_write_json may briefly make the file unavailable
    (between unlink and replace on some filesystems). Retry handles this.
    """
    path = Path(path)
    for attempt in range(retries + 1):
        try:
            if not path.exists():
                return default
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            if attempt < retries:
                time.sleep(delay)
                continue
            if default is not None:
                return default
            raise


def sqlite_wal_connect(path: Path | str, **kwargs) -> "sqlite3.Connection":
    """Open a SQLite database with WAL mode and safe concurrency defaults.

    WAL (Write-Ahead Logging) allows readers and a single writer to operate
    concurrently without blocking.  Essential when systemd timers, the
    Streamlit app, and the scanner daemon all hit the same database.

    Returns a connection with:
    - journal_mode=WAL (concurrent readers + writer)
    - busy_timeout=5000 (wait 5s instead of failing on lock)
    - synchronous=NORMAL (safe with WAL, faster than FULL)
    """
    import sqlite3
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), **kwargs)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn
