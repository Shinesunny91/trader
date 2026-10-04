"""Password gate for the dashboard.

The password is never stored: `data/auth.json` (git-ignored, mode 0600) holds a
random salt and a PBKDF2-SHA256 hash.  Set or change it with

    PYTHONPATH=src .venv/bin/python -m nse_intraday_ai.auth --set-password

`NSE_AUTH_FILE` overrides the file location (used by the tests).
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import os
import secrets
from pathlib import Path

from nse_intraday_ai.atomic_io import atomic_read_json, atomic_write_json

ROOT = Path(__file__).resolve().parents[2]
ITERATIONS = 300_000


def auth_file() -> Path:
    return Path(os.environ.get("NSE_AUTH_FILE", ROOT / "data" / "auth.json"))


def _digest(password: str, salt: bytes, iterations: int) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations).hex()


def set_password(password: str, path: Path | None = None) -> None:
    if len(password) < 6:
        raise ValueError("password must be at least 6 characters")
    path = path or auth_file()
    salt = secrets.token_bytes(16)
    atomic_write_json(path, {"salt": salt.hex(), "iterations": ITERATIONS,
                             "hash": _digest(password, salt, ITERATIONS)})
    os.chmod(path, 0o600)


def is_configured(path: Path | None = None) -> bool:
    rec = atomic_read_json(path or auth_file(), default={}) or {}
    return bool(rec.get("hash") and rec.get("salt"))


def verify(password: str, path: Path | None = None) -> bool:
    rec = atomic_read_json(path or auth_file(), default={}) or {}
    if not rec.get("hash") or not rec.get("salt"):
        return False
    got = _digest(password, bytes.fromhex(rec["salt"]), int(rec.get("iterations", ITERATIONS)))
    return hmac.compare_digest(got, rec["hash"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-password", action="store_true", help="prompt for a new dashboard password")
    args = ap.parse_args()
    if not args.set_password:
        print(f"configured: {is_configured()} ({auth_file()})")
        return
    pw = getpass.getpass("New dashboard password: ")
    if pw != getpass.getpass("Repeat: "):
        raise SystemExit("passwords differ")
    set_password(pw)
    print(f"saved (hashed) to {auth_file()}")


if __name__ == "__main__":
    main()
