"""One command that answers "is this thing actually working?".

Written after 2026-08-14, when three failures had been running silently:

  * the Streamlit process had leaked to 15.2 GB over 35 days, stopped answering
    HTTP, and filled swap;
  * because swap was full, the paper book's 60-second run took over 5 minutes
    and systemd killed it mid-flight every tick — a whole session recorded
    nothing;
  * the macro context symbols had gone stale in the candle cache, so the gate
    refused *every* signal for two sessions and the screen simply showed
    nothing, which looks identical to a quiet market.

None of those raised an alarm. Each is trivially detectable. This checks them.

Usage:
    python scripts/health_check.py
    python scripts/health_check.py --quiet   # only print problems (for cron)
"""
from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "candles.sqlite3"

OK, WARN, FAIL = "ok", "warn", "FAIL"
UNITS = [
    "nse-signal-lab.service",
    "nse-scanner.timer",
    "nse-paper-book.timer",
    "nse-context.timer",
    "nse-learn.timer",
    "nse-retrain.timer",
    "nse-logrotate.timer",
]


def _run(*args: str) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def check_units() -> list[tuple[str, str, str]]:
    out = []
    for unit in UNITS:
        state = _run("systemctl", "--user", "is-active", unit) or "unknown"
        status = OK if state in ("active", "running", "waiting") else FAIL
        out.append((f"unit {unit}", status, state))
    linger = "yes" in _run("loginctl", "show-user", str(Path.home().name), "-p", "Linger").lower()
    out.append((
        "survives logout (linger)", OK if linger else WARN,
        "enabled" if linger else "disabled — services die when you log out",
    ))
    return out


def check_app() -> tuple[str, str, str]:
    code = _run("curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                "--max-time", "20", "http://127.0.0.1:8501/")
    return ("webpage http://127.0.0.1:8501",
            OK if code == "200" else FAIL,
            f"HTTP {code or 'no response'}")


def check_memory() -> tuple[str, str, str]:
    """The leak that started all of this."""
    pid = _run("systemctl", "--user", "show", "nse-signal-lab.service", "-p", "MainPID")
    pid = pid.split("=")[-1] if "=" in pid else ""
    if not pid or pid == "0":
        return ("app memory", FAIL, "not running")
    try:
        rss_kb = int(Path(f"/proc/{pid}/status").read_text().split("VmRSS:")[1].split()[0])
    except Exception:
        return ("app memory", WARN, "unreadable")
    gb = rss_kb / 1024 / 1024
    status = OK if gb < 3 else (WARN if gb < 5 else FAIL)
    return ("app memory", status, f"{gb:.1f} GB (cap 5 GB, nightly restart)")


def _freshness(symbols: list[str], label: str, max_age_minutes: int) -> list[tuple[str, str, str]]:
    if not DB.exists():
        return [(label, FAIL, "candle cache missing")]
    now = datetime.now(tz=IST)
    con = sqlite3.connect(str(DB))
    out = []
    try:
        for symbol in symbols:
            row = con.execute(
                "SELECT MAX(ts) FROM candles WHERE symbol=? AND interval='5m'", (symbol,)
            ).fetchone()
            if not row or not row[0]:
                out.append((f"{label} {symbol}", FAIL, "no data at all"))
                continue
            last = datetime.fromisoformat(row[0]).astimezone(IST)
            age = (now - last).total_seconds() / 60
            status = OK if age <= max_age_minutes else FAIL
            out.append((f"{label} {symbol}", status,
                        f"last bar {last:%Y-%m-%d %H:%M} ({age / 60:.1f}h ago)"))
    finally:
        con.close()
    return out


def market_is_open(now: datetime) -> bool:
    return now.weekday() < 5 and "09:15" <= now.strftime("%H:%M") <= "15:30"


def check_record() -> list[tuple[str, str, str]]:
    record = ROOT / "data" / "daily_sim.csv"
    if not record.exists():
        return [("paper-book record", FAIL, "data/daily_sim.csv missing")]
    import csv

    with record.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return [("paper-book record", WARN, "no sessions recorded")]
    last = rows[-1]
    today = datetime.now(tz=IST).date().isoformat()
    weekday = datetime.now(tz=IST).weekday() < 5
    status = OK if (last["date"] == today or not weekday) else FAIL
    total = sum(float(r["net_pnl"]) for r in rows)
    return [
        ("paper-book record", status,
         f"{len(rows)} sessions, latest {last['date']}"),
        ("paper-book cumulative", OK, f"₹{total:+,.0f} over {len(rows)} sessions"),
    ]


def check_disk() -> tuple[str, str, str]:
    out = _run("df", "-h", str(ROOT))
    line = out.splitlines()[-1] if out else ""
    parts = line.split()
    if len(parts) < 5:
        return ("disk", WARN, "unreadable")
    used = int(parts[4].rstrip("%"))
    return ("disk", OK if used < 90 else WARN, f"{parts[4]} used, {parts[3]} free")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="print only problems")
    args = parser.parse_args()

    now = datetime.now(tz=IST)
    checks: list[tuple[str, str, str]] = []
    checks += check_units()
    checks.append(check_app())
    checks.append(check_memory())
    checks.append(check_disk())
    # During market hours the equity context must be minutes old, not days: a
    # stale panel makes the gate refuse everything, silently.
    context_age = 30 if market_is_open(now) else 24 * 60
    checks += _freshness(["^NSEI", "^INDIAVIX", "USDINR=X", "CL=F"], "context", context_age)
    checks += _freshness(["RELIANCE.NS", "HDFCBANK.NS"], "candles",
                         30 if market_is_open(now) else 24 * 60)
    checks += check_record()

    problems = [c for c in checks if c[1] != OK]
    width = max(len(name) for name, _, _ in checks)
    print(f"NSE Signal Lab health — {now:%Y-%m-%d %H:%M %Z}"
          f"  (market {'OPEN' if market_is_open(now) else 'closed'})")
    print("-" * (width + 34))
    for name, status, detail in checks:
        if args.quiet and status == OK:
            continue
        mark = {OK: "  ok ", WARN: " warn", FAIL: " FAIL"}[status]
        print(f"[{mark}] {name:<{width}}  {detail}")
    if not problems:
        print("\nall good.")
        return 0
    print(f"\n{len(problems)} problem(s). Common fixes:")
    print("  stale context  : systemctl --user start nse-context.service")
    print("  app not serving: systemctl --user restart nse-signal-lab.service")
    print("  missing session: python scripts/sim_today.py --date YYYY-MM-DD")
    return 1


if __name__ == "__main__":
    sys.exit(main())
