"""One command that answers "is this thing actually working?".

Originally written after 2026-08-14, when three failures had been running silently:

  * the Streamlit process had leaked to 15.2 GB over 35 days, stopped answering
    HTTP, and filled swap;
  * because swap was full, the (since retired) paper book's run took over 5
    minutes and systemd killed it mid-flight every tick — a whole session
    recorded nothing;
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
# Every timer in deploy/systemd plus the app: one source of truth with install.sh.
UNITS = ["nse-signal-lab.service"] + sorted(
    p.name for p in (ROOT / "deploy" / "systemd").glob("*.timer"))
sys.path.insert(0, str(ROOT / "src"))


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


def _prev_trading_day(day):
    from nse_intraday_ai.nse_calendar import is_trading_day

    d = day - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def check_data(now: datetime) -> list[tuple[str, str, str]]:
    """The inputs the 08:45 list is built from must cover the last session."""
    out = []
    files = sorted((ROOT / "data" / "nse_bhav").glob("*/bhav_*.parquet"))
    latest = datetime.strptime(files[-1].stem.split("_")[1], "%Y%m%d").date() if files else None
    # NSE publishes the bhavcopy ~18:00; the morning job fetches it at 08:45.
    need = _prev_trading_day(now.date()) if now.strftime("%H:%M") >= "08:50" else \
        _prev_trading_day(_prev_trading_day(now.date()))
    status = OK if latest and latest >= need else FAIL
    out.append(("NSE bhavcopy", status, f"latest {latest}, need {need}"))
    if DB.exists():
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        try:
            row = con.execute("SELECT MAX(ts) FROM candles WHERE symbol='^NSEI' AND interval='1d'").fetchone()
        finally:
            con.close()
        last = datetime.fromisoformat(row[0]).astimezone(IST).date() if row and row[0] else None
        stale = last is None or (now.date() - last).days > 5
        out.append(("daily candles (fallback rule)", WARN if stale else OK, f"^NSEI 1d through {last}"))
    else:
        out.append(("daily candles (fallback rule)", WARN, "candle cache missing"))
    return out


def market_is_open(now: datetime) -> bool:
    from nse_intraday_ai.nse_calendar import is_trading_day

    return is_trading_day(now.date()) and "09:15" <= now.strftime("%H:%M") <= "15:30"


def check_book(now: datetime) -> list[tuple[str, str, str]]:
    """The validated book: is today's pick list out, and is the record current?"""
    import csv
    import json

    from nse_intraday_ai.nse_calendar import is_trading_day

    out = []
    book = ROOT / "data" / "gap_reversal"
    picks = book / "picks.json"
    if not picks.exists():
        out.append(("gap-reversal picks", FAIL, "data/gap_reversal/picks.json missing"))
    else:
        session = json.loads(picks.read_text()).get("session", "?")
        # From 08:45 on a trading day the list must be for today.
        due = is_trading_day(now.date()) and now.strftime("%H:%M") >= "08:50"
        stale = due and now.strftime("%H:%M") <= "15:30" and session < now.date().isoformat()
        out.append(("gap-reversal picks", FAIL if stale else OK, f"list for {session}"))

    report = ROOT / "data" / "models" / "weekly_report.json"
    if report.exists():
        r = json.loads(report.read_text())
        age = (now.date() - datetime.fromisoformat(r["at"]).date()).days
        out.append(("model review", OK if age <= 9 else WARN,
                    f"{r['at'][:10]} ({age}d ago), {'promoted' if r['promoted'] else 'kept previous'}"))
    else:
        out.append(("weekly model review", WARN, "never run (python scripts/learn.py)"))
    state = ROOT / "data" / "models" / "meta_state.json"
    if state.exists():
        st = json.loads(state.read_text())
        out.append(("drift alarm", FAIL if st.get("alarm") else OK,
                    f"CUSUM {st.get('cusum', 0):.0f}/{st.get('threshold_bps')} bps, "
                    f"learned through {st.get('updated_through')}"))
    record = book / "paper_book.csv"
    if not record.exists():
        out.append(("gap-reversal paper book", WARN, "no session recorded yet"))
        return out
    with record.open() as handle:
        rows = list(csv.DictReader(handle))
    sessions = sorted({r["session"] for r in rows})
    total = sum(float(r["net"]) for r in rows)
    # After 15:45 on a trading day that session must be recorded.
    expected = now.date() if is_trading_day(now.date()) and now.strftime("%H:%M") >= "15:45" else None
    status = WARN if expected and sessions[-1] != expected.isoformat() else OK
    out.append(("gap-reversal paper book", status,
                f"{len(sessions)} sessions, latest {sessions[-1]}, ₹{total:+,.0f} cumulative"))
    return out


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
    checks += check_data(now)
    checks += check_book(now)

    problems = [c for c in checks if c[1] != OK]
    width = max(len(name) for name, _, _ in checks)
    print(f"NSE Gap-Reversal Book health — {now:%Y-%m-%d %H:%M %Z}"
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
    print("  stale bhavcopy : python scripts/gap_reversal.py picks   (fetches NSE data)")
    print("  app not serving: systemctl --user restart nse-signal-lab.service")
    print("  no pick list   : python scripts/gap_reversal.py picks")
    print("  missing session: python scripts/gap_reversal.py record --date YYYY-MM-DD")
    # Warnings are reported but do not fail the unit; only real failures do.
    return 1 if any(c[1] == FAIL for c in problems) else 0


if __name__ == "__main__":
    sys.exit(main())
