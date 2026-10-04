"""Tiny JSON API for the Android app to poll.

Streamlit serves a UI, not data — the phone cannot ask it "is there a trade?"
without scraping HTML. This exposes the files the book already writes, so the
app can poll cheaply, notify on a genuinely new ticket, and keep an offline
copy on the phone's SD card.

Read-only, stdlib only, binds to the LAN like `run_android_server.sh` does.
There is no authentication: run it on a home network, not a public one.

    python scripts/ticket_api.py --port 8502

Endpoints
    /health   process + data freshness
    /tickets  today's gap-reversal shorts (data/gap_reversal/picks.json)
    /record   the gap-reversal paper book, per session, as JSON
"""
from __future__ import annotations

import argparse
import csv
import json
import socket
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IST = ZoneInfo("Asia/Kolkata")


BOOK = DATA / "gap_reversal"


def _age(path: Path) -> int:
    return round(datetime.now(timezone.utc).timestamp() - path.stat().st_mtime)


def _payload_tickets() -> dict:
    """Today's gap-reversal shorts, in the shape the Android poller expects."""
    path = BOOK / "picks.json"
    if not path.exists():
        return {"error": "no pick list yet", "tickets": [], "ticket_ids": [], "ticket_count": 0}
    body = json.loads(path.read_text())
    square_off = body.get("config", {}).get("square_off", "15:15")
    tickets = []
    for p in body.get("picks", []):
        if p.get("reserve"):
            continue
        stop = (f"BUY SL-M ₹{p['stop_price']:,.2f}" if p.get("stop_price")
                else f"BUY SL-M at fill + ₹{p['stop_distance']:,.2f}")
        entry = (f"SELL LIMIT ₹{p['limit_price']:,.2f} in pre-open (cancel if unfilled at 09:15)"
                 if p.get("limit_price") else "at the open")
        tickets.append({
            "symbol": p["symbol"], "side": p["side"], "quantity": p["quantity"],
            "ticket": f"SELL SHORT {p['quantity']} {p['symbol']} (MIS) {entry} | {stop} | "
                      f"cover at {square_off}",
            **p,
        })
    ids = [f"gap|{body.get('session')}|{t['symbol']}" for t in tickets]
    return {
        "session": body.get("session"),
        "generated_at": body.get("generated_at"),
        "status": f"{len(tickets)} gap-reversal shorts for {body.get('session')}: enter at the "
                  f"open, stop as shown, cover at {square_off}",
        "tickets": tickets, "ticket_ids": ids, "ticket_count": len(ids),
        "reserves": [p["symbol"] for p in body.get("picks", []) if p.get("reserve")],
        "file_age_seconds": _age(path),
    }


def _book_sessions() -> list[dict]:
    path = BOOK / "paper_book.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    per: dict[str, dict] = {}
    for r in rows:
        day = per.setdefault(r["session"], {"date": r["session"], "trades": 0, "net_pnl": 0.0})
        day["trades"] += 1
        day["net_pnl"] = round(day["net_pnl"] + float(r["net"]), 2)
    sessions, cum = [], 0.0
    for day in sorted(per):
        cum += per[day]["net_pnl"]
        sessions.append({**per[day], "cumulative_net": round(cum, 2)})
    return sessions


def _payload_record() -> dict:
    sessions = _book_sessions()
    return {"sessions": sessions, "session_count": len(sessions),
            "cumulative_net": sessions[-1]["cumulative_net"] if sessions else 0.0}


def _payload_health() -> dict:
    now = datetime.now(IST)
    files = {}
    for name in ("gap_reversal/picks.json", "gap_reversal/paper_book.csv",
                 "models/champion.json"):
        p = DATA / name
        files[name] = {"exists": p.exists(), "age_seconds": _age(p) if p.exists() else None}
    return {"ok": True, "now_ist": now.isoformat(timespec="seconds"), "files": files}


def _payload_portfolio() -> dict:
    """The gap-reversal paper book at a glance."""
    sessions = _book_sessions()
    tickets = _payload_tickets()
    return {
        "status": "active",
        "total_sessions": len(sessions),
        "cumulative_net_pnl": sessions[-1]["cumulative_net"] if sessions else 0.0,
        "active_today_tickets": tickets.get("ticket_count", 0),
        "now_ist": datetime.now(IST).isoformat(timespec="seconds"),
    }


ROUTES = {
    "/tickets": _payload_tickets,
    "/portfolio": _payload_portfolio,
    "/record": _payload_record,
    "/health": _payload_health,
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:                                   # noqa: N802
        route = self.path.split("?")[0].rstrip("/") or "/health"
        handler = ROUTES.get(route)
        if handler is None:
            self._send(404, {"error": "not found", "routes": sorted(ROUTES)})
            return
        try:
            self._send(200, handler())
        except Exception as exc:                                # noqa: BLE001
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args) -> None:             # quieter
        return


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8502)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ticket API on http://{lan_ip()}:{args.port}  (routes: {', '.join(sorted(ROUTES))})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
