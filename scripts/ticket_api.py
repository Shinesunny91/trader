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
    /tickets  data/today_tickets.json as-is
    /record   the accumulated daily_sim.csv track record as JSON
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


def _payload_tickets() -> dict:
    path = DATA / "today_tickets.json"
    if not path.exists():
        return {"error": "no ticket file yet", "tickets": []}
    body = json.loads(path.read_text())
    body["file_age_seconds"] = round(
        datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
    )
    # A stable identity for "have I already told the user about this?".
    ids = [f"{t.get('symbol')}|{t.get('side')}|{t.get('signal_time')}"
           for t in body.get("tickets", [])]
    body["ticket_ids"] = ids
    body["ticket_count"] = len(ids)
    return body


def _payload_record() -> dict:
    path = DATA / "daily_sim.csv"
    if not path.exists():
        return {"sessions": []}
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("gross_pnl", "costs", "net_pnl", "net_pct"):
            if k in r:
                try:
                    r[k] = float(r[k])
                except (TypeError, ValueError):
                    pass
    cum = 0.0
    for r in rows:
        cum += float(r.get("net_pnl") or 0)
        r["cumulative_net"] = round(cum, 2)
    return {"sessions": rows, "session_count": len(rows), "cumulative_net": round(cum, 2)}


def _payload_health() -> dict:
    now = datetime.now(IST)
    files = {}
    for name in ("today_tickets.json", "daily_sim.csv", "scan_state.json"):
        p = DATA / name
        files[name] = {
            "exists": p.exists(),
            "age_seconds": round(now.timestamp() - p.stat().st_mtime) if p.exists() else None,
        }
    return {"ok": True, "now_ist": now.isoformat(timespec="seconds"), "files": files}


def _payload_swing() -> dict:
    """Intra-week candidates, so the phone sees the same book the desktop does."""
    path = DATA / "swing_tickets.json"
    if not path.exists():
        return {"error": "no swing picks yet — run scripts/swing_today.py", "tickets": []}
    body = json.loads(path.read_text())
    body["file_age_seconds"] = round(
        datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
    )
    body["ticket_ids"] = [
        f"swing|{t.get('symbol')}|{t.get('side')}|{body.get('data_through')}"
        for t in body.get("tickets", [])
    ]
    body["ticket_count"] = len(body["ticket_ids"])
    return body


def _payload_signals() -> dict:
    """Live ranked signals with model predictions and cross-sectional rankings."""
    path = DATA / "today_signals.json"
    if not path.exists():
        return {"error": "no signals file yet", "signals": []}
    try:
        body = json.loads(path.read_text())
        body["file_age_seconds"] = round(
            datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
        )
        return body
    except Exception as exc:
        return {"error": str(exc), "signals": []}


def _payload_portfolio() -> dict:
    """Current paper portfolio metrics, open positions, and account summary."""
    trades_path = DATA / "daily_sim.csv"
    tickets_path = DATA / "today_tickets.json"
    trades_count = 0
    net_pnl = 0.0
    if trades_path.exists():
        try:
            with trades_path.open() as fh:
                rows = list(csv.DictReader(fh))
                trades_count = len(rows)
                net_pnl = sum(float(r.get("net_pnl", 0.0) or 0.0) for r in rows)
        except Exception:
            pass
    active_count = 0
    if tickets_path.exists():
        try:
            t_data = json.loads(tickets_path.read_text())
            active_count = len(t_data.get("tickets", []))
        except Exception:
            pass
    return {
        "status": "active",
        "total_sessions": trades_count,
        "cumulative_net_pnl": round(net_pnl, 2),
        "active_today_tickets": active_count,
        "now_ist": datetime.now(IST).isoformat(timespec="seconds"),
    }


def _payload_macro() -> dict:
    """Real-time macro status: NIFTY regime, India VIX, USDINR, Crude tailwinds."""
    try:
        from nse_intraday_ai.market_context import fetch_index_vix_context
        ctx = fetch_index_vix_context()
        return {
            "nifty_regime": getattr(ctx, "index_regime", "UNKNOWN"),
            "nifty_change_pct": getattr(ctx, "nifty_change_pct", 0.0),
            "vix_value": getattr(ctx, "vix_value", None),
            "vix_level": getattr(ctx, "vix_level", "NORMAL"),
            "usdinr_change_pct": getattr(ctx, "usdinr_change_pct", None),
            "crude_change_pct": getattr(ctx, "crude_change_pct", None),
            "fetch_error": getattr(ctx, "fetch_error", None),
            "now_ist": datetime.now(IST).isoformat(timespec="seconds"),
        }
    except Exception as exc:
        return {"error": str(exc), "nifty_regime": "UNKNOWN"}


ROUTES = {
    "/tickets": _payload_tickets,
    "/signals": _payload_signals,
    "/portfolio": _payload_portfolio,
    "/macro": _payload_macro,
    "/swing": _payload_swing,
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
