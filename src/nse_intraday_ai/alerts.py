"""Push notifications for the gap-reversal book (ntfy.sh; optional Telegram).

ntfy topic resolution (first hit wins):
  1. env NTFY_TOPIC
  2. data/notify.json  {"ntfy_topic": "..."}   (git-ignored, machine-local)
Nothing personal lives in source.  An ntfy topic is a shared secret — anyone who
knows it can read and post — so use a long random one (e.g. `openssl rand -hex 12`)
and subscribe to it in the ntfy phone app.

Telegram (optional): env TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID, or
`python -m nse_intraday_ai.alerts --setup-telegram`.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from nse_intraday_ai.atomic_io import atomic_read_json, atomic_write_json

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "data"
_NOTIFY_PATH = _CONFIG_DIR / "notify.json"
_TG_CONFIG_PATH = _CONFIG_DIR / "telegram_config.json"
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh")


def ntfy_topic() -> str | None:
    return os.environ.get("NTFY_TOPIC") or (atomic_read_json(_NOTIFY_PATH, default={}) or {}).get("ntfy_topic")


def send_ntfy(message: str, title: str = "NSE Gap-Reversal", priority: str = "high",
              tags: str = "chart_with_downwards_trend", topic: str | None = None) -> bool:
    """Push `message` to the configured ntfy topic. Returns True on HTTP 200."""
    topic = topic or ntfy_topic()
    if not topic:
        print("[alerts] no ntfy topic configured (env NTFY_TOPIC or data/notify.json) — not sent")
        return False
    payload = json.dumps({"topic": topic, "title": title, "message": message,
                          "priority": 4 if priority == "high" else 3,
                          "tags": tags.split(",")}).encode("utf-8")
    try:
        req = urllib.request.Request(NTFY_SERVER, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:                                     # noqa: BLE001 — never break the caller
        print(f"[alerts] ntfy send failed: {exc}")
        return False


def _telegram_config() -> dict:
    cfg = atomic_read_json(_TG_CONFIG_PATH, default={}) or {}
    return {"bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", cfg.get("bot_token", "")),
            "chat_id": os.environ.get("TELEGRAM_CHAT_ID", cfg.get("chat_id", ""))}


def send_telegram(message: str, parse_mode: str = "Markdown") -> bool:
    cfg = _telegram_config()
    if not cfg["bot_token"] or not cfg["chat_id"]:
        return False
    payload = json.dumps({"chat_id": cfg["chat_id"], "text": message, "parse_mode": parse_mode,
                          "disable_web_page_preview": True}).encode("utf-8")
    try:
        req = urllib.request.Request(f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage",
                                     data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as response:
            return bool(json.loads(response.read()).get("ok", False))
    except Exception as exc:                                     # noqa: BLE001
        print(f"[alerts] Telegram send failed: {exc}")
        return False


def _telegram_setup(bot_token: str) -> str | None:
    """Find the chat id from the latest message sent to the bot and save it."""
    try:
        req = urllib.request.Request(f"https://api.telegram.org/bot{bot_token}/getUpdates")
        with urllib.request.urlopen(req, timeout=10) as response:
            updates = json.loads(response.read()).get("result", [])
    except Exception as exc:                                     # noqa: BLE001
        print(f"[alerts] Telegram getUpdates failed: {exc}")
        return None
    for update in reversed(updates):
        chat_id = update.get("message", {}).get("chat", {}).get("id")
        if chat_id:
            atomic_write_json(_TG_CONFIG_PATH, {"bot_token": bot_token, "chat_id": str(chat_id)})
            return str(chat_id)
    return None


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Notification setup and tests")
    parser.add_argument("--set-ntfy-topic", metavar="TOPIC", help="save the ntfy topic to data/notify.json")
    parser.add_argument("--test-ntfy", action="store_true")
    parser.add_argument("--setup-telegram", metavar="BOT_TOKEN")
    parser.add_argument("--test-telegram", action="store_true")
    args = parser.parse_args()
    if args.set_ntfy_topic:
        atomic_write_json(_NOTIFY_PATH, {"ntfy_topic": args.set_ntfy_topic})
        print(f"saved; subscribe to '{args.set_ntfy_topic}' in the ntfy app")
    elif args.test_ntfy:
        print("sent" if send_ntfy("Test alert — push notifications work.", title="Test") else "FAILED")
    elif args.setup_telegram:
        chat = _telegram_setup(args.setup_telegram)
        print(f"Telegram chat_id = {chat}" if chat else "No messages found: send /start to your bot first.")
        sys.exit(0 if chat else 1)
    elif args.test_telegram:
        print("sent" if send_telegram("Test alert — Telegram works.") else "FAILED (configured?)")
    else:
        parser.print_help()
