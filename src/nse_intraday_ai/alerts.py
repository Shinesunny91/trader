"""Automated Instant Mobile Alerts (ntfy.sh, Telegram, WhatsApp).

Push notification channels (in order of ease):

1. **ntfy.sh** (INSTANT, ZERO CONFIG):
   - Install "ntfy" app on your phone (Android/iOS)
   - Subscribe to topic: nse-shine-8123157952
   - Done! You get push alerts immediately.

2. **Telegram Bot** (FREE, 2-min setup):
   - Search @BotFather on Telegram, send /newbot
   - Paste the bot token in the Streamlit sidebar
   - Send /start to your bot, then click Connect

3. **WhatsApp** (link only, no auto-send without paid API)
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

PHONE_NUMBER = "+918123157952"

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "data"
_TG_CONFIG_PATH = _CONFIG_DIR / "telegram_config.json"

# ---------------------------------------------------------------------------
# ntfy.sh — zero-config instant push notifications
# ---------------------------------------------------------------------------

NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "nse-shine-8123157952")
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh")


def send_ntfy(
    message: str,
    title: str = "NSE Trade Alert",
    priority: str = "high",
    tags: str = "chart_with_upwards_trend,moneybag",
    topic: str | None = None,
) -> bool:
    """Send instant push notification via ntfy.sh. Zero config required.

    The user just needs to install the 'ntfy' app on their phone
    and subscribe to the topic (default: nse-shine-8123157952).
    """
    topic = topic or NTFY_TOPIC
    url = f"{NTFY_SERVER}"
    payload = json.dumps({
        "topic": topic,
        "title": title,
        "message": message,
        "priority": 4 if priority == "high" else 3,
        "tags": tags.split(","),
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        print(f"[alerts] ntfy send failed: {exc}")
        return False


def send_trade_alert_ntfy(
    symbol: str,
    side: str,
    entry: float,
    stop_loss: float,
    target_1: float,
    target_2: float,
    confidence: float,
) -> bool:
    """Send a formatted trade alert via ntfy.sh."""
    verb = "BUY (LONG)" if side == "LONG" else "SELL (SHORT)"
    risk = abs(entry - stop_loss)
    reward = abs(target_1 - entry)
    msg = (
        f"{symbol} — {verb}\n"
        f"Confidence: {confidence:.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Entry: ₹{entry:.2f}\n"
        f"Stop Loss: ₹{stop_loss:.2f} (Risk: ₹{risk:.2f})\n"
        f"Target 1: ₹{target_1:.2f} (+₹{reward:.2f})\n"
        f"Target 2: ₹{target_2:.2f}\n"
    )
    return send_ntfy(msg, title=f"🚨 {symbol} — {verb}")


def send_order_ticket_ntfy(ticket_text: str) -> bool:
    """Send a raw order ticket via ntfy.sh."""
    return send_ntfy(ticket_text, title="🚨 NSE Trade Ticket")


# ---------------------------------------------------------------------------
# Telegram Bot API
# ---------------------------------------------------------------------------

def _load_telegram_config() -> dict:
    """Load Telegram bot token and chat_id from config file or env vars."""
    config: dict = {}
    if _TG_CONFIG_PATH.exists():
        try:
            config = json.loads(_TG_CONFIG_PATH.read_text())
        except Exception:
            pass
    config["bot_token"] = os.environ.get("TELEGRAM_BOT_TOKEN", config.get("bot_token", ""))
    config["chat_id"] = os.environ.get("TELEGRAM_CHAT_ID", config.get("chat_id", ""))
    return config


def save_telegram_config(bot_token: str, chat_id: str) -> None:
    """Persist Telegram config to data/telegram_config.json."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _TG_CONFIG_PATH.write_text(json.dumps({
        "bot_token": bot_token,
        "chat_id": chat_id,
    }, indent=2))


def telegram_configured() -> bool:
    """Return True if Telegram bot_token and chat_id are available."""
    cfg = _load_telegram_config()
    return bool(cfg.get("bot_token")) and bool(cfg.get("chat_id"))


def send_telegram(message: str, parse_mode: str = "Markdown") -> bool:
    """Send a Telegram message via Bot API. Returns True on success."""
    cfg = _load_telegram_config()
    token = cfg.get("bot_token", "")
    chat_id = cfg.get("chat_id", "")
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "NSE-Signal-Lab/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            body = json.loads(response.read())
            return body.get("ok", False)
    except Exception as exc:
        print(f"[alerts] Telegram send failed: {exc}")
        return False


def telegram_get_updates(token: str) -> list[dict]:
    """Fetch recent updates from the bot to auto-detect chat_id."""
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NSE-Signal-Lab/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            body = json.loads(response.read())
            return body.get("result", [])
    except Exception as exc:
        print(f"[alerts] Telegram getUpdates failed: {exc}")
        return []


def telegram_setup_interactive(bot_token: str) -> str | None:
    """Auto-detect chat_id by reading recent messages to the bot."""
    updates = telegram_get_updates(bot_token)
    if not updates:
        return None
    for update in reversed(updates):
        msg = update.get("message", {})
        chat = msg.get("chat", {})
        chat_id = chat.get("id")
        if chat_id:
            chat_id_str = str(chat_id)
            save_telegram_config(bot_token, chat_id_str)
            os.environ["TELEGRAM_BOT_TOKEN"] = bot_token
            os.environ["TELEGRAM_CHAT_ID"] = chat_id_str
            send_telegram(
                "✅ *NSE Quant Terminal Connected!*\n\n"
                "You will now receive instant trade alerts here.\n"
                "📊 High-conviction Grade-A+ setups\n"
                "🎯 Entry, Stop Loss & Targets\n"
                "📈 Real-time confidence scores\n\n"
                "_Developer: Shine_"
            )
            return chat_id_str
    return None


def send_order_ticket_telegram(ticket_text: str) -> bool:
    """Send a raw order ticket text block via Telegram."""
    if not telegram_configured():
        return False
    msg = f"🚨 *NSE TRADE TICKET*\n```\n{ticket_text}\n```"
    return send_telegram(msg)


# ---------------------------------------------------------------------------
# WhatsApp helpers
# ---------------------------------------------------------------------------

def get_whatsapp_web_link(message: str, phone: str = "918123157952") -> str:
    """Generate 1-click direct WhatsApp link to send to phone."""
    encoded_text = urllib.parse.quote(message)
    return f"https://api.whatsapp.com/send?phone={phone}&text={encoded_text}"


# ---------------------------------------------------------------------------
# Unified alert dispatcher — sends to ALL configured channels
# ---------------------------------------------------------------------------

def send_all_channels(ticket_text: str) -> dict[str, bool]:
    """Send alert to all available channels. Returns status per channel."""
    results: dict[str, bool] = {}

    # ntfy.sh — always available, zero config
    results["ntfy"] = send_order_ticket_ntfy(ticket_text)

    # Telegram — if configured
    if telegram_configured():
        results["telegram"] = send_order_ticket_telegram(ticket_text)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="NSE Quant Terminal Alert Setup")
    parser.add_argument("--test-ntfy", action="store_true",
                        help="Send a test push notification via ntfy.sh")
    parser.add_argument("--setup-telegram", action="store_true",
                        help="Interactive Telegram bot setup")
    parser.add_argument("--test-telegram", action="store_true",
                        help="Send a test message via Telegram")
    parser.add_argument("--bot-token", type=str, default="",
                        help="Telegram bot token from @BotFather")
    args = parser.parse_args()

    if args.test_ntfy:
        ok = send_ntfy(
            "🧪 Test alert from NSE Quant Terminal\n\nPush notifications are working! ✅",
            title="🧪 Test Alert",
        )
        print("✅ ntfy test sent!" if ok else "❌ ntfy send failed")
        print(f"   Topic: {NTFY_TOPIC}")
        print(f"   Install 'ntfy' app → subscribe to: {NTFY_TOPIC}")

    elif args.setup_telegram:
        token = args.bot_token or input("Paste your Telegram Bot Token from @BotFather: ").strip()
        if not token:
            print("❌ No token provided.")
            sys.exit(1)
        print("Looking for your chat_id (send /start to your bot first)...")
        chat_id = telegram_setup_interactive(token)
        if chat_id:
            print(f"✅ Telegram configured! chat_id = {chat_id}")
        else:
            print("❌ No messages found. Send /start to your bot first.")
            sys.exit(1)

    elif args.test_telegram:
        if not telegram_configured():
            print("❌ Telegram not configured. Run --setup-telegram first.")
            sys.exit(1)
        ok = send_telegram("🧪 *Test Alert*\n\nTelegram alerts working! ✅")
        print("✅ Sent!" if ok else "❌ Failed")

    else:
        parser.print_help()
