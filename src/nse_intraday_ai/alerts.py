"""Automated Instant Mobile Alerts (WhatsApp, Telegram, Webhook).

Enables instant push notifications to the user's mobile device (+918123157952)
whenever a Grade-A+ high-conviction trade setup triggers.

Telegram Setup (FREE, no paid API required):
  1. Open Telegram, search @BotFather, send /newbot
  2. Follow prompts → you get a BOT_TOKEN like 123456:ABCdef...
  3. Start a chat with your new bot, send /start
  4. Run:  python -m nse_intraday_ai.alerts --setup-telegram
     to auto-detect your chat_id
  5. Set env vars or put in data/telegram_config.json:
       TELEGRAM_BOT_TOKEN=123456:ABCdef...
       TELEGRAM_CHAT_ID=987654321
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PHONE_NUMBER = "+918123157952"
CALLMEBOT_API_KEY = os.environ.get("CALLMEBOT_API_KEY", "")

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "data"
_TG_CONFIG_PATH = _CONFIG_DIR / "telegram_config.json"


# ---------------------------------------------------------------------------
# Telegram Bot API helpers
# ---------------------------------------------------------------------------

def _load_telegram_config() -> dict:
    """Load Telegram bot token and chat_id from config file or env vars."""
    config: dict = {}
    # Try config file first
    if _TG_CONFIG_PATH.exists():
        try:
            config = json.loads(_TG_CONFIG_PATH.read_text())
        except Exception:
            pass
    # Env vars override file
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
    """Send a Telegram message via Bot API.  Returns True on success."""
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
            url,
            data=payload,
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
    """Auto-detect chat_id by reading recent messages to the bot.

    Returns the chat_id string, or None if no messages found.
    The user must send /start to the bot before calling this.
    """
    updates = telegram_get_updates(bot_token)
    if not updates:
        return None
    # Take the most recent chat
    for update in reversed(updates):
        msg = update.get("message", {})
        chat = msg.get("chat", {})
        chat_id = chat.get("id")
        if chat_id:
            chat_id_str = str(chat_id)
            save_telegram_config(bot_token, chat_id_str)
            # Send a confirmation
            send_confirmation = (
                "✅ *NSE Quant Terminal Connected!*\n\n"
                "You will now receive instant trade alerts here.\n"
                "📊 High-conviction Grade-A+ setups\n"
                "🎯 Entry, Stop Loss & Targets\n"
                "📈 Real-time confidence scores\n\n"
                "_Developer: Shine_"
            )
            cfg_before = _load_telegram_config()
            # Temporarily set config to send
            os.environ["TELEGRAM_BOT_TOKEN"] = bot_token
            os.environ["TELEGRAM_CHAT_ID"] = chat_id_str
            send_telegram(send_confirmation)
            return chat_id_str
    return None


def format_telegram_trade_message(
    symbol: str,
    side: str,
    entry: float,
    stop_loss: float,
    target_1: float,
    target_2: float,
    confidence: float,
    live_url: str = "",
) -> str:
    """Format an institutional trade alert for Telegram (Markdown)."""
    verb = "BUY (LONG)" if side == "LONG" else "SELL (SHORT)"
    risk = abs(entry - stop_loss)
    reward_1 = abs(target_1 - entry)

    msg = (
        f"🚨 *HIGH-CONVICTION TRADE ALERT*\n"
        f"📈 *Instrument*: {symbol} — {verb}\n"
        f"⭐ *Confidence*: {confidence:.1f}%\n"
        f"─────────────────────\n"
        f"🎯 *Entry Price*: ₹{entry:.2f}\n"
        f"🛑 *Stop Loss*: ₹{stop_loss:.2f} (Risk: ₹{risk:.2f}/sh)\n"
        f"💰 *Target 1 (50%)*: ₹{target_1:.2f} (+₹{reward_1:.2f}/sh)\n"
        f"🏁 *Target 2 (50%)*: ₹{target_2:.2f}\n"
        f"─────────────────────\n"
    )
    if live_url:
        msg += f"📱 *Live Dashboard*: {live_url}"
    return msg


def send_trade_alert_telegram(
    symbol: str,
    side: str,
    entry: float,
    stop_loss: float,
    target_1: float,
    target_2: float,
    confidence: float,
    live_url: str = "",
) -> bool:
    """Format and send a trade alert via Telegram. Returns True on success."""
    if not telegram_configured():
        return False
    msg = format_telegram_trade_message(
        symbol, side, entry, stop_loss, target_1, target_2, confidence, live_url
    )
    return send_telegram(msg)


def send_order_ticket_telegram(ticket_text: str) -> bool:
    """Send a raw order ticket text block via Telegram."""
    if not telegram_configured():
        return False
    msg = f"🚨 *NSE TRADE TICKET*\n```\n{ticket_text}\n```"
    return send_telegram(msg)


# ---------------------------------------------------------------------------
# WhatsApp helpers (existing)
# ---------------------------------------------------------------------------

def format_whatsapp_trade_message(
    symbol: str,
    side: str,
    entry: float,
    stop_loss: float,
    target_1: float,
    target_2: float,
    confidence: float,
    live_url: str = "https://were-grid-residents-others.trycloudflare.com"
) -> str:
    """Format an institutional trade alert ready for WhatsApp."""
    verb = "BUY (LONG)" if side == "LONG" else "SELL (SHORT)"
    risk = abs(entry - stop_loss)
    reward_1 = abs(target_1 - entry)

    msg = (
        f"🚨 *HIGH-CONVICTION TRADE ALERT*\n"
        f"📈 *Instrument*: {symbol} — {verb}\n"
        f"⭐ *Confidence*: {confidence:.1f}%\n"
        f"─────────────────────\n"
        f"🎯 *Entry Price*: ₹{entry:.2f}\n"
        f"🛑 *Stop Loss*: ₹{stop_loss:.2f} (Risk: ₹{risk:.2f}/sh)\n"
        f"💰 *Target 1 (50%)*: ₹{target_1:.2f} (+₹{reward_1:.2f}/sh)\n"
        f"🏁 *Target 2 (50%)*: ₹{target_2:.2f}\n"
        f"─────────────────────\n"
        f"📱 *Live Dashboard*: {live_url}"
    )
    return msg

def send_whatsapp_callmebot(message: str, phone: str = "8123157952", api_key: str = "") -> bool:
    """Send instant WhatsApp alert using CallMeBot API."""
    key = api_key or os.environ.get("CALLMEBOT_API_KEY", "")
    if not key:
        return False

    encoded_text = urllib.parse.quote(message)
    url = f"https://api.callmebot.com/whatsapp.php?phone=+91{phone}&text={encoded_text}&apikey={key}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status == 200
    except Exception as e:
        print(f"Failed to send WhatsApp alert: {e}")
        return False

def get_whatsapp_web_link(message: str, phone: str = "918123157952") -> str:
    """Generate 1-click direct WhatsApp link to send to phone."""
    encoded_text = urllib.parse.quote(message)
    return f"https://api.whatsapp.com/send?phone={phone}&text={encoded_text}"


# ---------------------------------------------------------------------------
# CLI entry point for Telegram setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="NSE Quant Terminal Alert Setup")
    parser.add_argument("--setup-telegram", action="store_true",
                        help="Interactive Telegram bot setup")
    parser.add_argument("--test-telegram", action="store_true",
                        help="Send a test message via Telegram")
    parser.add_argument("--bot-token", type=str, default="",
                        help="Telegram bot token from @BotFather")
    args = parser.parse_args()

    if args.setup_telegram:
        token = args.bot_token or input("Paste your Telegram Bot Token from @BotFather: ").strip()
        if not token:
            print("❌ No token provided. Aborting.")
            sys.exit(1)
        print("Looking for your chat_id (make sure you sent /start to your bot)...")
        chat_id = telegram_setup_interactive(token)
        if chat_id:
            print(f"✅ Telegram configured! chat_id = {chat_id}")
            print(f"   Config saved to: {_TG_CONFIG_PATH}")
        else:
            print("❌ No messages found. Please:")
            print("   1. Open Telegram and find your bot")
            print("   2. Send /start to it")
            print("   3. Run this command again")
            sys.exit(1)

    elif args.test_telegram:
        if not telegram_configured():
            print("❌ Telegram not configured. Run --setup-telegram first.")
            sys.exit(1)
        ok = send_telegram("🧪 *Test Alert from NSE Quant Terminal*\n\nTelegram alerts are working! ✅")
        print("✅ Test message sent!" if ok else "❌ Failed to send test message.")

    else:
        parser.print_help()
