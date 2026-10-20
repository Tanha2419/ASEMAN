# -*- coding: utf-8 -*-
"""
CRYPTO SMT DIVERGENCE & LIQUIDATION HEATMAP RADAR ENGINE
--------------------------------------------------------
1. Crypto SMT Triad Divergence: BTC vs ETH vs SOL (from REAL Binance 24h changes)
2. Futures Funding Rate Trap Radar (REAL Binance Futures premiumIndex)
Fail-Closed: no fabricated liquidation clusters, funding rates or dominances.
"""

import time
import json
import urllib.request
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import analyzer_engine

TEHRAN_TZ = timezone(timedelta(hours=3, minutes=30))

_CACHE: Dict[str, Any] = {}
_CACHE_TIME: Dict[str, float] = {}
_TTL = 3.0  # seconds

def _get_price_decimals(p: float) -> int:
    if p >= 50:
        return 2
    elif p >= 1:
        return 4
    elif p >= 0.005:
        return 5
    elif p >= 0.0001:
        return 6
    else:
        return 8

def _format_price_adaptive(p: float) -> str:
    dec = _get_price_decimals(p)
    return f"${p:,.{dec}f}"

def _clean_symbol(symbol: str) -> str:
    return symbol.upper().replace("USDT", "").replace("PERP", "").replace("/", "").strip() or "BTC"

def _fetch_live_quote(symbol: str) -> Optional[Dict[str, float]]:
    """Fetch REAL live price and 24h change for a crypto symbol. None on failure (Fail-Closed)."""
    base = _clean_symbol(symbol)
    sym_pair = f"{base}USDT"
    try:
        fetcher = analyzer_engine.CryptoDataFetcher()
        ticker = fetcher.fetch_ticker(sym_pair)
        if ticker and ticker.get("last_price"):
            p = float(ticker["last_price"])
            c = float(ticker.get("price_change_pct", 0.0))
            if p > 0:
                return {"price": p, "chg_24h": c}
    except Exception:
        pass
    return None

def _fetch_real_funding_rate(base: str) -> Optional[float]:
    """Fetch REAL last funding rate (percent) from Binance Futures premiumIndex. None on failure."""
    try:
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={base}USDT"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode())
        last_rate = float(data.get("lastFundingRate", 0))
        return last_rate * 100.0  # decimal -> percent
    except Exception:
        return None

def get_crypto_smt_and_liquidation_data(symbol: str = "BTC") -> Dict[str, Any]:
    """Live Crypto SMT Triad (real 24h changes) + real funding rate. Fail-Closed on any missing quote."""
    base = _clean_symbol(symbol)
    now = time.time()
    cache_key = f"crypto_smt_liq_{base}"

    if cache_key in _CACHE and (now - _CACHE_TIME.get(cache_key, 0.0) < _TTL):
        return _CACHE[cache_key]

    # 1. Fetch REAL live quotes for the Crypto Triad: BTC, ETH, SOL
    btc_data = _fetch_live_quote("BTC")
    eth_data = _fetch_live_quote("ETH")
    sol_data = _fetch_live_quote("SOL")

    if btc_data is None or eth_data is None or sol_data is None:
        res_err = {
            "ok": False,
            "error": "LIVE_QUOTES_UNAVAILABLE",
            "symbol": base,
            "message": "قیمت زندهٔ بایننس برای سه‌قلوی BTC/ETH/SOL در دسترس نیست؛ تحلیل SMT ارائه نمی‌شود تا از اعداد ساختگی جلوگیری شود (Fail-Closed).",
            "updated_at": datetime.now(TEHRAN_TZ).strftime("%H:%M:%S")
        }
        _CACHE[cache_key] = res_err
        _CACHE_TIME[cache_key] = now
        return res_err

    # Current coin price (fail-closed)
    curr_data = _fetch_live_quote(base)
    if curr_data is None:
        res_err = {
            "ok": False,
            "error": "LIVE_QUOTES_UNAVAILABLE",
            "symbol": base,
            "message": f"قیمت زندهٔ {base} در دسترس نیست؛ تحلیل لیکوئیدیشن ارائه نمی‌شود (Fail-Closed).",
            "updated_at": datetime.now(TEHRAN_TZ).strftime("%H:%M:%S")
        }
        _CACHE[cache_key] = res_err
        _CACHE_TIME[cache_key] = now
        return res_err

    p_curr = float(curr_data["price"])

    # 2. SMT DIVERGENCE EVALUATION (from real 24h changes only)
    btc_chg = float(btc_data["chg_24h"])
    eth_chg = float(eth_data["chg_24h"])
    sol_chg = float(sol_data["chg_24h"])

    diff_btc_eth = btc_chg - eth_chg

    if diff_btc_eth > 1.2:
        smt_status = "BULLISH_BTC_SMT"
        smt_badge = "🟢 قدرت نسبی بالاتر بیت‌کوین در ۲۴ ساعت اخیر (SMT)"
        smt_color = "#00e676"
        smt_desc = "تغییر ۲۴ ساعتهٔ بیت‌کوین بیش از ۱.۲ واحد درصد از اتریوم بالاتر است؛ تفسیر رفتاری فراتر از این دادهٔ زنده ارائه نمی‌شود."
        smt_swing_btc = f"تغییر ۲۴ساعته: {btc_chg:+.2f}%"
        smt_swing_eth = f"تغییر ۲۴ساعته: {eth_chg:+.2f}%"
        smt_swing_sol = f"تغییر ۲۴ساعته: {sol_chg:+.2f}%"
    elif diff_btc_eth < -1.2:
        smt_status = "BEARISH_DIVERGENCE"
        smt_badge = "🔴 قدرت نسبی پایین‌تر بیت‌کوین در ۲۴ ساعت اخیر (SMT)"
        smt_color = "#ff3366"
        smt_desc = "تغییر ۲۴ ساعتهٔ بیت‌کوین بیش از ۱.۲ واحد درصد از اتریوم پایین‌تر است؛ تفسیر رفتاری فراتر از این دادهٔ زنده ارائه نمی‌شود."
        smt_swing_btc = f"تغییر ۲۴ساعته: {btc_chg:+.2f}%"
        smt_swing_eth = f"تغییر ۲۴ساعته: {eth_chg:+.2f}%"
        smt_swing_sol = f"تغییر ۲۴ساعته: {sol_chg:+.2f}%"
    else:
        smt_status = "CONFLUENCE"
        smt_badge = "💎 همسویی نسبی سه‌قلوی BTC/ETH/SOL در ۲۴ ساعت اخیر"
        smt_color = "#38bdf8"
        smt_desc = "تغییرات ۲۴ ساعتهٔ سه دارایی در محدودهٔ نزدیک به هم قرار دارد (فاصلهٔ بیت‌کوین تا اتریوم کمتر از ۱.۲ واحد درصد)."
        smt_swing_btc = f"تغییر ۲۴ساعته: {btc_chg:+.2f}%"
        smt_swing_eth = f"تغییر ۲۴ساعته: {eth_chg:+.2f}%"
        smt_swing_sol = f"تغییر ۲۴ساعته: {sol_chg:+.2f}%"

    triad_matrix = [
        {"symbol": "BTC", "name": "بیت‌کوین (لیدر کل)", "price": btc_data["price"], "change_24h": btc_chg, "swing": smt_swing_btc, "dominance": "نامشخص"},
        {"symbol": "ETH", "name": "اتریوم (موتور آلت‌کوین‌ها)", "price": eth_data["price"], "change_24h": eth_chg, "swing": smt_swing_eth, "dominance": "نامشخص"},
        {"symbol": "SOL", "name": "سولانا (نبض اسپکولیشن)", "price": sol_data["price"], "change_24h": sol_chg, "swing": smt_swing_sol, "dominance": "نامشخص"}
    ]

    # 3. LIQUIDATION HEATMAP — Fail-Closed: no invented clusters without a dedicated feed
    liq_clusters: List[Dict[str, Any]] = []
    clusters_note = "خوشه‌های لیکوئیدیشن واقعی نیازمند فید اختصاصی (Open Interest/Force Orders) چند صرافی هستند؛ بدون آن هیچ خوشهٔ ساختگی ارائه نمی‌شود (Fail-Closed)."

    # 4. Funding Rate — REAL Binance Futures premiumIndex
    live_funding_rate = _fetch_real_funding_rate(base)

    if live_funding_rate is None:
        funding_state = "UNAVAILABLE"
        funding_badge = "⚪ فاندینگ ریت زنده در دسترس نیست (Fail-Closed)"
        funding_color = "#9e9e9e"
        funding_advice = "دادهٔ فاندینگ ریت بایننس فیوچرز در دسترس نیست؛ هیچ عدد ساختگی جایگزین آن نشده است."
        hunt_direction = "نامشخص"
        funding_rate_str = "نامشخص"
    elif live_funding_rate > 0.0200:
        funding_state = "OVERHEATED_LONGS"
        funding_badge = f"⚠️ فاندینگ مثبت بالا ({live_funding_rate:+.4f}%) — ریسک لانگ‌اسکوئیز"
        funding_color = "#ff3366"
        funding_advice = "نرخ فاندینگ ۸ ساعتهٔ بایننس بالای ۰.۰۲٪ است؛ پوزیشن‌های لانگ اهرم‌دار هزینهٔ بالایی می‌پردازند و ریسک پاک‌سازی لانگ‌ها وجود دارد."
        hunt_direction = "بالای قیمت (محدودهٔ استاپ لانگ‌ها — فقط هشدار داده‌محور)"
        funding_rate_str = f"{live_funding_rate:+.4f}%"
    elif live_funding_rate < -0.0050:
        funding_state = "NEGATIVE_SHORT_TRAP"
        funding_badge = f"🔥 فاندینگ منفی ({live_funding_rate:+.4f}%) — پتانسیل شورت‌اسکوئیز"
        funding_color = "#00e676"
        funding_advice = "نرخ فاندینگ ۸ ساعتهٔ بایننس منفی است؛ فروشندگان اهرم‌دار کارمزد می‌پردازند و احتمال پرتاب قیمت برای شکار شورت‌ها وجود دارد."
        hunt_direction = "بالای قیمت (محدودهٔ استاپ شورت‌ها — فقط هشدار داده‌محور)"
        funding_rate_str = f"{live_funding_rate:+.4f}%"
    else:
        funding_state = "HEALTHY_BALANCED"
        funding_badge = f"💎 فاندینگ متعادل ({live_funding_rate:+.4f}%)"
        funding_color = "#38bdf8"
        funding_advice = "نرخ فاندینگ ۸ ساعتهٔ بایننس در محدودهٔ نرمال است؛ داغ‌شدگی یک‌طرفه در بازار فیوچرز مشاهده نمی‌شود."
        hunt_direction = "متعادل"
        funding_rate_str = f"{live_funding_rate:+.4f}%"

    res = {
        "ok": True,
        "symbol": base,
        "current_price": p_curr,
        "smt": {
            "status": smt_status,
            "badge": smt_badge,
            "color": smt_color,
            "desc": smt_desc,
            "triad": triad_matrix
        },
        "liquidation_heatmap": {
            "funding_rate": funding_rate_str,
            "funding_state": funding_state,
            "funding_badge": funding_badge,
            "funding_color": funding_color,
            "funding_advice": funding_advice,
            "hunt_direction": hunt_direction,
            "total_short_liq_usd": "نامشخص",
            "total_long_liq_usd": "نامشخص",
            "clusters": liq_clusters,
            "clusters_note": clusters_note
        },
        "updated_at": datetime.now(TEHRAN_TZ).strftime("%H:%M:%S")
    }

    _CACHE[cache_key] = res
    _CACHE_TIME[cache_key] = now
    return res

if __name__ == "__main__":
    data = get_crypto_smt_and_liquidation_data("BTC")
    print("Crypto SMT Badge:", data.get("smt", {}).get("badge"))
    print("Funding Badge:", data.get("liquidation_heatmap", {}).get("funding_badge"))
    print("Hunt Direction:", data.get("liquidation_heatmap", {}).get("hunt_direction"))
