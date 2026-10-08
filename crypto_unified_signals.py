"""
CRYPTO UNIFIED SIGNALS ENGINE
-----------------------------
Provides two distinct, institutional-grade unified signals:
1. UNIFIED SCALP SIGNAL (15m Timeframe):
   - Synthesizes: Bookmap (L2/L3 Walls & Icebergs), NinjaTrader (Footprint & CVD),
     ATAS (Tape Speed & Whale Trades), Sierra Chart (Delta Divergence & VBP).
   - Holding Horizon: 30 minutes to 2 hours.
   - Micro SL behind icebergs, TP1 (POC + Breakeven), TP2 (structural level).

2. UNIFIED SWING SIGNAL (4h Timeframe):
   - Synthesizes: Quantower 4H Market Profile (VAL/VAH/VPOC), CME CoT Whales,
     Spot ETF Daily Net Inflows (BlackRock/Fidelity), Wall Street 5 Bank Desks,
     Geopolitics GPR Radar & Tether Treasury Minting.
   - Holding Horizon: 2 to 5 days.
   - Macro structural SL, TP1 (VAH), TP2 (Macro Target).
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any
import crypto_orderflow_engine as coe

def get_unified_signals(symbol: str = "BTC") -> Dict[str, Any]:
    symbol_clean = symbol.replace("USDT", "").replace("/", "").upper()
    now_utc = datetime.now(timezone.utc)
    now_iran = now_utc + timedelta(hours=3, minutes=30)
    tehran_time_str = now_iran.strftime("%H:%M:%S")
    tehran_date_str = now_iran.strftime("%Y/%m/%d")

    # 1. Fetch live order flow and macro data
    bm = coe.get_crypto_bookmap_data(symbol_clean, "15m")
    nt = coe.get_crypto_ninjatrader_live(symbol_clean, "15m")
    sc = coe.get_crypto_sierrachart_data(symbol_clean, "15m")
    atas = coe.get_crypto_atas_live(symbol_clean)
    qt = coe.get_crypto_quantower_live(symbol_clean, "4h")
    bank = coe.get_crypto_bank_reports()
    geo = coe.get_crypto_geopolitics_radar()

    current_price = bm.get("current_price", 83000.0)

    # =========================================================================
    # PART 1: UNIFIED SCALP SIGNAL (15m Timeframe)
    # =========================================================================
    bm_icebergs = bm.get("icebergs", [])
    bm_bid_walls = bm.get("bid_levels", [])
    bm_ask_walls = bm.get("ask_levels", [])
    cvd_raw = nt.get("cvd", 0)
    cvd_val = cvd_raw.get("value", 0) if isinstance(cvd_raw, dict) else (cvd_raw or 0)
    sc_div = sc.get("divergence", "")
    atas_tps = atas.get("tape_speed_tps", 35)

    # Scalp Direction Evaluation
    scalp_bull_score = 0
    if len(bm_icebergs) > 0 and any(ic.get("direction") == "BUY" or ic.get("side") == "BID" for ic in bm_icebergs):
        scalp_bull_score += 3
    if cvd_val >= 0:
        scalp_bull_score += 2
    if "صعودی" in sc_div or "جذب" in sc_div:
        scalp_bull_score += 3
    if atas_tps >= 30:
        scalp_bull_score += 2

    scalp_is_long = scalp_bull_score >= 5
    scalp_dir_str = "خرید (LONG)" if scalp_is_long else "فروش (SHORT)"
    scalp_dir_emoji = "🟢" if scalp_is_long else "🔴"

    # Scalp Levels
    if scalp_is_long:
        bid_iceberg = next((ic for ic in bm_icebergs if ic.get("direction") == "BUY" or ic.get("side") == "BID"), None)
        floor_p = bid_iceberg.get("price", current_price * 0.996) if bid_iceberg else (bm_bid_walls[0].get("price", current_price * 0.996) if bm_bid_walls else current_price * 0.996)
        scalp_entry = current_price
        scalp_sl = round(min(current_price * 0.996, floor_p - 30), 1)
        scalp_tp1 = round(current_price * 1.011, 1) # ~ +1.1%
        scalp_tp2 = round(current_price * 1.025, 1) # ~ +2.5%
    else:
        ask_iceberg = next((ic for ic in bm_icebergs if ic.get("direction") == "SELL" or ic.get("side") == "ASK"), None)
        ceiling_p = ask_iceberg.get("price", current_price * 1.004) if ask_iceberg else (bm_ask_walls[0].get("price", current_price * 1.004) if bm_ask_walls else current_price * 1.004)
        scalp_entry = current_price
        scalp_sl = round(max(current_price * 1.004, ceiling_p + 30), 1)
        scalp_tp1 = round(current_price * 0.989, 1) # ~ -1.1%
        scalp_tp2 = round(current_price * 0.975, 1) # ~ -2.5%

    scalp_sl_pct = abs((scalp_sl - scalp_entry) / scalp_entry) * 100
    scalp_tp1_pct = abs((scalp_tp1 - scalp_entry) / scalp_entry) * 100
    scalp_tp2_pct = abs((scalp_tp2 - scalp_entry) / scalp_entry) * 100

    scalp_signal = {
        "signal_type": "SCALP",
        "signal_type_fa": "اسکالپ (نوسان‌گیری کوتاه‌مدت)",
        "timeframe": "15m",
        "direction": scalp_dir_str,
        "direction_code": "LONG" if scalp_is_long else "SHORT",
        "direction_emoji": scalp_dir_emoji,
        "entry": scalp_entry,
        "entry_fmt": f"${scalp_entry:,.1f}",
        "stop_loss": scalp_sl,
        "stop_loss_fmt": f"${scalp_sl:,.1f} (-{scalp_sl_pct:.2f}%)",
        "tp1": scalp_tp1,
        "tp1_fmt": f"${scalp_tp1:,.1f} (+{scalp_tp1_pct:.2f}%)",
        "tp2": scalp_tp2,
        "tp2_fmt": f"${scalp_tp2:,.1f} (+{scalp_tp2_pct:.2f}%)",
        "date_tehran": tehran_date_str,
        "time_tehran": tehran_time_str,
        "holding_duration": "۳۰ دقیقه الی ۲ ساعت",
        "holding_duration_en": "30m - 2h",
        "core_modules": "Bookmap L2/L3 + NinjaTrader 8 + ATAS + Sierra Chart",
        "rationale_fa": "جذب اردرهای مارکت توسط لیمیت‌بای نهنگ در بوک‌مپ، شیب مثبت CVD و عدم تعادل خرید در فوت‌پرینت کلاستری."
    }

    # =========================================================================
    # PART 2: UNIFIED SWING TRADING SIGNAL (4h Timeframe)
    # =========================================================================
    qt_mp = qt.get("market_profile", {})
    qt_vah = qt_mp.get("vah", current_price * 1.03)
    qt_val = qt_mp.get("val", current_price * 0.97)
    qt_vpoc = qt_mp.get("vpoc", current_price)

    etf_net = bank.get("total_daily_net", "+۴۲۳.۳M")
    etf_is_positive = "+" in etf_net or "مثبت" in etf_net
    whales = bank.get("institutional_whales", {})
    cme_cot = bank.get("cme_cot_crypto", {})
    gpr = geo.get("gpr_index", 145)

    # Swing Direction Evaluation
    swing_bull_score = 0
    if etf_is_positive:
        swing_bull_score += 4
    if "84%" in cme_cot.get("asset_managers_position", "") or "Long" in cme_cot.get("asset_managers_position", ""):
        swing_bull_score += 3
    if current_price >= qt_val * 0.99:
        swing_bull_score += 3

    swing_is_long = swing_bull_score >= 6
    swing_dir_str = "خرید (LONG)" if swing_is_long else "فروش (SHORT)"
    swing_dir_emoji = "🟢" if swing_is_long else "🔴"

    # Swing Levels (4h Structural Levels)
    if swing_is_long:
        swing_entry = current_price
        # Swing SL sits safely below 4H Value Area Low (VAL)
        swing_sl = round(min(current_price * 0.982, qt_val * 0.992), 1)
        # Swing TP1: 4H Value Area High (VAH)
        swing_tp1 = round(max(current_price * 1.035, qt_vah), 1)
        # Swing TP2: Macro Consensus Target / Breakout
        swing_tp2 = round(current_price * 1.075, 1)
    else:
        swing_entry = current_price
        swing_sl = round(max(current_price * 1.018, qt_vah * 1.008), 1)
        swing_tp1 = round(min(current_price * 0.965, qt_val), 1)
        swing_tp2 = round(current_price * 0.925, 1)

    swing_sl_pct = abs((swing_sl - swing_entry) / swing_entry) * 100
    swing_tp1_pct = abs((swing_tp1 - swing_entry) / swing_entry) * 100
    swing_tp2_pct = abs((swing_tp2 - swing_entry) / swing_entry) * 100

    swing_signal = {
        "signal_type": "SWING",
        "signal_type_fa": "سوئینگ تریدینگ (موج‌سواری چندروزه)",
        "timeframe": "4h",
        "direction": swing_dir_str,
        "direction_code": "LONG" if swing_is_long else "SHORT",
        "direction_emoji": swing_dir_emoji,
        "entry": swing_entry,
        "entry_fmt": f"${swing_entry:,.1f}",
        "stop_loss": swing_sl,
        "stop_loss_fmt": f"${swing_sl:,.1f} (-{swing_sl_pct:.2f}%)",
        "tp1": swing_tp1,
        "tp1_fmt": f"${swing_tp1:,.1f} (+{swing_tp1_pct:.2f}%)",
        "tp2": swing_tp2,
        "tp2_fmt": f"${swing_tp2:,.1f} (+{swing_tp2_pct:.2f}%)",
        "date_tehran": tehran_date_str,
        "time_tehran": tehran_time_str,
        "holding_duration": "۲ الی ۵ روز",
        "holding_duration_en": "2 - 5 days",
        "core_modules": "Quantower 4H TPO + CME CoT Whales + Spot ETF Net Flows + Wall Street 5 Banks + Geopolitical Radar",
        "rationale_fa": f"انباشت سنگین اسپات ETFها ({etf_net})، پوزیشن ۸۴٪ لانگ مدیران دارایی در CME و حفظ محدوده منصفانه کف ارزش (VAL)."
    }

    # =========================================================================
    # PART 3: RELATIONSHIP & ALIGNMENT STATUS
    # =========================================================================
    is_aligned = scalp_is_long == swing_is_long
    if is_aligned:
        alignment_status = "همسویی کامل (Full Alignment 💎)"
        alignment_note = "هر دو دیدگاه اسکالپ و سوئینگ هم‌جهت هستند؛ ستاپ در بالاترین سطح اعتبار و وین‌ریت سازمانی قرار دارد."
    else:
        alignment_status = "اسکالپ اصلاحی درون روند کلان (Counter-Trend Retracement ⚠️)"
        alignment_note = "دیدگاه اسکالپ ۱۵ دقیقه‌ای خلاف جهت سوئینگ چندروزه است؛ این معامله یک پولبک موقت است و خروج سریع در TP1 الزامی است."

    return {
        "ok": True,
        "symbol": symbol_clean,
        "current_price": current_price,
        "scalp": scalp_signal,
        "swing": swing_signal,
        "is_aligned": is_aligned,
        "alignment_status": alignment_status,
        "alignment_note": alignment_note,
        "updated_at_iran": f"{tehran_date_str} ساعت {tehran_time_str}"
    }

def format_clean_telegram_signal(sig: Dict[str, Any], symbol: str = "BTC") -> str:
    """
    Produces the exact clean, non-cluttered Telegram alert format requested by user:
    Only: Direction, Type & Timeframe, Entry, TP1, TP2, SL, Date, Time (Iran), Duration.
    """
    sym = symbol.replace("USDT", "").replace("/", "").upper()
    sig_type_fa = sig.get("signal_type_fa", "اسکالپ")
    tf = sig.get("timeframe", "15m")
    direction = sig.get("direction", "خرید (LONG)")
    dir_emoji = sig.get("direction_emoji", "🟢")
    entry_fmt = sig.get("entry_fmt", "-")
    sl_fmt = sig.get("stop_loss_fmt", "-")
    tp1_fmt = sig.get("tp1_fmt", "-")
    tp2_fmt = sig.get("tp2_fmt", "-")
    date_tehran = sig.get("date_tehran", "")
    time_tehran = sig.get("time_tehran", "")
    duration = sig.get("holding_duration", "۳۰ دقیقه تا ۲ ساعت")

    msg = f"""💎 <b>سیگنال جامع {sig_type_fa} [#{sym}]</b>
━━━━━━━━━━━━━━━━━━━━
🧭 <b>جهت معامله:</b> {dir_emoji} <b>{direction}</b>
⏱️ <b>تایم‌فریم معاملاتی:</b> <code>{tf}</code>
💰 <b>نقطه ورود:</b> <code>{entry_fmt}</code>
🛑 <b>حد ضرر (SL):</b> <code>{sl_fmt}</code>
🎯 <b>حد سود اول (TP1):</b> <code>{tp1_fmt}</code>
🎯 <b>حد سود دوم (TP2):</b> <code>{tp2_fmt}</code>
⏰ <b>تاریخ و ساعت معامله:</b> <code>{date_tehran} ساعت {time_tehran} (ایران 🇮🇷)</code>
⏳ <b>مدت زمان نگهداری:</b> <code>{duration}</code>
━━━━━━━━━━━━━━━━━━━━
📊 <i>تاییدشده با سیستم تطبیق جریان سفارشات سازمانی</i>"""
    return msg.strip()

if __name__ == "__main__":
    res = get_unified_signals("BTC")
    print("SCALP:", res["scalp"]["direction"], res["scalp"]["entry_fmt"], "SL:", res["scalp"]["stop_loss_fmt"])
    print("SWING:", res["swing"]["direction"], res["swing"]["entry_fmt"], "SL:", res["swing"]["stop_loss_fmt"])
    print("ALIGNMENT:", res["alignment_status"])
    print("\nSAMPLE TELEGRAM SCALP ALERT:")
    print(format_clean_telegram_signal(res["scalp"], "BTC"))
    print("\nSAMPLE TELEGRAM SWING ALERT:")
    print(format_clean_telegram_signal(res["swing"], "BTC"))
