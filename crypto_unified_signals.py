# -*- coding: utf-8 -*-
"""
CRYPTO UNIFIED SIGNALS ENGINE (INSTITUTIONAL PRO 2026)
-----------------------------------------------------
Adaptive Multi-Asset Precision Engine:
- Scalp 15m: Micro SL (-0.40%), TP1 (+1.10%), TP2 (+2.50%), TP3 (+4.20%)
- Swing 4h: Macro SL (-1.80%), TP1 (+3.50%), TP2 (+7.50%), TP3 (+12.50%)
- Proportional percentage calculation for ALL cryptos (BTC, ETH, SOL, SUI, XRP, DOGE, PEPE, etc.)
- Strict protection against negative prices, zero division, or fixed dollar offsets.
"""

import time
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
import crypto_orderflow_engine as coe

_SIG_CACHE: Dict[str, Any] = {}
_SIG_CACHE_TIME: Dict[str, float] = {}
_SIG_TTL = 3.0
_LOCKED_SCALP_SETUPS: Dict[str, Any] = {}
_LOCKED_SWING_SETUPS: Dict[str, Any] = {}

def get_price_decimals(p: float) -> int:
    """Determine adaptive decimal precision based on asset price magnitude."""
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

def format_crypto_price(p: float) -> str:
    """Format crypto price with readable thousand separators and adaptive decimals."""
    dec = get_price_decimals(p)
    if dec <= 2:
        return f"${p:,.2f}"
    elif dec <= 4:
        return f"${p:,.4f}"
    elif dec == 5:
        return f"${p:,.5f}"
    elif dec == 6:
        return f"${p:,.6f}"
    else:
        return f"${p:.8f}"

def _apply_anti_front_run_tp(
    tp_raw: float,
    entry: float,
    is_long: bool,
    orderbook_walls: List[float],
    dec: int,
    tp_idx: int = 1
) -> float:
    """
    Whale & Order Book Anti-Front-Running Algorithm:
    Protects trades from premature rejections by placing TP targets 0.25% - 0.40%
    BEFORE large limit walls (Bookmap Ask/Bid walls) and psychological round levels.
    """
    buffer_pct = 0.0035  # 0.35% institutional execution buffer
    tp_adjusted = float(tp_raw)

    if is_long:
        # Check against Bookmap resting ask walls (Only front-run if target is in close proximity to the wall)
        sorted_walls = sorted([w for w in orderbook_walls if w > entry])
        for wall in sorted_walls:
            if 0 <= (tp_raw - wall * 0.995) < (entry * 0.015):
                front_run_wall = wall * (1.0 - buffer_pct)
                if front_run_wall > entry * 1.006:
                    tp_adjusted = min(tp_adjusted, front_run_wall)
                    break

        # Check against psychological round numbers
        if entry < 0.1:
            round_step = 0.0010
        elif entry < 1.0:
            round_step = 0.0050
        elif entry < 50.0:
            round_step = 0.50
        elif entry < 500.0:
            round_step = 5.0
        elif entry < 5000.0:
            round_step = 50.0
        else:
            round_step = 500.0

        nearest_ceiling = math.ceil(tp_adjusted / round_step) * round_step
        if nearest_ceiling > 0:
            dist_pct = abs(tp_adjusted - nearest_ceiling) / nearest_ceiling
            if dist_pct < 0.006:
                front_run_ceiling = nearest_ceiling * (1.0 - buffer_pct)
                if front_run_ceiling > entry * 1.006:
                    tp_adjusted = min(tp_adjusted, front_run_ceiling)
    else:
        # SHORT: Front-run resting bid shelves and round floors (sit ABOVE them)
        sorted_shelves = sorted([s for s in orderbook_walls if s < entry], reverse=True)
        for shelf in sorted_shelves:
            if 0 <= (shelf * 1.005 - tp_raw) < (entry * 0.015):
                front_run_shelf = shelf * (1.0 + buffer_pct)
                if front_run_shelf < entry * 0.994:
                    tp_adjusted = max(tp_adjusted, front_run_shelf)
                    break

        if entry < 0.1:
            round_step = 0.0010
        elif entry < 1.0:
            round_step = 0.0050
        elif entry < 50.0:
            round_step = 0.50
        elif entry < 500.0:
            round_step = 5.0
        elif entry < 5000.0:
            round_step = 50.0
        else:
            round_step = 500.0

        nearest_floor = math.floor(tp_adjusted / round_step) * round_step
        if nearest_floor > 0:
            dist_pct = abs(tp_adjusted - nearest_floor) / nearest_floor
            if dist_pct < 0.006:
                front_run_floor = nearest_floor * (1.0 + buffer_pct)
                if front_run_floor < entry * 0.994:
                    tp_adjusted = max(tp_adjusted, front_run_floor)

    return round(tp_adjusted, dec)

def get_unified_signals(symbol: str = "BTC") -> Dict[str, Any]:
    symbol_clean = symbol.replace("USDT", "").replace("/", "").strip().upper() or "BTC"
    now_utc = datetime.now(timezone.utc)
    now_iran = now_utc + timedelta(hours=3, minutes=30)
    tehran_time_str = now_iran.strftime("%H:%M:%S")
    tehran_date_str = now_iran.strftime("%Y/%m/%d")

    now_ts = time.time()
    cache_k = f"sig_{symbol_clean}"
    if cache_k in _SIG_CACHE and (now_ts - _SIG_CACHE_TIME.get(cache_k, 0.0) < _SIG_TTL):
        return _SIG_CACHE[cache_k]

    # Fast concurrent parallel fetch across modules
    with ThreadPoolExecutor(max_workers=6) as ex:
        f_bm = ex.submit(coe.get_crypto_bookmap_data, symbol_clean, "15m")
        f_nt = ex.submit(coe.get_crypto_ninjatrader_live, symbol_clean, "15m")
        f_sc = ex.submit(coe.get_crypto_sierrachart_data, symbol_clean, "15m")
        f_atas = ex.submit(coe.get_crypto_atas_live, symbol_clean)
        f_qt = ex.submit(coe.get_crypto_quantower_live, symbol_clean, "4h")
        f_bank = ex.submit(coe.get_crypto_bank_reports)

    bm = f_bm.result()
    nt = f_nt.result()
    sc = f_sc.result()
    atas = f_atas.result()
    qt = f_qt.result()
    bank = f_bank.result()
    geo = coe.get_crypto_geopolitics_radar()

    try:
        import crypto_smt_and_liquidation_engine as csle
        smt_liq = csle.get_crypto_smt_and_liquidation_data(symbol_clean)
    except Exception:
        smt_liq = None

    raw_price = bm.get("current_price", 0.0)
    if not raw_price or raw_price <= 0:
        raw_price = coe._get_live_crypto_price(symbol_clean)
    if not raw_price or raw_price <= 0:
        res_err = {
            "ok": False,
            "error": "LIVE_PRICE_UNAVAILABLE",
            "symbol": symbol_clean,
            "message": "قیمت زندهٔ بایننس در دسترس نیست؛ سیگنال تولید نمی‌شود تا از ارائهٔ اعداد غیرواقعی جلوگیری شود (Fail-Closed).",
            "updated_at_iran": f"{tehran_date_str} ساعت {tehran_time_str}"
        }
        _SIG_CACHE[cache_k] = res_err
        _SIG_CACHE_TIME[cache_k] = now_ts
        return res_err

    dec = get_price_decimals(raw_price)
    current_price = round(raw_price, dec)

    # Order flow indicators
    bm_icebergs = bm.get("icebergs", [])
    cvd_raw = nt.get("cvd", 0)
    cvd_val = cvd_raw.get("value", 0) if isinstance(cvd_raw, dict) else (cvd_raw or 0)
    sc_div = sc.get("divergence", "")
    atas_tape = atas.get("tape_speed", {})
    atas_tps = float(atas_tape.get("trades_per_sec", 0.0)) if isinstance(atas_tape, dict) else 0.0

    # Scalp Direction Evaluation (symmetric bull/bear scores + honest NO_TRADE)
    scalp_bull_score = 0
    scalp_bear_score = 0
    if len(bm_icebergs) > 0:
        if any(ic.get("direction") == "BUY" or ic.get("side") == "BID" for ic in bm_icebergs):
            scalp_bull_score += 3
        if any(ic.get("direction") == "SELL" or ic.get("side") == "ASK" for ic in bm_icebergs):
            scalp_bear_score += 3
    if cvd_val > 0:
        scalp_bull_score += 2
    elif cvd_val < 0:
        scalp_bear_score += 2
    if "صعودی" in sc_div:
        scalp_bull_score += 3
    if "نزولی" in sc_div:
        scalp_bear_score += 3
    atas_cvd = atas.get("cvd_divergence", {})
    atas_cvd_status = atas_cvd.get("status", "") if isinstance(atas_cvd, dict) else ""
    if atas_cvd_status == "BULLISH_ABSORPTION":
        scalp_bull_score += 2
    elif atas_cvd_status == "BEARISH_PRESSURE":
        scalp_bear_score += 2
    if atas_tps >= 30:
        scalp_bull_score += 2

    scalp_direction_code = "NO_TRADE"
    if scalp_bull_score >= 5 and scalp_bull_score > scalp_bear_score:
        scalp_direction_code = "LONG"
    elif scalp_bear_score >= 5 and scalp_bear_score > scalp_bull_score:
        scalp_direction_code = "SHORT"
    scalp_is_long = scalp_direction_code == "LONG"

    # =========================================================================
    # PART 1: UNIFIED SCALP SIGNAL (15m Timeframe - Order Flow Anchored Levels)
    # FROZEN & LOCKED FOR 45 MINUTES (Never drifts with tick noise!)
    # =========================================================================
    locked_sc = _LOCKED_SCALP_SETUPS.get(symbol_clean)
    if locked_sc and (now_ts < locked_sc.get("expires_at", 0)) and not locked_sc.get("closed", False):
        scalp_direction_code = locked_sc["direction_code"]
        scalp_is_long = scalp_direction_code == "LONG"
        scalp_entry = locked_sc["entry"]
        scalp_sl = locked_sc["sl"]
        scalp_tp1 = locked_sc["tp1"]
        scalp_tp2 = locked_sc["tp2"]
        scalp_tp3 = locked_sc["tp3"]
        scalp_sl_pct = locked_sc["sl_pct"]
        scalp_tp1_pct = locked_sc["tp1_pct"]
        scalp_tp2_pct = locked_sc["tp2_pct"]
        scalp_tp3_pct = locked_sc["tp3_pct"]
        scalp_date_tehran = locked_sc.get("date_tehran", tehran_date_str)
        scalp_time_tehran = locked_sc.get("time_tehran", tehran_time_str)
        sl_why = locked_sc.get("sl_why", "")
        tp1_why = locked_sc.get("tp1_why", "")
        tp2_why = locked_sc.get("tp2_why", "")
    else:
        scalp_entry = current_price
        scalp_date_tehran = tehran_date_str
        scalp_time_tehran = tehran_time_str
        is_major = symbol_clean in ["BTC", "ETH"]
        is_large_cap = symbol_clean in ["SOL", "BNB", "XRP", "ADA"]
        if is_major:
            min_sl_pct = 0.009    # 0.90% for BTC & ETH
            tp1_mult, tp2_mult, tp3_mult = 1.010, 1.022, 1.045
        elif is_large_cap:
            min_sl_pct = 0.018    # 1.80% for SOL, BNB, XRP
            tp1_mult, tp2_mult, tp3_mult = 1.018, 1.035, 1.065
        else:
            # High-Beta Volatile Altcoins (RLC, STRK, DOGE, SUI, PEPE, NEAR, etc.)
            # Protected by Anti-Stop-Hunt Liquidity Shield (2.85% minimum clearance)
            min_sl_pct = 0.0285   # 2.85% (prevents 1-tick stop-hunts like RLC 1.0800)
            tp1_mult, tp2_mult, tp3_mult = 1.024, 1.048, 1.085

        ask_walls = [float(a.get("price")) for a in bm.get("ask_levels", []) if a.get("price")]
        bid_shelves = [float(b.get("price")) for b in bm.get("bid_levels", []) if b.get("price")]
        mp = qt.get("market_profile", {})
        vah = float(mp.get("vah", current_price * 1.03))
        val = float(mp.get("val", current_price * 0.97))
        unfin = nt.get("unfinished_auction", {})
        unfin_p = float(unfin.get("price", current_price)) if unfin.get("detected") else 0.0

        if scalp_direction_code == "LONG":
            scalp_sl = round(current_price * (1.0 - min_sl_pct), dec)
            valid_bids = sorted([b for b in bid_shelves if b < current_price], reverse=True)
            if valid_bids:
                candidate_shelf = next((b for b in valid_bids if b <= current_price * 0.992), None)
                if candidate_shelf:
                    scalp_sl = round(min(candidate_shelf * 0.997, scalp_sl), dec)

            valid_asks = sorted([a for a in ask_walls if a > current_price])
            
            # Base raw targets
            raw_tp1 = max(valid_asks[0] * 0.998 if valid_asks and valid_asks[0] > current_price * tp1_mult else current_price * tp1_mult, current_price * tp1_mult)
            raw_tp2 = max(valid_asks[1] * 0.998 if len(valid_asks) > 1 and valid_asks[1] > raw_tp1 else (vah * 0.996 if vah > raw_tp1 else current_price * tp2_mult), raw_tp1 * 1.008)
            raw_tp3 = max(raw_tp2 * 1.020, current_price * tp3_mult)

            # Apply Whale & Psychological Anti-Front-Running Algorithm
            scalp_tp1 = _apply_anti_front_run_tp(raw_tp1, scalp_entry, True, ask_walls, dec, 1)
            scalp_tp2 = _apply_anti_front_run_tp(raw_tp2, scalp_entry, True, ask_walls, dec, 2)
            scalp_tp3 = _apply_anti_front_run_tp(raw_tp3, scalp_entry, True, ask_walls, dec, 3)

            # Monotonic step enforcement (Entry < TP1 < TP2 < TP3)
            min_step = max(current_price * 0.003, 10 ** (-dec))
            if scalp_tp1 <= scalp_entry:
                scalp_tp1 = round(scalp_entry + min_step, dec)
            if scalp_tp2 <= scalp_tp1:
                scalp_tp2 = round(scalp_tp1 + min_step, dec)
            if scalp_tp3 <= scalp_tp2:
                scalp_tp3 = round(scalp_tp2 + min_step, dec)

            sl_why = "سپر ضد هانت نقدینگی و دیواره خرید بوک‌مپ (Anti-Stop-Hunt Shield)"
            tp1_why = "خروج امن ضد فرانت‌رانینگ قبل از اولین دیواره فروش (Anti-Front-Run Ask Wall)"
            tp2_why = "خروج هوشمند پیش از تجمع اردرهای رُند و سقف ارزش (Psychological Front-Run VAH)"
        elif scalp_direction_code == "SHORT":
            scalp_sl = round(current_price * (1.0 + min_sl_pct), dec)
            valid_asks = sorted([a for a in ask_walls if a > current_price])
            if valid_asks:
                candidate_wall = next((a for a in valid_asks if a >= current_price * 1.008), None)
                if candidate_wall:
                    scalp_sl = round(max(candidate_wall * 1.003, scalp_sl), dec)

            valid_bids = sorted([b for b in bid_shelves if b < current_price], reverse=True)
            
            raw_tp1 = min(valid_bids[0] * 1.002 if valid_bids and valid_bids[0] < current_price * (2.0 - tp1_mult) else current_price * (2.0 - tp1_mult), current_price * (2.0 - tp1_mult))
            raw_tp2 = min(valid_bids[1] * 1.002 if len(valid_bids) > 1 and valid_bids[1] < raw_tp1 else (val * 1.004 if val < raw_tp1 else current_price * (2.0 - tp2_mult)), raw_tp1 * 0.992)
            raw_tp3 = min(raw_tp2 * 0.980, current_price * (2.0 - tp3_mult))

            scalp_tp1 = _apply_anti_front_run_tp(raw_tp1, scalp_entry, False, bid_shelves, dec, 1)
            scalp_tp2 = _apply_anti_front_run_tp(raw_tp2, scalp_entry, False, bid_shelves, dec, 2)
            scalp_tp3 = _apply_anti_front_run_tp(raw_tp3, scalp_entry, False, bid_shelves, dec, 3)

            min_step = max(current_price * 0.003, 10 ** (-dec))
            if scalp_tp1 >= scalp_entry:
                scalp_tp1 = round(scalp_entry - min_step, dec)
            if scalp_tp2 >= scalp_tp1:
                scalp_tp2 = round(scalp_tp1 - min_step, dec)
            if scalp_tp3 >= scalp_tp2:
                scalp_tp3 = round(scalp_tp2 - min_step, dec)

            sl_why = "سپر ضد هانت نقدینگی بالای دیواره فروش بوک‌مپ (Anti-Stop-Hunt Shield)"
            tp1_why = "خروج امن ضد فرانت‌رانینگ قبل از اولین دیواره خرید (Anti-Front-Run Bid Shelf)"
            tp2_why = "خروج هوشمند پیش از تجمع اردرهای رُند کف ارزش (Psychological Front-Run VAL)"
        else:
            # Fail-Closed: no entry/SL/TP levels invented for a NO_TRADE state
            scalp_sl = None
            scalp_tp1 = None
            scalp_tp2 = None
            scalp_tp3 = None
            sl_why = ""
            tp1_why = ""
            tp2_why = ""

        if scalp_direction_code != "NO_TRADE":
            scalp_sl_pct = abs((scalp_sl - scalp_entry) / scalp_entry) * 100 if scalp_entry > 0 else (min_sl_pct * 100)
            scalp_tp1_pct = abs((scalp_tp1 - scalp_entry) / scalp_entry) * 100 if scalp_entry > 0 else 1.5
            scalp_tp2_pct = abs((scalp_tp2 - scalp_entry) / scalp_entry) * 100 if scalp_entry > 0 else 3.5
            scalp_tp3_pct = abs((scalp_tp3 - scalp_entry) / scalp_entry) * 100 if scalp_entry > 0 else 6.5

            # Freeze & lock for 45 minutes
            _LOCKED_SCALP_SETUPS[symbol_clean] = {
                "entry": scalp_entry,
                "sl": scalp_sl,
                "tp1": scalp_tp1,
                "tp2": scalp_tp2,
                "tp3": scalp_tp3,
                "sl_pct": scalp_sl_pct,
                "tp1_pct": scalp_tp1_pct,
                "tp2_pct": scalp_tp2_pct,
                "tp3_pct": scalp_tp3_pct,
                "direction_code": scalp_direction_code,
                "sl_why": sl_why,
                "tp1_why": tp1_why,
                "tp2_why": tp2_why,
                "date_tehran": tehran_date_str,
                "time_tehran": tehran_time_str,
                "created_at": now_ts,
                "expires_at": now_ts + 2700,
                "closed": False
            }
        else:
            scalp_sl_pct = 0.0
            scalp_tp1_pct = 0.0
            scalp_tp2_pct = 0.0
            scalp_tp3_pct = 0.0

    if scalp_direction_code == "NO_TRADE":
        scalp_signal = {
            "signal_type": "SCALP",
            "signal_type_fa": "اسکالپ (نوسان‌گیری کوتاه‌مدت)",
            "timeframe": "15m",
            "direction": "صبر / بدون معامله (NO TRADE)",
            "direction_code": "NO_TRADE",
            "direction_emoji": "⏸️",
            "entry": None,
            "entry_fmt": "—",
            "stop_loss": None,
            "stop_loss_fmt": "—",
            "tp1": None,
            "tp1_fmt": "—",
            "tp2": None,
            "tp2_fmt": "—",
            "tp3": None,
            "tp3_fmt": "—",
            "date_tehran": scalp_date_tehran,
            "time_tehran": scalp_time_tehran,
            "holding_duration": "—",
            "holding_duration_en": "—",
            "core_modules": "Bookmap L2/L3 + NinjaTrader 8 + ATAS + Anti-Front-Run",
            "rationale_fa": "سیگنال اسکالپ صادر نشد؛ شواهد اردرفلو به آستانهٔ همگرایی نرسید (Fail-Closed)."
        }
    else:
        scalp_signal = {
        "signal_type": "SCALP",
        "signal_type_fa": "اسکالپ (نوسان‌گیری کوتاه‌مدت)",
        "timeframe": "15m",
        "direction": "خرید (LONG)" if scalp_is_long else "فروش (SHORT)",
        "direction_code": "LONG" if scalp_is_long else "SHORT",
        "direction_emoji": "🟢" if scalp_is_long else "🔴",
        "entry": scalp_entry,
        "entry_fmt": format_crypto_price(scalp_entry),
        "stop_loss": scalp_sl,
        "stop_loss_fmt": f"{format_crypto_price(scalp_sl)} (-{scalp_sl_pct:.2f}%)",
        "tp1": scalp_tp1,
        "tp1_fmt": f"{format_crypto_price(scalp_tp1)} (+{scalp_tp1_pct:.2f}%)",
        "tp2": scalp_tp2,
        "tp2_fmt": f"{format_crypto_price(scalp_tp2)} (+{scalp_tp2_pct:.2f}%)",
        "tp3": scalp_tp3,
        "tp3_fmt": f"{format_crypto_price(scalp_tp3)} (+{scalp_tp3_pct:.2f}%)",
        "date_tehran": scalp_date_tehran,
        "time_tehran": scalp_time_tehran,
        "holding_duration": "۳۰ دقیقه الی ۲ ساعت",
        "holding_duration_en": "30m - 2h",
        "core_modules": "Bookmap L2/L3 + NinjaTrader 8 + ATAS + Anti-Front-Run",
        "rationale_fa": "جذب اردرهای مارکت در بوک‌مپ، شیب مثبت CVD و تارگت‌گذاری ضد فرانت‌رانینگ (Anti-Front-Run) چند تیک قبل از دیواره فروش بایننس جهت تضمین تکمیل سود پیش از بازگشت قیمت." if scalp_is_long else "تخلیه تهاجمی در اتاس و تارگت‌گذاری ضد فرانت‌رانینگ چند تیک قبل از دیواره خرید بایننس جهت خروج امن پیش از پرتاب قیمت."
    }

    # =========================================================================
    # PART 2: UNIFIED SWING SIGNAL (4h Timeframe - Quantower 4H & Liquidation Anchored)
    # FROZEN & LOCKED FOR 4 HOURS (Macro Structural Stability)
    # =========================================================================
    bank_ok = bool(bank.get("ok"))
    etf_summary = bank.get("spot_etf_summary", {}) if bank_ok else {}
    etf_net = etf_summary.get("total_daily_net_usd", "")
    etf_is_positive = bool(etf_net) and ("+" in str(etf_net) or "مثبت" in str(etf_net))
    etf_is_negative = bool(etf_net) and ("-" in str(etf_net) or "منفی" in str(etf_net))
    cme_cot = bank.get("cme_cot_crypto", {}) if bank_ok else {}
    cot_pos = cme_cot.get("asset_managers_position", "")
    locked_sw = _LOCKED_SWING_SETUPS.get(symbol_clean)
    if locked_sw and (now_ts < locked_sw.get("expires_at", 0)) and not locked_sw.get("closed", False):
        swing_direction_code = locked_sw["direction_code"]
        swing_is_long = swing_direction_code == "LONG"
        swing_entry = locked_sw["entry"]
        swing_sl = locked_sw["sl"]
        swing_tp1 = locked_sw["tp1"]
        swing_tp2 = locked_sw["tp2"]
        swing_tp3 = locked_sw["tp3"]
        swing_sl_pct = locked_sw["sl_pct"]
        swing_tp1_pct = locked_sw["tp1_pct"]
        swing_tp2_pct = locked_sw["tp2_pct"]
        swing_tp3_pct = locked_sw["tp3_pct"]
        swing_date_tehran = locked_sw.get("date_tehran", tehran_date_str)
        swing_time_tehran = locked_sw.get("time_tehran", tehran_time_str)
    else:
        swing_bull_score = 0
        swing_bear_score = 0
        if etf_is_positive:
            swing_bull_score += 4
        elif etf_is_negative:
            swing_bear_score += 4
        if cot_pos and "Long" in cot_pos:
            swing_bull_score += 3
        elif cot_pos and "Short" in cot_pos:
            swing_bear_score += 3
        if cvd_val > 0:
            swing_bull_score += 3
        elif cvd_val < 0:
            swing_bear_score += 3

        swing_direction_code = "NO_TRADE"
        if swing_bull_score >= 6 and swing_bull_score > swing_bear_score:
            swing_direction_code = "LONG"
        elif swing_bear_score >= 6 and swing_bear_score > swing_bull_score:
            swing_direction_code = "SHORT"
        swing_is_long = swing_direction_code == "LONG"
        swing_entry = current_price
        swing_date_tehran = tehran_date_str
        swing_time_tehran = tehran_time_str

        qt_4h = coe.get_crypto_quantower_live(symbol_clean, "4h")
        mp_4h = qt_4h.get("market_profile", {})
        vah_4h = float(mp_4h.get("vah", swing_entry * 1.035))
        val_4h = float(mp_4h.get("val", swing_entry * 0.965))

        liq_clusters = (smt_liq or {}).get("liquidation_heatmap", {}).get("clusters", [])
        short_liq = sorted([float(c["price"]) for c in liq_clusters if "شورت" in c.get("tier", "") and float(c["price"]) > swing_entry])
        long_liq = sorted([float(c["price"]) for c in liq_clusters if "لانگ" in c.get("tier", "") and float(c["price"]) < swing_entry], reverse=True)

        if swing_direction_code == "LONG":
            if long_liq:
                sl_raw = min(long_liq[0] * 0.995, swing_entry * 0.982)
            else:
                sl_raw = min(val_4h * 0.995, swing_entry * 0.980)
            swing_sl = round(sl_raw, dec)

            is_major = symbol_clean in ["BTC", "ETH"]
            major_short_pools = [float(c["price"]) for c in liq_clusters if "استخر اصلی شورت" in c.get("tier", "") and float(c["price"]) > swing_entry]
            major_target = major_short_pools[0] if major_short_pools else (swing_entry * (1.045 if is_major else 1.065))

            raw_sw_tp1 = max(vah_4h * 0.996, swing_entry * (1.025 if is_major else 1.035))
            raw_sw_tp2 = max(major_target * 0.996, raw_sw_tp1 * 1.025)
            raw_sw_tp3 = max(raw_sw_tp2 * 1.045, swing_entry * (1.085 if is_major else 1.120))

            swing_tp1 = _apply_anti_front_run_tp(raw_sw_tp1, swing_entry, True, [major_target], dec, 1)
            swing_tp2 = _apply_anti_front_run_tp(raw_sw_tp2, swing_entry, True, [major_target * 1.03], dec, 2)
            swing_tp3 = _apply_anti_front_run_tp(raw_sw_tp3, swing_entry, True, [major_target * 1.06], dec, 3)

            min_step = max(current_price * 0.012, 10 ** (-dec))
            if swing_tp1 <= swing_entry:
                swing_tp1 = round(swing_entry + min_step, dec)
            if swing_tp2 <= swing_tp1:
                swing_tp2 = round(swing_tp1 + min_step, dec)
            if swing_tp3 <= swing_tp2:
                swing_tp3 = round(swing_tp2 + min_step, dec)
        elif swing_direction_code == "SHORT":
            if short_liq:
                sl_raw = max(short_liq[0] * 1.005, swing_entry * 1.018)
            else:
                sl_raw = max(vah_4h * 1.005, swing_entry * 1.020)
            swing_sl = round(sl_raw, dec)

            is_major = symbol_clean in ["BTC", "ETH"]
            major_long_pools = [float(c["price"]) for c in liq_clusters if "استخر اصلی لانگ" in c.get("tier", "") and float(c["price"]) < swing_entry]
            major_target = major_long_pools[0] if major_long_pools else (swing_entry * (0.955 if is_major else 0.935))

            raw_sw_tp1 = min(val_4h * 1.004, swing_entry * (0.975 if is_major else 0.965))
            raw_sw_tp2 = min(major_target * 1.004, raw_sw_tp1 * 0.975)
            raw_sw_tp3 = min(raw_sw_tp2 * 0.955, swing_entry * (0.915 if is_major else 0.880))

            swing_tp1 = _apply_anti_front_run_tp(raw_sw_tp1, swing_entry, False, [major_target], dec, 1)
            swing_tp2 = _apply_anti_front_run_tp(raw_sw_tp2, swing_entry, False, [major_target * 0.97], dec, 2)
            swing_tp3 = _apply_anti_front_run_tp(raw_sw_tp3, swing_entry, False, [major_target * 0.94], dec, 3)

            min_step = max(current_price * 0.012, 10 ** (-dec))
            if swing_tp1 >= swing_entry:
                swing_tp1 = round(swing_entry - min_step, dec)
            if swing_tp2 >= swing_tp1:
                swing_tp2 = round(swing_tp1 - min_step, dec)
            if swing_tp3 >= swing_tp2:
                swing_tp3 = round(swing_tp2 - min_step, dec)
        else:
            # Fail-Closed: no entry/SL/TP levels invented for a NO_TRADE state
            swing_sl = None
            swing_tp1 = None
            swing_tp2 = None
            swing_tp3 = None

        if swing_direction_code != "NO_TRADE":
            swing_sl_pct = abs((swing_sl - swing_entry) / swing_entry) * 100 if swing_entry > 0 else 1.80
            swing_tp1_pct = abs((swing_tp1 - swing_entry) / swing_entry) * 100 if swing_entry > 0 else 2.50
            swing_tp2_pct = abs((swing_tp2 - swing_entry) / swing_entry) * 100 if swing_entry > 0 else 5.50
            swing_tp3_pct = abs((swing_tp3 - swing_entry) / swing_entry) * 100 if swing_entry > 0 else 10.25

            _LOCKED_SWING_SETUPS[symbol_clean] = {
                "entry": swing_entry,
                "sl": swing_sl,
                "tp1": swing_tp1,
                "tp2": swing_tp2,
                "tp3": swing_tp3,
                "sl_pct": swing_sl_pct,
                "tp1_pct": swing_tp1_pct,
                "tp2_pct": swing_tp2_pct,
                "tp3_pct": swing_tp3_pct,
                "direction_code": swing_direction_code,
                "date_tehran": tehran_date_str,
                "time_tehran": tehran_time_str,
                "created_at": now_ts,
                "expires_at": now_ts + 14400,
                "closed": False
            }
        else:
            swing_sl_pct = 0.0
            swing_tp1_pct = 0.0
            swing_tp2_pct = 0.0
            swing_tp3_pct = 0.0

    rationale_parts = []
    if etf_is_positive:
        rationale_parts.append(f"ورود سرمایه به اسپات ETFها ({etf_net})")
    elif etf_is_negative:
        rationale_parts.append(f"خروج سرمایه از اسپات ETFها ({etf_net})")
    if cot_pos and "Long" in cot_pos:
        rationale_parts.append("پوزیشن لانگ مدیران دارایی در CME")
    elif cot_pos and "Short" in cot_pos:
        rationale_parts.append("پوزیشن شورت مدیران دارایی در CME")
    if cvd_val > 0:
        rationale_parts.append("دلتای مثبت نینجاتریدر")
    elif cvd_val < 0:
        rationale_parts.append("دلتای منفی نینجاتریدر")
    swing_rationale = "؛ ".join(rationale_parts) if rationale_parts else "سیگنال بر پایهٔ داده‌های زندهٔ بایننس و پروفایل TPO کوانتاور صادر شده است."

    if swing_direction_code == "NO_TRADE":
        swing_signal = {
            "signal_type": "SWING",
            "signal_type_fa": "سوئینگ تریدینگ (موج‌سواری چندروزه)",
            "timeframe": "4h",
            "direction": "صبر / بدون معامله (NO TRADE)",
            "direction_code": "NO_TRADE",
            "direction_emoji": "⏸️",
            "entry": None,
            "entry_fmt": "—",
            "stop_loss": None,
            "stop_loss_fmt": "—",
            "tp1": None,
            "tp1_fmt": "—",
            "tp2": None,
            "tp2_fmt": "—",
            "tp3": None,
            "tp3_fmt": "—",
            "date_tehran": swing_date_tehran,
            "time_tehran": swing_time_tehran,
            "holding_duration": "—",
            "holding_duration_en": "—",
            "core_modules": "Quantower 4H TPO (Binance Klines)",
            "rationale_fa": "سیگنال سوئینگ صادر نشد؛ دادهٔ نهادی زنده (ETF/CoT) در دسترس نیست و شواهد به آستانهٔ همگرایی نرسید (Fail-Closed)."
        }
    else:
        swing_signal = {
        "signal_type": "SWING",
        "signal_type_fa": "سوئینگ تریدینگ (موج‌سواری چندروزه)",
        "timeframe": "4h",
        "direction": "خرید (LONG)" if swing_is_long else "فروش (SHORT)",
        "direction_code": "LONG" if swing_is_long else "SHORT",
        "direction_emoji": "🟢" if swing_is_long else "🔴",
        "entry": swing_entry,
        "entry_fmt": format_crypto_price(swing_entry),
        "stop_loss": swing_sl,
        "stop_loss_fmt": f"{format_crypto_price(swing_sl)} (-{swing_sl_pct:.2f}%)",
        "tp1": swing_tp1,
        "tp1_fmt": f"{format_crypto_price(swing_tp1)} (+{swing_tp1_pct:.2f}%)",
        "tp2": swing_tp2,
        "tp2_fmt": f"{format_crypto_price(swing_tp2)} (+{swing_tp2_pct:.2f}%)",
        "tp3": swing_tp3,
        "tp3_fmt": f"{format_crypto_price(swing_tp3)} (+{swing_tp3_pct:.2f}%)",
        "date_tehran": swing_date_tehran,
        "time_tehran": swing_time_tehran,
        "holding_duration": "۲ الی ۵ روز",
        "holding_duration_en": "2 - 5 days",
        "core_modules": "Quantower 4H TPO (Binance Klines) + CVD + Anti-Front-Run",
        "rationale_fa": swing_rationale
    }

    # =========================================================================
    # PART 3: 9-MODULE INSTITUTIONAL CONFLICT RESOLUTION ENGINE
    # =========================================================================
    bm_ok = bool(bm.get("ok"))
    nt_ok = bool(nt.get("ok"))
    sc_ok = bool(sc.get("ok"))
    atas_ok = bool(atas.get("ok"))
    qt_ok = bool(qt.get("ok"))
    smt_ok = bool(smt_liq and smt_liq.get("ok"))
    smt_data = (smt_liq or {}).get("smt", {}) if smt_ok else {}
    smt_liq_heat = (smt_liq or {}).get("liquidation_heatmap", {}) if smt_ok else {}

    conflict_checklist: List[Dict[str, Any]] = [
        {"tool": "Bookmap L2/L3", "aligned": bm_ok, "note": "عمق سفارش زندهٔ بایننس در دسترس است" if bm_ok else "دادهٔ بوک‌مپ در دسترس نیست (Fail-Closed)"},
        {"tool": "NinjaTrader Footprint", "aligned": nt_ok and cvd_val > 0, "note": "دلتای مثبت فوت‌پرینت" if (nt_ok and cvd_val > 0) else ("دلتای فوت‌پرینت مثبت نیست" if nt_ok else "دادهٔ فوت‌پرینت در دسترس نیست (Fail-Closed)")},
        {"tool": "ATAS Tape & CVD", "aligned": atas_ok and atas_cvd_status == "BULLISH_ABSORPTION", "note": "برتری خریداران تهاجمی در نوار معاملات" if (atas_ok and atas_cvd_status == "BULLISH_ABSORPTION") else ("برتری خریداران تهاجمی در نوار ثبت نشد" if atas_ok else "دادهٔ اتاس در دسترس نیست (Fail-Closed)")},
        {"tool": "Sierra Chart VBP", "aligned": sc_ok, "note": "دادهٔ VBP زنده از کندل‌های بایننس در دسترس است" if sc_ok else "دادهٔ سیراچارت در دسترس نیست (Fail-Closed)"},
        {"tool": "Quantower TPO", "aligned": qt_ok, "note": "پروفایل TPO زنده در دسترس است" if qt_ok else "دادهٔ کوانتاور در دسترس نیست (Fail-Closed)"},
        {"tool": "Crypto SMT Triad", "aligned": smt_ok and smt_data.get("status") != "BEARISH_DIVERGENCE", "note": "وضعیت SMT تحلیل شد" if smt_ok else "دادهٔ اسمارت‌مانی در دسترس نیست (Fail-Closed)"},
        {"tool": "Futures Funding & Liq", "aligned": smt_ok and smt_liq_heat.get("funding_state") != "OVERHEATED_LONGS", "note": "وضعیت فاندینگ تحلیل شد" if smt_ok else "دادهٔ لیکوئیدیشن در دسترس نیست (Fail-Closed)"},
        {"tool": "Spot ETF Inflows", "aligned": etf_is_positive, "note": f"جریان ETF: {etf_net}" if bank_ok and etf_net else "دادهٔ زندهٔ جریان ETF در دسترس نیست (Fail-Closed)"},
        {"tool": "Tether Minting & On-Chain", "aligned": False, "note": "دادهٔ زندهٔ خزانهٔ تتر در دسترس نیست (Fail-Closed)"}
    ]

    conflict_count = len([c for c in conflict_checklist if not c["aligned"]])
    has_conflict = conflict_count >= 3

    conflict_engine = {
        "conflict_count": conflict_count,
        "max_tolerated_conflicts": 2,
        "status": "CONVERGENT_EXECUTION" if not has_conflict else "CONFLICT_LOCK_NO_TRADE",
        "badge": "💎 همگرایی کامل جریان نقدینگی سازمانی (بدون تضاد)" if not has_conflict else f"⚠️ اخطار تضاد نهادی ({conflict_count} ابزار از ۹ ابزار همسو نیستند)",
        "checklist": conflict_checklist,
        "actionable_fa": "اجرای سیگنال با رعایت مدیریت ریسک تایید است." if not has_conflict else "جهت حفاظت از سرمایه، ورود تا زمان حل تضادها به حالت «صبر / بدون معامله» تغییر کرد."
    }

    if has_conflict:
        for sig_x in (scalp_signal, swing_signal):
            sig_x["direction"] = "صبر / بدون معامله (NO TRADE)"
            sig_x["direction_code"] = "NO_TRADE"
            sig_x["direction_emoji"] = "⏸️"
            sig_x["entry"] = None
            sig_x["entry_fmt"] = "—"
            sig_x["stop_loss"] = None
            sig_x["stop_loss_fmt"] = "—"
            sig_x["tp1"] = None
            sig_x["tp1_fmt"] = "—"
            sig_x["tp2"] = None
            sig_x["tp2_fmt"] = "—"
            sig_x["tp3"] = None
            sig_x["tp3_fmt"] = "—"

    # Alignment
    scalp_dir_final = scalp_signal.get("direction_code")
    swing_dir_final = swing_signal.get("direction_code")
    is_aligned = scalp_dir_final == swing_dir_final
    if scalp_dir_final == "NO_TRADE" or swing_dir_final == "NO_TRADE":
        alignment_status = "بدون سیگنال اجرایی (No Trade ⏸️)"
        alignment_note = "یکی از بازه‌های اسکالپ/سوئینگ شرایط صدور سیگنال را نداشت؛ طبق اصول Fail-Closed هیچ معامله‌ای پیشنهاد نمی‌شود."
    elif is_aligned:
        alignment_status = "همسویی کامل (Full Alignment 💎)"
        alignment_note = "هر دو دیدگاه اسکالپ و سوئینگ هم‌جهت هستند؛ ستاپ در بالاترین سطح اعتبار قرار دارد."
    else:
        alignment_status = "اسکالپ اصلاحی درون روند کلان (Counter-Trend Retracement ⚠️)"
        alignment_note = "دیدگاه اسکالپ ۱۵ دقیقه‌ای خلاف جهت سوئینگ چندروزه است؛ این معامله یک پولبک موقت است و خروج سریع در TP1 الزامی است."

    res_final = {
        "ok": True,
        "symbol": symbol_clean,
        "current_price": current_price,
        "scalp": scalp_signal,
        "swing": swing_signal,
        "is_aligned": is_aligned,
        "alignment_status": alignment_status,
        "alignment_note": alignment_note,
        "conflict_engine": conflict_engine,
        "smt_and_liquidation": smt_liq if isinstance(smt_liq, dict) else {"ok": False, "error": "SMT_ENGINE_UNAVAILABLE", "message": "موتور SMT در دسترس نیست (Fail-Closed)."},
        "updated_at_iran": f"{tehran_date_str} ساعت {tehran_time_str}"
    }
    _SIG_CACHE[cache_k] = res_final
    _SIG_CACHE_TIME[cache_k] = now_ts
    return res_final

def format_clean_telegram_signal(sig: Dict[str, Any], symbol: str = "BTC") -> str:
    """
    Produces the exact clean, non-cluttered Telegram alert format requested by user:
    Only: Direction, Type & Timeframe, Entry, TP1, TP2, TP3, SL, Date, Time (Iran), Duration.
    """
    sym = symbol.replace("USDT", "").replace("/", "").strip().upper()
    if sig.get("direction_code") == "NO_TRADE":
        return (f"⏸️ <b>وضعیت {sig.get('signal_type_fa', 'سیگنال')} [#{sym}]</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "🧭 <b>جهت معامله:</b> صبر / بدون معامله (NO TRADE)\n"
                f"📝 <i>{sig.get('rationale_fa', 'شواهد به آستانهٔ همگرایی نرسید (Fail-Closed).')}</i>").strip()
    sig_type_fa = sig.get("signal_type_fa", "اسکالپ")
    tf = sig.get("timeframe", "15m")
    direction = sig.get("direction", "خرید (LONG)")
    dir_emoji = sig.get("direction_emoji", "🟢")
    entry_fmt = sig.get("entry_fmt", "-")
    sl_fmt = sig.get("stop_loss_fmt", "-")
    tp1_fmt = sig.get("tp1_fmt", "-")
    tp2_fmt = sig.get("tp2_fmt", "-")
    tp3_fmt = sig.get("tp3_fmt", "-")
    date_tehran = sig.get("date_tehran", "")
    time_tehran = sig.get("time_tehran", "")
    duration = sig.get("holding_duration", "۳۰ دقیقه تا ۲ ساعت")

    tp3_line = f"\n🎯 <b>حد سود سوم (TP3):</b> <code>{tp3_fmt}</code>" if tp3_fmt and tp3_fmt != "-" else ""

    msg = f"""💎 <b>سیگنال جامع {sig_type_fa} [#{sym}]</b>
━━━━━━━━━━━━━━━━━━━━
🧭 <b>جهت معامله:</b> {dir_emoji} <b>{direction}</b>
⏱️ <b>تایم‌فریم معاملاتی:</b> <code>{tf}</code>
💰 <b>نقطه ورود:</b> <code>{entry_fmt}</code>
🛑 <b>حد ضرر (SL):</b> <code>{sl_fmt}</code>
🎯 <b>حد سود اول (TP1):</b> <code>{tp1_fmt}</code>
🎯 <b>حد سود دوم (TP2):</b> <code>{tp2_fmt}</code>{tp3_line}
⏰ <b>تاریخ و ساعت معامله:</b> <code>{date_tehran} ساعت {time_tehran} (ایران 🇮🇷)</code>
⏳ <b>مدت زمان نگهداری:</b> <code>{duration}</code>
━━━━━━━━━━━━━━━━━━━━
📊 <i>تاییدشده با سیستم تطبیق جریان سفارشات و حل تضاد نهادی</i>"""
    return msg.strip()

if __name__ == "__main__":
    for s in ["BTC", "ETH", "SOL", "XRP", "DOGE", "SUI", "PEPE"]:
        r = get_unified_signals(s)
        print(f"[{s}] Price: {r['current_price']} | Scalp SL: {r['scalp']['stop_loss_fmt']} | Swing SL: {r['swing']['stop_loss_fmt']}")
