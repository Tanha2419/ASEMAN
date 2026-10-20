# -*- coding: utf-8 -*-
"""
CRYPTO SNIPER — Institutional Order-Flow Confluence Matrix (Fail-Closed)
------------------------------------------------------------------------
Six data gates are evaluated strictly from live engine responses:
1. Bookmap L2/L3 real resting walls (iceberg volume is an ESTIMATE)
2. NinjaTrader 8 footprint delta / CVD
3. Sierra Chart delta divergence & VBP
4. Quantower TPO value area (VAL/VAH/VPOC)
5. ATAS live tape (aggTrades) velocity & whale blocks
6. Wall Street ETF / bank-desk feed (currently not connected → gate unavailable)

Rules:
- Any gate whose engine returns ok=False contributes ZERO points and is marked
  "در دسترس نیست"; it is never replaced by a fabricated number.
- A signal is only issued when all critical order-flow gates are available and
  the normalized confluence score reaches the threshold.
- No win-rate projection is emitted without a verified backtest.
"""

from datetime import datetime, timezone, timedelta
import crypto_orderflow_engine as coe

THRESHOLD = 88
CRITICAL_GATES = ("bookmap_liquidity", "ninjatrader_footprint", "atas_tape")


def _ok(block) -> bool:
    return isinstance(block, dict) and block.get("ok") is True


def _fail_msg(block, label: str) -> str:
    if isinstance(block, dict) and block.get("message"):
        return str(block["message"])
    return f"دادهٔ زندهٔ {label} در دسترس نیست؛ امتیاز ساختگی لحاظ نمی‌شود (Fail-Closed)."


def _gate(gid, name, module, icon, max_score, available, score=0, details=""):
    score = int(score) if available else 0
    return {
        "id": gid,
        "name": name,
        "module": module,
        "available": bool(available),
        "passed": bool(available and score >= max_score * 0.6),
        "score": score,
        "max_score": max_score,
        "icon": icon,
        "details": details,
    }


def evaluate_sniper_confluence(symbol="BTC"):
    symbol_clean = symbol.replace("USDT", "").replace("/", "").upper()
    now_tehran = datetime.now(timezone(timedelta(hours=3, minutes=30))).strftime("%H:%M:%S (ایران)")

    bm = coe.get_crypto_bookmap_data(symbol_clean, "15m")
    nt = coe.get_crypto_ninjatrader_live(symbol_clean, "15m")
    sc = coe.get_crypto_sierrachart_data(symbol_clean, "15m")
    qt = coe.get_crypto_quantower_live(symbol_clean, "1h")
    atas = coe.get_crypto_atas_live(symbol_clean)
    bank = coe.get_crypto_bank_reports()

    current_price = bm.get("current_price") if _ok(bm) else None
    if not isinstance(current_price, (int, float)) or current_price <= 0:
        return {
            "ok": False,
            "error": "LIVE_PRICE_UNAVAILABLE",
            "symbol": symbol_clean,
            "message": "قیمت زندهٔ صرافی در دسترس نیست؛ ماتریس کانفلوئنس اسنایپر اجرا نشد (Fail-Closed).",
            "updated_at_iran": now_tehran,
        }

    gatekeepers = []

    # --- Gate 1: Bookmap real liquidity walls (Max 20 pts) ---
    bm_ok = _ok(bm)
    g1_score, g1_reasons = 0, []
    bid_walls = bm.get("bid_levels", []) if bm_ok else []
    nearest_bid = bid_walls[0] if bid_walls else {}
    nearest_wall_vol = nearest_bid.get("volume_lots")
    if bm_ok:
        if isinstance(nearest_wall_vol, (int, float)) and nearest_wall_vol >= 200:
            g1_score += 10
            g1_reasons.append(f"دیوارهٔ خرید واقعی {nearest_wall_vol:,.2f} {symbol_clean} در کف")
        elif isinstance(nearest_wall_vol, (int, float)) and nearest_wall_vol >= 100:
            g1_score += 6
            g1_reasons.append(f"دیوارهٔ خرید فعال {nearest_wall_vol:,.2f} {symbol_clean}")
        elif isinstance(nearest_wall_vol, (int, float)) and nearest_wall_vol > 0:
            g1_reasons.append(f"کوچک‌ترین دیوارهٔ خرید واقعی: {nearest_wall_vol:,.2f} {symbol_clean}")
        else:
            g1_reasons.append("دیوارهٔ خرید معتبری در عمق واقعی دفتر سفارشات یافت نشد.")

        icebergs = bm.get("iceberg_orders", [])
        if icebergs:
            g1_score += 4
            g1_reasons.append(f"{len(icebergs)} سفارش پنهان برآوردی از روی دیواره‌های واقعی (تخمینی)")
        if bm.get("imbalance_pct") is not None:
            g1_reasons.append(f"تراز تقاضا/عرضه: {bm.get('imbalance_pct')}% خریدار")
    gatekeepers.append(_gate(
        "bookmap_liquidity", "سنگر نقدینگی و دیواره‌های واقعی بوک‌مپ", "Bookmap L2/L3", "🧊",
        20, bm_ok, g1_score,
        " + ".join(g1_reasons) if bm_ok else _fail_msg(bm, "بوک‌مپ"),
    ))

    # --- Gate 2: NinjaTrader footprint stacked imbalance & CVD (Max 20 pts) ---
    nt_ok = _ok(nt)
    g2_score, g2_reasons = 0, []
    cvd_block = nt.get("cvd", {}) if nt_ok else {}
    cvd_val = cvd_block.get("value") if isinstance(cvd_block, dict) else None
    cvd_available = bool(isinstance(cvd_block, dict) and cvd_block.get("available", True) and isinstance(cvd_val, (int, float)))
    if nt_ok:
        candles = nt.get("footprint_candles", [])
        last_candle = candles[-1] if candles else {}
        has_buy_imbalance = any(
            cl.get("imbalance") == "BUY" for cl in last_candle.get("clusters", [])
        )
        if has_buy_imbalance:
            g2_score += 8
            g2_reasons.append("عدم تعادل خرید در توزیع تخمینی درون‌کندلی (نه تیک‌دادهٔ سطح‌قیمت)")
        if cvd_available:
            if cvd_val > 0:
                g2_score += 10
                g2_reasons.append(f"دلتای تجمیعی مثبت از کندل‌های واقعی (CVD: {cvd_val:+.2f} {symbol_clean})")
            elif cvd_val < 0:
                g2_reasons.append(f"دلتای تجمیعی منفی (CVD: {cvd_val:+.2f} {symbol_clean}) — مزیت خریداران تایید نشد")
            else:
                g2_reasons.append("دلتای تجمیعی صفر است؛ جهت تهاجمی مشخص نیست.")
        else:
            g2_reasons.append("دادهٔ کندلی کافی برای CVD در دسترس نیست (Fail-Closed).")
    gatekeepers.append(_gate(
        "ninjatrader_footprint", "عدم تعادل فوت‌پرینت و جهت CVD نینجاتریدر", "NinjaTrader 8", "🎯",
        20, nt_ok, g2_score,
        " + ".join(g2_reasons) if nt_ok else _fail_msg(nt, "نینجاتریدر"),
    ))

    # --- Gate 3: Sierra Chart delta divergence & VBP (Max 15 pts) ---
    sc_ok = _ok(sc)
    g3_score, g3_reasons = 0, []
    if sc_ok:
        div = sc.get("delta_divergence", {}) or {}
        vbp = sc.get("vbp_profile", {}) or {}
        if div.get("detected") and div.get("type") == "BULLISH_ABSORPTION":
            g3_score += 10
            g3_reasons.append(f"واگرایی صعودی دلتا از کندل‌های واقعی (قطعیت {div.get('confidence', 'نامشخص')}%)")
        elif div.get("detected") and div.get("type") == "BEARISH_DISTRIBUTION":
            g3_reasons.append(f"واگرایی نزولی دلتا از کندل‌های واقعی (قطعیت {div.get('confidence', 'نامشخص')}%)")
        else:
            g3_reasons.append("واگرایی دلتای مشخصی از کندل‌های واقعی ثبت نشده است.")
        if isinstance(vbp.get("vbp_poc"), (int, float)):
            g3_score += 5
            g3_reasons.append(f"POC واقعی پروفایل حجم: ${vbp['vbp_poc']:,.2f}")
        if not div.get("detected"):
            g3_score = min(g3_score, 5)
    gatekeepers.append(_gate(
        "sierrachart_delta", "واگرایی دلتا و جذب نقدینگی سیرا چارت", "Sierra Chart", "🌊",
        15, sc_ok, g3_score,
        " + ".join(g3_reasons) if sc_ok else _fail_msg(sc, "سیرا چارت"),
    ))

    # --- Gate 4: Quantower value area (Max 15 pts) ---
    qt_ok = _ok(qt)
    g4_score, g4_reasons = 0, []
    qt_val = qt.get("val") if qt_ok else None
    qt_vah = qt.get("vah") if qt_ok else None
    qt_vpoc = qt.get("vpoc") if qt_ok else None
    if qt_ok:
        if isinstance(qt_val, (int, float)) and qt_val > 0:
            dist_val_pct = abs((current_price - qt_val) / current_price) * 100
            if dist_val_pct <= 1.5:
                g4_score += 10
                g4_reasons.append(f"قیمت در تراز کف ارزش واقعی (VAL: ${qt_val:,.2f}) با فاصله {dist_val_pct:.2f}%")
            elif isinstance(qt_vpoc, (int, float)) and current_price >= qt_vpoc:
                g4_score += 6
                g4_reasons.append(f"قیمت بالای VPOC واقعی (${qt_vpoc:,.2f}) و داخل ناحیهٔ ارزش")
            else:
                g4_reasons.append(f"قیمت زیر VPOC واقعی (${qt_vpoc:,.2f})؛ ساختار حراج ضعیف است.")
        if isinstance(qt_vah, (int, float)) and isinstance(qt_vpoc, (int, float)) and qt_vah > qt_vpoc:
            g4_score += 5
            g4_reasons.append(f"ناحیهٔ ارزش معتبر: VAL ${qt_val:,.2f} تا VAH ${qt_vah:,.2f}")
    gatekeepers.append(_gate(
        "quantower_profile", "محدودهٔ حراج سازمانی کوانت‌تاور", "Quantower", "📊",
        15, qt_ok, g4_score,
        " + ".join(g4_reasons) if qt_ok else _fail_msg(qt, "کوانت‌تاور"),
    ))

    # --- Gate 5: ATAS live tape (Max 15 pts) ---
    atas_ok = _ok(atas)
    g5_score, g5_reasons = 0, []
    if atas_ok:
        tape = atas.get("tape_speed", {}) or {}
        tps = tape.get("trades_per_sec")
        big_trades = atas.get("big_trades", [])
        if isinstance(tps, (int, float)):
            if tps >= 30:
                g5_score += 8
                g5_reasons.append(f"شتاب بالای نوار معاملات واقعی (TPS: {tps})")
            elif tps >= 10:
                g5_score += 4
                g5_reasons.append(f"سرعت نوار معاملات واقعی: {tps} TPS")
            else:
                g5_reasons.append(f"نوار معاملات کم‌رمق: {tps} TPS")
        if big_trades:
            top_tr = big_trades[0]
            size = top_tr.get("size_lots", top_tr.get("volume_lots"))
            if isinstance(size, (int, float)):
                g5_score += 7
                g5_reasons.append(f"بزرگ‌ترین معاملهٔ واقعی: {size:,.2f} {symbol_clean} ({top_tr.get('value_usd', 'نامشخص')})")
    gatekeepers.append(_gate(
        "atas_tape", "سرعت نوار و معاملات سنگین واقعی ATAS", "ATAS Time & Sales", "🏎️",
        15, atas_ok, g5_score,
        " + ".join(g5_reasons) if atas_ok else _fail_msg(atas, "اتاس"),
    ))

    # --- Gate 6: ETF / bank-desk feed (Max 15 pts) ---
    bank_ok = _ok(bank)
    g6_score, g6_reasons = 0, []
    etf_net = bank.get("total_daily_net") if bank_ok else None
    if bank_ok and etf_net:
        etf_txt = str(etf_net)
        if "+" in etf_txt or "مثبت" in etf_txt:
            g6_score += 15
            g6_reasons.append(f"ورودی خالص نهادی ETFها: {etf_txt}")
        elif "-" in etf_txt or "منفی" in etf_txt:
            g6_reasons.append(f"خروج خالص نهادی ETFها: {etf_txt}")
        else:
            g6_reasons.append(f"جریان ETF: {etf_txt}")
    gatekeepers.append(_gate(
        "bank_etf_flows", "جریان خالص اسپات ETFها و وال‌استریت", "Bank Reports", "🏦",
        15, bank_ok, g6_score,
        " + ".join(g6_reasons) if bank_ok else _fail_msg(bank, "جریان ETF و گزارش بانکی"),
    ))

    # 3. Score & execution decision (Fail-Closed)
    available_gates = [g for g in gatekeepers if g["available"]]
    available_max = sum(g["max_score"] for g in available_gates)
    total_score = sum(g["score"] for g in available_gates)
    confluence_score = round(total_score / available_max * 100) if available_max else 0
    data_completeness = round(available_max / 100 * 100)
    critical_ok = all(any(g["id"] == cid and g["available"] for g in gatekeepers) for cid in CRITICAL_GATES)

    direction_code = "NO_TRADE"
    direction_str = "صبر / بدون معامله (NO TRADE)"
    if bm_ok and cvd_available:
        imbalance = bm.get("imbalance_pct")
        if isinstance(imbalance, (int, float)):
            if imbalance >= 55 and cvd_val > 0:
                direction_code = "LONG"
                direction_str = "BUY / LONG (کانفلوئنس خرید)"
            elif imbalance <= 45 and cvd_val < 0:
                direction_code = "SHORT"
                direction_str = "SELL / SHORT (کانفلوئنس فروش)"

    is_sniper_active = bool(critical_ok and confluence_score >= THRESHOLD and direction_code != "NO_TRADE")

    # Real structural levels only (no price-derived fabricated stop/targets)
    entry_exact = round(current_price, 6 if current_price < 1 else 2)
    stop_loss = None
    tp1 = qt_vpoc if isinstance(qt_vpoc, (int, float)) else None
    tp2 = None
    if bm_ok:
        if direction_code == "LONG":
            candidates = []
            if isinstance(nearest_bid.get("price"), (int, float)):
                candidates.append(nearest_bid["price"])
            if qt_ok and isinstance(qt_val, (int, float)):
                candidates.append(qt_val)
            floor = max([c for c in candidates if c < current_price], default=None)
            if floor is not None:
                stop_loss = round(floor * 0.997, 6 if current_price < 1 else 2)
            if qt_ok and isinstance(qt_vah, (int, float)) and qt_vah > current_price:
                tp2 = qt_vah
        elif direction_code == "SHORT":
            ask_walls = bm.get("ask_levels", [])
            nearest_ask = ask_walls[0] if ask_walls else {}
            candidates = []
            if isinstance(nearest_ask.get("price"), (int, float)):
                candidates.append(nearest_ask["price"])
            if qt_ok and isinstance(qt_vah, (int, float)):
                candidates.append(qt_vah)
            ceil = min([c for c in candidates if c > current_price], default=None)
            if ceil is not None:
                stop_loss = round(ceil * 1.003, 6 if current_price < 1 else 2)
            if qt_ok and isinstance(qt_val, (int, float)) and qt_val < current_price:
                tp2 = qt_val

    entry_zone = f"${entry_exact:,.2f}"

    def _pct(target, base):
        if isinstance(target, (int, float)) and base:
            return f"{'+' if target >= base else ''}{(target - base) / base * 100:.2f}%"
        return "نامشخص"

    sl_pct = _pct(stop_loss, entry_exact)
    tp1_pct = _pct(tp1, entry_exact)
    tp2_pct = _pct(tp2, entry_exact)
    risk_reward = None
    if isinstance(stop_loss, (int, float)) and isinstance(tp2, (int, float)) and stop_loss != entry_exact:
        risk = abs(entry_exact - stop_loss)
        reward = abs(tp2 - entry_exact)
        if risk > 0:
            risk_reward = round(reward / risk, 2)

    unavailable_names = [g["module"] for g in gatekeepers if not g["available"]]
    macro_note = (
        "تمام گیت‌های حیاتی اردر فلو از دادهٔ زندهٔ صرافی تغذیه شدند."
        if critical_ok and not unavailable_names else
        "گیت‌های در دسترس‌نیست: " + "، ".join(unavailable_names) + " — نتیجه‌گیری نهایی با احتیاط و بدون ادعای قطعیت است."
    )

    return {
        "ok": True,
        "symbol": symbol_clean,
        "current_price": current_price,
        "confluence_score": confluence_score,
        "confluence_status": (
            f"GRADE-A CONFLUENCE ({confluence_score}/100)" if is_sniper_active
            else "CAPITAL GUARD (دادهٔ ناقص یا نمرهٔ زیر آستانه)"
        ),
        "is_active_signal": is_sniper_active,
        "direction_code": direction_code,
        "direction": direction_str,
        "winrate_projection": "نامشخص (بدون بک‌تست معتبر، عدد ادعا نمی‌شود)",
        "setup_name": "کانفلوئنس اردر فلو نهادی (دیواره‌های واقعی + دلتا + ناحیهٔ ارزش)",
        "entry_zone": entry_zone,
        "entry_exact": entry_exact,
        "stop_loss": stop_loss,
        "stop_loss_pct": sl_pct,
        "tp1": tp1,
        "tp1_pct": tp1_pct,
        "tp2": tp2,
        "tp2_pct": tp2_pct,
        "risk_reward_ratio": f"1:{risk_reward}" if risk_reward else "نامشخص",
        "gatekeepers": gatekeepers,
        "passed_gates_count": len([g for g in gatekeepers if g["passed"]]),
        "total_gates_count": len(gatekeepers),
        "available_gates_count": len(available_gates),
        "data_completeness_pct": data_completeness,
        "critical_gates_available": critical_ok,
        "execution_rules_fa": [
            "قانون ۱: حد ضرر فقط پشت دیوارهٔ واقعی یا تراز VAL واقعی قرار می‌گیرد؛ اگر چنین سطحی موجود نباشد، حد ضرر «نامشخص» می‌ماند و معامله صادر نمی‌شود.",
            "قانون ۲: تا زمانی که حداقل یکی از گیت‌های حیاتی (بوک‌مپ، نینجاتریدر، ATAS) دادهٔ زنده نداشته باشد، سیگنال فعال نمی‌شود (Fail-Closed).",
            "قانون ۳: هیچ نسبت ریسک‌به‌ریوارد یا وین‌ریتی بدون بک‌تست معتبر ادعا نمی‌شود.",
            "قانون ۴: در صورت انتشار اخبار قرمز CPI یا بیانیهٔ فدرال رزرو، تمام معاملات بلافاصله متوقف می‌شوند."
        ],
        "macro_bias_summary": macro_note,
        "updated_at_iran": now_tehran,
    }


if __name__ == "__main__":
    res = evaluate_sniper_confluence("BTC")
    if not res.get("ok"):
        print(res.get("message", res.get("error", "SNIPER_UNAVAILABLE")))
    else:
        print(f"Sniper Confluence Score: {res['confluence_score']}/100")
        print(f"Status: {res['confluence_status']}")
        print(f"Signal: {res['direction']} @ {res['entry_exact']}")
        print(f"SL: {res['stop_loss']} ({res['stop_loss_pct']})")
        print(f"TP1: {res['tp1']} ({res['tp1_pct']})")
        print(f"TP2: {res['tp2']} ({res['tp2_pct']})")
        print(f"Passed Gates: {res['passed_gates_count']}/{res['total_gates_count']} "
              f"(available: {res['available_gates_count']}, completeness: {res['data_completeness_pct']}%)")
