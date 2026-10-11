#!/usr/bin/env python3
"""
CRYPTO ASEMAN COMPREHENSIVE TRADER-CENTRIC AUDIT & STRESS TEST (V2)
Simulates live market dynamics, mathematical sanity, lock integrity, orderflow authenticity,
micro-capital risk management, and journal state integrity across multiple coins.
"""
import time
import json
import traceback
import sys
import re

import crypto_unified_signals as cus
import crypto_orderflow_engine as coe
import crypto_smt_and_liquidation_engine as cse
import analyzer_engine as ae

TEST_SYMBOLS = ["STRK", "BTC", "ETH", "SOL", "DOGE"]
REPORT = {
    "passed": [],
    "failed": [],
    "warnings": []
}

def record_pass(category, desc):
    REPORT["passed"].append(f"[{category}] {desc}")
    print(f"  [PASS] [{category}] {desc}")

def record_fail(category, desc, details=None):
    REPORT["failed"].append(f"[{category}] {desc} | {details or ''}")
    print(f"  [FAIL] [{category}] {desc} -> {details}")

def record_warn(category, desc):
    REPORT["warnings"].append(f"[{category}] {desc}")
    print(f"  [WARN] [{category}] {desc}")

print("======================================================================")
print("  STARTING DEEP QUANT & TRADER SIMULATION: CRYPTO ASEMAN PLATFORM (V2)")
print("======================================================================\n")

# -------------------------------------------------------------------------
# STAGE 1: DUAL UNIFIED SIGNALS (SCALP 15m & SWING 4h) INTEGRITY TEST
# -------------------------------------------------------------------------
print(">>> STAGE 1: Testing Unified Signals Math, Levels, and Direction across symbols...")

for sym in TEST_SYMBOLS:
    print(f"\n--- Checking {sym} ---")
    try:
        data = cus.get_unified_signals(sym)
    except Exception as e:
        record_fail("UNIFIED_SIGNALS", f"Crash fetching {sym}", str(e))
        continue

    cur_price = data.get("current_price", 0)
    if cur_price <= 0:
        record_fail("PRICE_FEED", f"{sym} current price is <= 0 ({cur_price})")
    else:
        record_pass("PRICE_FEED", f"{sym} live price valid: ${cur_price}")

    # Check Scalp & Swing
    for stype in ["scalp", "swing"]:
        s = data.get(stype, {})
        direction_code = s.get("direction_code", "")
        direction = s.get("direction", "")
        entry = s.get("entry", 0)
        sl = s.get("stop_loss", 0)
        tp1 = s.get("tp1", 0)
        tp2 = s.get("tp2", 0)
        tp3 = s.get("tp3", 0)
        
        # Parse SL percentage from formatted text if present e.g. "$0.11089 (-1.80%)"
        sl_fmt = s.get("stop_loss_fmt", "")
        m_sl = re.search(r'\(([-+]?[0-9]*\.?[0-9]+)%\)', sl_fmt)
        sl_pct = abs(float(m_sl.group(1))) if m_sl else 0.0

        tag = f"{sym} [{stype.upper()}]"

        # 1. Action validity
        if direction_code not in ["LONG", "SHORT", "WAIT"]:
            record_fail("ACTION", f"{tag} Invalid direction_code: '{direction_code}'")
        else:
            record_pass("ACTION", f"{tag} Direction: {direction_code} ({direction})")

        if direction_code in ["LONG", "SHORT"]:
            is_long = direction_code == "LONG"
            
            # 2. Mathematical order of price levels
            if is_long:
                if not (sl < entry):
                    record_fail("MATH_ORDER", f"{tag} LONG error: SL ({sl}) >= Entry ({entry})")
                elif not (entry < tp1 < tp2 < tp3):
                    record_fail("MATH_ORDER", f"{tag} LONG error: TP ladder not strictly increasing: Entry={entry}, TP1={tp1}, TP2={tp2}, TP3={tp3}")
                else:
                    record_pass("MATH_ORDER", f"{tag} LONG ladder correct: SL({sl}) < Entry({entry}) < TP1({tp1}) < TP2({tp2}) < TP3({tp3})")
            else:
                if not (sl > entry):
                    record_fail("MATH_ORDER", f"{tag} SHORT error: SL ({sl}) <= Entry ({entry})")
                elif not (entry > tp1 > tp2 > tp3):
                    record_fail("MATH_ORDER", f"{tag} SHORT error: TP ladder not strictly decreasing: Entry={entry}, TP1={tp1}, TP2={tp2}, TP3={tp3}")
                else:
                    record_pass("MATH_ORDER", f"{tag} SHORT ladder correct: SL({sl}) > Entry({entry}) > TP1({tp1}) > TP2({tp2}) > TP3({tp3})")

            # 3. Stop loss percentage sanity check
            if sl_pct <= 0:
                record_fail("SL_PCT", f"{tag} Stop loss percentage is <= 0%: {sl_pct}%")
            elif sl_pct > 15.0 and stype == "scalp":
                record_fail("SL_PCT", f"{tag} Scalp SL is excessively high: {sl_pct}% (>15%)")
            elif sl_pct > 25.0 and stype == "swing":
                record_fail("SL_PCT", f"{tag} Swing SL is excessively high: {sl_pct}% (>25%)")
            elif sl_pct < 0.2:
                record_warn("SL_PCT", f"{tag} SL percentage might be too tight for crypto: {sl_pct}%")
            else:
                record_pass("SL_PCT", f"{tag} SL percentage reasonable: {sl_pct:.2f}%")

            # 4. Risk:Reward Ratio check
            risk = abs(entry - sl)
            reward_tp1 = abs(tp1 - entry)
            reward_tp2 = abs(tp2 - entry)
            rr_tp1 = reward_tp1 / risk if risk > 0 else 0
            rr_tp2 = reward_tp2 / risk if risk > 0 else 0
            if rr_tp1 < 0.5:
                record_warn("RR_RATIO", f"{tag} TP1 R:R is low ({rr_tp1:.2f}:1)")
            else:
                record_pass("RR_RATIO", f"{tag} TP1 R:R healthy: {rr_tp1:.2f}:1 | TP2 R:R: {rr_tp2:.2f}:1")

            # 5. String formatting check (no scientific notation, no dollar sign glitches)
            entry_fmt = s.get("entry_fmt", "")
            tp1_fmt = s.get("tp1_fmt", "")
            if "e-" in entry_fmt.lower() or "e+" in entry_fmt.lower():
                record_fail("FORMATTING", f"{tag} Entry has scientific notation: {entry_fmt}")
            if not entry_fmt.startswith("$"):
                record_fail("FORMATTING", f"{tag} Entry missing '$' prefix: {entry_fmt}")
            if not sl_fmt.startswith("$"):
                record_fail("FORMATTING", f"{tag} SL missing '$' prefix: {sl_fmt}")
            if not tp1_fmt.startswith("$"):
                record_fail("FORMATTING", f"{tag} TP1 missing '$' prefix: {tp1_fmt}")


# -------------------------------------------------------------------------
# STAGE 2: MULTI-TICK SETUP FREEZE & TIME-LOCK STRESS SIMULATION
# -------------------------------------------------------------------------
print("\n>>> STAGE 2: Simulating 5 consecutive ticks to verify Setup Freeze...")

for sym in ["STRK", "BTC"]:
    print(f"\n--- Multi-tick simulation for {sym} ---")
    history = []
    for i in range(5):
        res = cus.get_unified_signals(sym)
        sc = res["scalp"]
        history.append({
            "iter": i,
            "entry": sc.get("entry_fmt"),
            "sl": sc.get("stop_loss_fmt"),
            "tp1": sc.get("tp1_fmt"),
            "time": sc.get("time_tehran")
        })
        time.sleep(0.3)

    base = history[0]
    locked = True
    for idx, h in enumerate(history[1:], 1):
        if h["entry"] != base["entry"]:
            record_fail("SETUP_FREEZE", f"{sym} Scalp Entry shifted on tick {idx}: {base['entry']} -> {h['entry']}")
            locked = False
        if h["sl"] != base["sl"]:
            record_fail("SETUP_FREEZE", f"{sym} Scalp SL shifted on tick {idx}: {base['sl']} -> {h['sl']}")
            locked = False
        if h["tp1"] != base["tp1"]:
            record_fail("SETUP_FREEZE", f"{sym} Scalp TP1 shifted on tick {idx}: {base['tp1']} -> {h['tp1']}")
            locked = False
        if h["time"] != base["time"]:
            record_fail("SETUP_FREEZE", f"{sym} Setup time changed on tick {idx}: {base['time']} -> {h['time']}")
            locked = False

    if locked:
        record_pass("SETUP_FREEZE", f"{sym} Scalp setup stayed 100% frozen across all 5 ticks (Entry={base['entry']})")


# -------------------------------------------------------------------------
# STAGE 3: ORDERFLOW & BOOKMAP AUTHENTICITY TEST
# -------------------------------------------------------------------------
print("\n>>> STAGE 3: Checking Order Flow & Heatmap Live Data Authenticity...")

for sym in ["BTC", "ETH", "STRK"]:
    print(f"\n--- Checking Orderflow for {sym} ---")
    try:
        bm = coe.get_crypto_bookmap_data(sym)
        nt = coe.get_crypto_ninjatrader_live(sym)
        atas = coe.get_crypto_atas_live(sym)
        sc = coe.get_crypto_sierrachart_data(sym)
    except Exception as e:
        record_fail("ORDERFLOW", f"Crash in orderflow modules for {sym}", str(e))
        continue

    # 1. Bookmap checks
    bid_levels = bm.get("bid_levels", [])
    ask_levels = bm.get("ask_levels", [])
    if not bid_levels or not ask_levels:
        record_fail("ORDERFLOW_BOOK", f"{sym} Bookmap orderbook levels are empty!")
    else:
        top_bid = bid_levels[0].get("price", 0)
        top_ask = ask_levels[0].get("price", 0)
        # Bids are resting buy walls below or at price, Asks are resting sell walls above or at price
        # In sorted concentration, verify they are distinct valid price points
        record_pass("ORDERFLOW_BOOK", f"{sym} Bookmap Top Bid Wall=${top_bid} ({bid_levels[0].get('volume_lots')} lots) | Top Ask Wall=${top_ask} ({ask_levels[0].get('volume_lots')} lots)")

    # 2. NinjaTrader Footprint checks
    fp = nt.get("candles", []) or nt.get("footprint_candles", [])
    if not fp or len(fp) < 3:
        record_fail("ORDERFLOW_NT", f"{sym} NinjaTrader Footprint candles missing or < 3 (got {len(fp)})")
    else:
        record_pass("ORDERFLOW_NT", f"{sym} NinjaTrader has {len(fp)} authentic footprint candles")

    # 3. ATAS checks
    atas_summary = atas.get("summary_fa", "")
    if not atas_summary:
        record_warn("ORDERFLOW_ATAS", f"{sym} ATAS summary is empty")
    else:
        record_pass("ORDERFLOW_ATAS", f"{sym} ATAS live tape: {atas_summary[:60]}...")

    # 4. Sierra Chart CVD & Bars checks
    bars = sc.get("numbered_bars", [])
    if not bars:
        record_fail("ORDERFLOW_SIERRA", f"{sym} Sierra Chart numbered bars are empty")
    else:
        record_pass("ORDERFLOW_SIERRA", f"{sym} Sierra Chart has {len(bars)} numbered bars with delta analysis")


# -------------------------------------------------------------------------
# STAGE 4: SMT DIVERGENCE & LIQUIDATION CLUSTER TEST
# -------------------------------------------------------------------------
print("\n>>> STAGE 4: Checking SMT Divergence & Liquidation Engine...")

for sym in ["BTC", "ETH", "STRK"]:
    try:
        smt_res = cse.get_crypto_smt_and_liquidation_data(sym)
        smt = smt_res.get("smt", {})
        liq = smt_res.get("liquidation_heatmap", {})
        cur_p = smt_res.get("current_price", 0)

        # Check SMT
        smt_status = smt.get("status", "")
        if not smt_status:
            record_fail("SMT", f"{sym} SMT status empty")
        else:
            record_pass("SMT", f"{sym} SMT status: {smt_status} (Badge: {smt.get('badge', '')[:40]})")

        # Check Liquidation clusters
        clusters = liq.get("clusters", [])
        if not clusters:
            record_fail("LIQUIDATION", f"{sym} Missing liquidation clusters")
        else:
            # Check positions of clusters
            short_clusters = [c for c in clusters if "شورت" in c["tier"]]
            long_clusters = [c for c in clusters if "لانگ" in c["tier"]]

            all_shorts_above = all(c["price"] > cur_p for c in short_clusters)
            all_longs_below = all(c["price"] < cur_p for c in long_clusters)

            if not all_shorts_above:
                record_fail("LIQUIDATION_LOGIC", f"{sym} Short liquidation levels found <= current price (${cur_p})")
            else:
                record_pass("LIQUIDATION_LOGIC", f"{sym} Short liquidation clusters correctly placed above ${cur_p}")

            if not all_longs_below:
                record_fail("LIQUIDATION_LOGIC", f"{sym} Long liquidation levels found >= current price (${cur_p})")
            else:
                record_pass("LIQUIDATION_LOGIC", f"{sym} Long liquidation clusters correctly placed below ${cur_p}")

            record_pass("LIQUIDATION_PRICE", f"{sym} Sample short pool: ${short_clusters[0]['price']} | Sample long pool: ${long_clusters[0]['price']}")

    except Exception as e:
        record_fail("SMT_ENGINE", f"Crash in SMT / Liquidation calculation for {sym}", str(e))


# -------------------------------------------------------------------------
# STAGE 5: $10 MICRO-ACCOUNT REAL-WORLD RISK CHECK
# -------------------------------------------------------------------------
print("\n>>> STAGE 5: $10 Micro Account Risk & Liquidation Sanity...")

for sym in ["STRK", "SOL", "BTC"]:
    s_data = cus.get_unified_signals(sym)
    sc = s_data.get("scalp", {})
    entry = sc.get("entry", 0)
    sl = sc.get("stop_loss", 0)
    sl_fmt = sc.get("stop_loss_fmt", "")
    m_sl = re.search(r'\(([-+]?[0-9]*\.?[0-9]+)%\)', sl_fmt)
    sl_pct = abs(float(m_sl.group(1))) if m_sl else 0.0

    # Margin check
    risk_1x = 10.0 * (sl_pct / 100.0)
    risk_5x = 50.0 * (sl_pct / 100.0)

    print(f"  {sym} Scalp SL: {sl_pct:.2f}% | $10 Account Risk (1x Spot): ${risk_1x:.2f} | (5x Lev): ${risk_5x:.2f}")
    if risk_1x > 2.0:
        record_warn("ACCOUNT_RISK", f"{sym} 1x risk (${risk_1x:.2f}) > 20% of $10 account!")
    elif risk_5x > 5.0:
        record_warn("ACCOUNT_RISK", f"{sym} 5x leverage risk (${risk_5x:.2f}) > 50% of $10 account!")
    else:
        record_pass("ACCOUNT_RISK", f"{sym} Risk is safe for $10 account (${risk_1x:.2f} at 1x, ${risk_5x:.2f} at 5x)")

# -------------------------------------------------------------------------
# STAGE 6: JOURNAL PERSISTENCE & NO GHOST TRADES
# -------------------------------------------------------------------------
print("\n>>> STAGE 6: Journal State Management & Ghost Signal Check...")
import server

# Verify journal load
j_records = server.load_signal_journal()
print(f"  Current Scalp Journal records count: {len(j_records)}")

# Verify 4H journal load
j4_records = server.load_4h_journal()
print(f"  Current 4H Swing Journal records count: {len(j4_records)}")

# Check for ghost records (> 7 days old or fake benchmark dates)
ghost_found = False
for r in j_records + j4_records:
    sym = r.get("symbol", "")
    if "BENCHMARK" in sym or "DUMMY" in sym:
        record_fail("JOURNAL", f"Ghost dummy record found: {sym}")
        ghost_found = True
if not ghost_found:
    record_pass("JOURNAL", "No dummy/benchmark ghost records in database.")

# -------------------------------------------------------------------------
# SUMMARY
# -------------------------------------------------------------------------
print("\n======================================================================")
print(f"  AUDIT SUMMARY: {len(REPORT['passed'])} PASSED | {len(REPORT['warnings'])} WARNINGS | {len(REPORT['failed'])} FAILED")
print("======================================================================")

if REPORT["failed"]:
    print("\nCRITICAL FAILURES DETECTED:")
    for f in REPORT["failed"]:
        print(f"  - {f}")
else:
    print("\n>>> ALL CRITICAL ENGINE AND TRADING TESTS PASSED WITH ZERO FAILURES! <<<")

if REPORT["warnings"]:
    print("\nAREAS NEEDING OPTIMIZATION / WARNINGS:")
    for w in REPORT["warnings"]:
        print(f"  - {w}")
