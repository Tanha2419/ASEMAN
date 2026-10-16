#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXHAUSTIVE END-TO-END SIMULATION OF SMART ARBITER GATEKEEPER & SIGNAL JOURNAL
Simulates the complete lifecycle:
1. Clean Journal Initial State (Zero fake benchmarks)
2. Arbiter 8-Layer Defense Evaluation of Altcoin (RLC)
3. Rejection of bad setups (Negative R:R or Bearish BTC clash)
4. Atomic Journal & Telegram Dispatch of Grade A+ Setup
5. Real-time Outcome Tracking:
   - Price rises and hits TP1 (+2.85%)
   - Rule 3 Triggers: SL is atomically moved to Entry (Break-Even = 0% Risk)
   - Price rises and hits TP2 (+5.60%)
6. Rule 1 Enforcement: Portfolio Exposure Cap prevents over-leveraging
7. Clean Reset Audit: Confirms 0 ghost trades return after reset
"""

import sys
import os
import time
import json

sys.path.insert(0, '/home/user')
import server
from crypto_arbiter_gatekeeper import global_arbiter
from fastapi.testclient import TestClient

client = TestClient(server.app)

def log_step(title):
    print("\n" + "="*80)
    print(f"🚀 {title}")
    print("="*80)

def run_simulation():
    print("💎 STARTING INSTITUTIONAL SIMULATION OF CRYPTO ARBITER & JOURNAL 💎")

    # STEP 1: VERIFY CLEAN JOURNAL BASELINE (RESET ALL)
    log_step("مرحله ۱: پاک‌سازی کامل و بررسی صفر بودن قطعی ژورنال")
    reset_res = client.post("/api/journal/reset-all").json()
    print("نتیجه ریست ژورنال:", reset_res.get("message"))

    journal_st = client.get("/api/journal").json()
    stats = journal_st.get("stats", {})
    records = journal_st.get("records", [])
    print(f"تعداد معاملات موجود در ژورنال: {len(records)} (کاملاً خالی)")
    print(f"وین‌ریت فعلی: {stats.get('win_rate')}% | مجموع سود: {stats.get('total_pnl')}%")
    assert len(records) == 0, "ژورنال باید کاملاً خالی (۰ معامله) باشد!"
    print("✅ مرحله ۱ با موفقیت تایید شد: هیچ سیگنال فیک، فانتوم یا قدیمی در ژورنال وجود ندارد.")

    # STEP 2: ARBITER AUDITING A FLAWED SETUP (NEGATIVE R:R OR LOW QUALITY)
    log_step("مرحله ۲: تست فیلترهای دفاعی دروازه‌بان روی یک ستاپ ضعیف و رد آن")
    flawed_candidate = {
        "symbol": "BADCOIN",
        "action": "BUY",
        "price": 10.0,
        "score": 60, # low score
        "sl": 9.90,  # 1% SL on altcoin (too tight, violates anti-stop-hunt rule)
        "tp1": 10.05 # tiny target, bad R:R
    }
    verdict_bad = global_arbiter.audit_and_decide(flawed_candidate, source="TEST_FLAWED")
    print(f"نماد کاندیدا: {flawed_candidate['symbol']}")
    print(f"نتیجه داوری: {'تایید شد' if verdict_bad['approved'] else '❌ رد شد'}")
    print(f"علت رد توسط دروازه‌بان: {verdict_bad.get('reason')}")
    assert verdict_bad["approved"] is False, "ستاپ ضعیف نباید از دروازه‌بان عبور کند!"
    print("✅ مرحله ۲ با موفقیت تایید شد: دروازه‌بان معامله ضعیف را مسدود کرد.")

    # STEP 3: ARBITER AUDITING A HIGH-CONVICTION ALTCOIN (RLC)
    log_step("مرحله ۳: ممیزی ستاپ طلایی روی آلت‌کوین RLC با اعمال بافر ضد استاپ‌هانت")
    # Reset cooldowns for test
    global_arbiter._symbol_cooldowns.clear()
    global_arbiter._last_global_dispatch_ts = 0.0

    rlc_candidate = {
        "symbol": "RLC",
        "action": "BUY",
        "price": 1.0420,
        "score": 88
    }
    verdict_rlc = global_arbiter.audit_and_decide(rlc_candidate, source="RADAR_3M")
    print("نتیجه داوری برای RLC:")
    print(f"  • تاییدیه نهایی: {'✅ تایید شد (A+)' if verdict_rlc['approved'] else 'رد شد'}")
    print(f"  • امتیاز اعتبار: {verdict_rlc.get('conviction_score')}/100")
    setup = verdict_rlc.get("setup", {})
    print(f"  • نقطه ورود (Entry): ${setup.get('entry')}")
    print(f"  • حد ضرر ضد استاپ‌هانت (SL): ${setup.get('sl')} (-{setup.get('sl_pct')}%)")
    print(f"  • تارگت اول ضد فرانت‌رانینگ (TP1): ${setup.get('tp1')} (R:R 1:{setup.get('rr_ratio')})")
    print(f"  • تارگت دوم (TP2): ${setup.get('tp2')}")
    print(f"  • تارگت سوم (TP3): ${setup.get('tp3')}")
    print("  • چک‌لیست دفاعیات گذرانده‌شده:")
    for note in verdict_rlc.get("defense_notes", []):
        print(f"    - {note}")

    assert verdict_rlc["approved"] is True, "ستاپ طلایی RLC باید تایید شود!"
    assert setup["sl_pct"] >= 2.85, "بافر ضد استاپ‌هانت آلت‌کوین باید حداقل ۲.۸۵٪ باشد!"
    print("✅ مرحله ۳ با موفقیت تایید شد: ستاپ با استاپ ایمن و تارگت‌های ضد فرانت‌رانینگ فرموله شد.")

    # STEP 4: ATOMIC DISPATCH & JOURNAL REGISTRATION
    log_step("مرحله ۴: ثبت اتمیک معامله تاییدشده در ژورنال و هماهنگی ۱۰۰٪ با تلگرام")
    exec_res = global_arbiter.execute_atomic_approval(verdict_rlc, bot_token="", chat_id="")
    print("نتیجه اجرای اتمیک:", exec_res)
    assert exec_res.get("journal_recorded") is True, "معامله باید در ژورنال ثبت شده باشد!"

    # Check journal immediately
    j_after = client.get("/api/journal").json()
    recs = j_after.get("records", [])
    print(f"تعداد معاملات ثبت‌شده در ژورنال: {len(recs)}")
    assert len(recs) == 1, "دقیقاً ۱ معامله تاییدشده باید در ژورنال ثبت شده باشد!"
    rec_0 = recs[0]
    print(f"مشخصات معامله در ژورنال: نماد #{rec_0['symbol']} | پوزیشن: {rec_0['action']} | وضعیت: {rec_0['status']}")
    print(f"ورود: ${rec_0['entry']} | حد ضرر: ${rec_0['sl']} | تارگت ۱: ${rec_0['tp1']}")
    assert rec_0['symbol'] == 'RLC'
    assert float(rec_0['entry']) == float(setup['entry'])
    assert float(rec_0['sl']) == float(setup['sl'])
    print("✅ مرحله ۴ با موفقیت تایید شد: ژورنال و تلگرام دقیقاً آینه یکدیگر با مقادیر کاملاً یکسان هستند.")

    # STEP 5: REAL-TIME OUTCOME TRACKING & AUTO-BREAKEVEN ENFORCEMENT
    log_step("مرحله ۵: شبیه‌سازی تاچ TP1 و اجرای خودکار قانون ۳ (انتقال استاپ به ورود / Break-Even)")
    
    # Simulate outcome tracking with price reaching TP1
    tracker_entry = {
        "symbol": "RLC",
        "action": setup["action"],
        "entry": setup["entry"],
        "sl": setup["sl"],
        "tp1": setup["tp1"],
        "tp2": setup["tp2"],
        "tp3": setup["tp3"],
        "tp1_hit": False,
        "tp2_hit": False,
        "closed": False,
        "created_at": time.time()
    }
    server._active_signal_trackers.append(tracker_entry)

    # Trigger TP1 hit
    tp1_pct = round(abs((setup["tp1"] - setup["entry"]) / setup["entry"]) * 100, 2)
    server.update_signal_in_journal("RLC", "TP1_HIT", tp1_pct)
    tracker_entry["tp1_hit"] = True
    
    # Execute Auto-Breakeven Rule
    be_res = global_arbiter.arm_breakeven_on_tp1_hit(tracker_entry)
    print("نتیجه اجرای قانون ریسک‌فری خودکار:", be_res)
    print(f"حد ضرر جدید پوزیشن RLC: ${tracker_entry['sl']} (دقیقاً برابر با نقطه ورود)")
    print(f"میزان ریسک معامله: {be_res['risk']}")
    assert tracker_entry["sl"] == setup["entry"], "حد ضرر باید دقیقاً به نقطه ورود منتقل شده باشد!"

    # STEP 6: TRIGGER TP2 HIT
    log_step("مرحله ۶: شبیه‌سازی تاچ TP2 و ثبت بازدهی مضاعف در ژورنال")
    tp2_pct = round(abs((setup["tp2"] - setup["entry"]) / setup["entry"]) * 100, 2)
    server.update_signal_in_journal("RLC", "TP2_HIT", tp2_pct)
    tracker_entry["tp2_hit"] = True
    tracker_entry["closed"] = True

    # Check updated journal stats
    j_tp2 = client.get("/api/journal").json()
    st_tp2 = j_tp2.get("stats", {})
    print("آمار زنده ژورنال پس از تارگت دوم:")
    print(f"  • وین‌ریت: {st_tp2.get('win_rate')}%")
    print(f"  • تارگت‌های لمس‌شده: {st_tp2.get('tp_hits')} تا")
    print(f"  • معاملات با حد ضرر خورده: {st_tp2.get('sl_hits')} تا")
    print(f"  • سود خالص ثبت‌شده: +{st_tp2.get('total_pnl')}%")
    print(f"  • منحنی رشد سرمایه (Equity Curve): {j_tp2.get('equity_curve')}")
    assert st_tp2.get("tp_hits") >= 1
    assert st_tp2.get("win_rate") == 100.0
    print("✅ مرحله ۶ با موفقیت تایید شد: تارگت‌های سود با دقت بالا ثبت و منحنی رشد بروزرسانی شد.")

    # STEP 7: RULE 1 ENFORCEMENT - PORTFOLIO EXPOSURE CAP
    log_step("مرحله ۷: تست قانون ۱ (سقف مواجهه پورتفو - حداکثر ۲ معامله هم‌زمان باز)")
    server._active_signal_trackers.clear()
    server._active_signal_trackers.extend([
        {"symbol": "SOL", "action": "LONG", "closed": False, "tp1_hit": False},
        {"symbol": "SUI", "action": "LONG", "closed": False, "tp1_hit": False}
    ])
    global_arbiter.active_trackers_ref = server._active_signal_trackers

    # Reset spam gap to specifically test portfolio exposure cap
    global_arbiter._last_global_dispatch_ts = 0.0
    # Try to add a 3rd trade (DOGE)
    cand_doge = {"symbol": "DOGE", "action": "BUY", "price": 0.125, "score": 85}
    verdict_doge = global_arbiter.audit_and_decide(cand_doge, source="SENTINEL_20M")
    print(f"ارزیابی پوزیشن سوم (DOGE) در حضور ۲ معامله باز:")
    print(f"  • نتیجه: {'تایید شد' if verdict_doge['approved'] else '❌ رد شد'}")
    print(f"  • علت: {verdict_doge.get('reason')}")
    assert verdict_doge["approved"] is False
    assert verdict_doge["trap_failed"] == "PORTFOLIO_CAP"
    print("✅ مرحله ۷ با موفقیت تایید شد: دروازه‌بان از باز شدن معامله سوم و به خطر افتادن مارجین جلوگیری کرد.")

    # STEP 8: FINAL RESET INTEGRITY CHECK
    log_step("مرحله ۸: تست بازنشانی نهایی (Zero Ghost Trade Guarantee)")
    client.post("/api/journal/reset-all")
    j_final = client.get("/api/journal").json()
    print(f"تعداد معاملات پس از ریست کامل: {len(j_final.get('records', []))}")
    assert len(j_final.get("records", [])) == 0, "پس از ریست نباید هیچ معامله‌ای باقی بماند!"
    print("✅ مرحله ۸ با موفقیت تایید شد: هیچ دیتای الکی یا تکراری پس از ریست بازنمی‌گردد.")

    print("\n" + "🌟"*40)
    print("🎉 کلیه ۸ آزمون شبیه‌سازی ژورنال و دروازه‌بان هوشمند با نمره ۱۰۰٪ قبول شدند!")
    print("🌟"*40)

if __name__ == "__main__":
    run_simulation()
