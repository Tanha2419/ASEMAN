#!/usr/bin/env python3
"""
CryptoAgent AI - Terminal CLI Client
Supports Smart Money Concepts (SMC), FVG, BSL/SSL Sweeps, HFT Footprint, and Trap Detection.
Usage:
    python3 crypto_agent.py BTC
    python3 crypto_agent.py SOL --mode scalp
    python3 crypto_agent.py ETH --mode smc
"""

import sys
import argparse
from analyzer_engine import CryptoTradingAgent

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

def print_banner():
    banner = f"""
{CYAN}{BOLD}========================================================================
   ⚡ CryptoAgent AI - ایجنت اسمارت‌مانی، نقدینگی و تحلیل کریپتو ⚡
========================================================================{RESET}
"""
    print(banner)

def format_price(val):
    if val is None:
        return "$0.00"
    num = float(val)
    if num >= 1:
        return f"${num:,.4f}"
    else:
        return f"${num:.8f}"

def run_cli():
    parser = argparse.ArgumentParser(description="Crypto Trading AI Agent CLI")
    parser.add_argument("symbol", nargs="?", default="BTC", help="Crypto symbol (e.g. BTC, ETH, SOL, PEPE, SUI)")
    parser.add_argument("--mode", choices=["all", "scalp", "swing", "smc", "inst", "macro", "news", "backtest", "precision", "alpha", "risk", "golden"], default="all", help="Analysis mode")
    parser.add_argument("--telegram", action="store_true", help="Send institutional signal to Telegram")
    parser.add_argument("--lookback", type=int, default=300, help="Backtest lookback candles count (default: 300)")
    parser.add_argument("--balance", type=float, default=5000.0, help="Account balance for Kelly Risk Calculator (default: 5000)")
    parser.add_argument("--risk-pct", type=float, default=1.0, help="Risk percentage per trade (default: 1.0%)")
    args = parser.parse_args()

    print_banner()
    print(f"{YELLOW}⏳ در حال دریافت داده‌های زنده و کالبدشکافی اسمارت‌مانی نماد {args.symbol.upper()}...{RESET}")

    agent = CryptoTradingAgent()
    data = agent.analyze_symbol(args.symbol)

    if not data.get("success"):
        print(f"\n{RED}❌ خطا: {data.get('error')}{RESET}\n")
        return

    sym = data["symbol"]
    price = data["price"]
    ticker = data["ticker"]
    change = ticker.get("price_change_pct", 0)
    change_color = GREEN if change >= 0 else RED
    fng = data.get("market_sentiment", {})
    ob = data.get("orderbook", {})
    scalp = data.get("scalp_setup", {})
    swing = data.get("swing_setup", {})
    smc = data.get("smc", {}).get("15m", {})
    derivatives = data.get("derivatives", {})
    scores_3d = data.get("scores_3d", {})
    vwap_cvd = data.get("vwap_cvd", {})
    matrix = data.get("derivatives_matrix", {})
    inst_table = data.get("inst_table", [])
    macro = data.get("macro", {})
    onchain = data.get("onchain", {})
    news_data = data.get("news", {})
    backtest = data.get("backtest", {})
    correlation = data.get("correlation", {})

    print(f"\n{BOLD}📊 نمای کلی نماد:{RESET} {CYAN}{BOLD}{sym}{RESET}")
    print(f"💰 {BOLD}قیمت لحظه‌ای:{RESET} {BOLD}{format_price(price)}{RESET} | {BOLD}تغییر ۲۴ ساعته:{RESET} {change_color}{change:+.2f}%{RESET}")
    print(f"📈 {BOLD}بالاترین ۲۴ ساعت:{RESET} {format_price(ticker.get('high_24h'))} | {BOLD}پایین‌ترین:{RESET} {format_price(ticker.get('low_24h'))}")
    print(f"🌊 {BOLD}حجم ۲۴ ساعته:{RESET} ${ticker.get('volume_quote', 0)/1_000_000:,.1f}M USD")
    print(f"🧭 {BOLD}شاخص طمع و ترس کل بازار:{RESET} {YELLOW}{fng.get('value')} ({fng.get('classification')}){RESET}")

    # Institutional 3D Scores Summary
    if scores_3d:
        g_col = YELLOW if scores_3d.get("grade") == "A+" else (GREEN if scores_3d.get("grade") == "A" else RED)
        print(f"\n{CYAN}{BOLD}------------------------------------------------------------------------{RESET}")
        print(f"{CYAN}{BOLD}🏛️ تفکیک سه‌گانه امتیازات سازمانی (3-Dimensional Score Separation):{RESET}")
        print(f"  🎯 {BOLD}امتیاز جهت روند و جریان (Direction Score):{RESET} {GREEN}{scores_3d.get('direction_score', 0)}/100{RESET}")
        print(f"  ⏱️ {BOLD}امتیاز زمان‌بندی و نقطه ورود (Entry Score):{RESET} {BLUE}{scores_3d.get('entry_score', 0)}/100{RESET}")
        print(f"  🛡️ {BOLD}امتیاز کنترل و فشردگی ریسک (Risk Score):{RESET} {YELLOW}{scores_3d.get('risk_score', 0)}/100{RESET}")
        print(f"  👑 {BOLD}ضریب اطمینان نهایی و گرید:{RESET} {g_col}{BOLD}{scores_3d.get('composite_confidence', 0)}% [{scores_3d.get('grade')}]{RESET}")
        print(f"  💡 {BOLD}دستورالعمل:{RESET} {scores_3d.get('action_advice')}")

    # Backtest Mode
    if args.mode in ["backtest"]:
        print(f"\n{YELLOW}{BOLD}------------------------------------------------------------------------{RESET}")
        print(f"{YELLOW}{BOLD}🧪 شبیه‌ساز و بک‌تست تاریخی استراتژی (Empirical Backtest Simulation):{RESET}")
        print(f"📊 {BOLD}تعداد کل تریدهای شناسایی‌شده:{RESET} {backtest.get('total_trades')}")
        print(f"🟢 {BOLD}تعداد معاملات برنده (Wins):{RESET} {GREEN}{backtest.get('win_count')}{RESET} | 🔴 {BOLD}بازنده (Losses):{RESET} {RED}{backtest.get('loss_count')}{RESET}")
        print(f"🎯 {BOLD}وین‌ریت تجربی سیستم (Win Rate):{RESET} {CYAN}{BOLD}{backtest.get('win_rate_pct')}%{RESET}")
        print(f"⚖️ {BOLD}فاکتور سود (Profit Factor):{RESET} {GREEN}{backtest.get('profit_factor')}{RESET}")
        print(f"💰 {BOLD}سود خالص شبیه‌سازی:{RESET} {GREEN}{backtest.get('net_profit_pct'):+.2f}% (${backtest.get('net_profit_usd'):+,.2f}){RESET}")
        print(f"🛡️ {BOLD}حداکثر افت سرمایه (Max Drawdown):{RESET} {RED}{backtest.get('max_drawdown_pct')}%{RESET}")
        print(f"👑 {BOLD}ضریب اطمینان کالیبره‌شده ریاضی:{RESET} {YELLOW}{backtest.get('calibrated_confidence')}/100{RESET}")
        
        trades = backtest.get("recent_trades", [])
        if trades:
            print(f"\n{BOLD}📋 لیست آخرین معاملات شبیه‌سازی‌شده:{RESET}")
            for tr in trades[-6:]:
                res_col = GREEN if tr['pnl_pct'] > 0 else RED
                print(f"  • {tr['time']} | {tr['type']} @ {format_price(tr['entry'])} -> خروج در {format_price(tr['exit'])} | {res_col}{tr['outcome']} ({tr['pnl_pct']:+.2f}%){RESET}")

    # Macro & On-Chain Mode
    if args.mode in ["all", "macro"]:
        print(f"\n{BLUE}{BOLD}------------------------------------------------------------------------{RESET}")
        print(f"{BLUE}{BOLD}🌐 وضعیت کلان بازار و داده‌های آن‌چین (Macro & On-chain):{RESET}")
        print(f"  🌍 {BOLD}ارزش کل بازار کریپتو (Total Market Cap):{RESET} {macro.get('total_market_cap_fmt')} ({macro.get('mcap_change_24h'):+.2f}%)")
        print(f"  🪙 {BOLD}سلطه بیت‌کوین (BTC Dominance):{RESET} {YELLOW}{macro.get('btc_dominance')}%{RESET} | {BOLD}سلطه تتر (USDT.D):{RESET} {macro.get('usdt_dominance')}%")
        print(f"  🌊 {BOLD}رژیم معاملاتی بازار:{RESET} {GREEN if 'Risk-On' in macro.get('regime','') else YELLOW}{macro.get('regime')}{RESET}")
        print(f"  🔗 {BOLD}همبستگی با بیت‌کوین:{RESET} {correlation.get('desc', 'N/A')}")
        print(f"  ⛓️ {BOLD}حجم تراکنش‌های ۲۴ ساعته زنجیره بیت‌کوین:{RESET} {onchain.get('tx_volume_usd')} ({onchain.get('n_tx_24h')} تراکنش)")
        print(f"  ⚡ {BOLD}ترافیک و کارمزد شبکه:{RESET} {onchain.get('network_load')} (کارمزد پیشنهادی: {onchain.get('recommended_fee_sat_vb')} sat/vB)")

    # News & Circuit Breaker Mode
    if args.mode in ["all", "news"]:
        cb_color = GREEN if news_data.get("circuit_status") == "NORMAL_CLEAR" else (YELLOW if news_data.get("circuit_status") == "CAUTION_HIGH_VOLATILITY" else RED)
        print(f"\n{RED if news_data.get('badge')=='RED' else YELLOW}{BOLD}------------------------------------------------------------------------{RESET}")
        print(f"{BOLD}📰 فیوز اطمینان اخبار و سنتیمنت (News Circuit Breaker):{RESET}")
        print(f"  🛡️ {BOLD}وضعیت فیوز حفاظتی:{RESET} {cb_color}{BOLD}{news_data.get('circuit_title')}{RESET}")
        print(f"  💡 {BOLD}دستورالعمل خبری:{RESET} {news_data.get('circuit_advice')}")
        
        n_items = news_data.get("news_items", [])
        if n_items:
            print(f"  📌 {BOLD}آخرین تیترهای فوری بازار:{RESET}")
            for it in n_items[:4]:
                s_col = RED if it['sentiment'] == 'BEARISH_PANIC' else (GREEN if it['sentiment'] == 'BULLISH_CATALYST' else CYAN)
                print(f"    • [{s_col}{it['sentiment_fa']}{RESET}] {it['title']}")

    # Smart Money & HFT Section
    if args.mode in ["all", "smc"]:
        hft = smc.get("hft", {})
        fake = smc.get("fake_trend", {})
        poc = smc.get("poc", price)
        fvgs = smc.get("unmitigated_fvgs", [])
        sweeps = smc.get("sweeps", [])
        
        print(f"\n{BLUE}{BOLD}------------------------------------------------------------------------{RESET}")
        print(f"{BLUE}{BOLD}🧠 کالبدشکافی اسمارت‌مانی (ICT / Smart Money Concepts):{RESET}")
        print(f"🤖 {BOLD}شاخص نفوذ ربات‌های HFT و بانک‌ها:{RESET} {YELLOW}{hft.get('score')}/100{RESET} ({hft.get('status')})")
        print(f"🎯 {BOLD}وضعیت سلامت روند / تله‌ها:{RESET} {GREEN if fake.get('badge')=='ORGANIC' else RED}{fake.get('title')}{RESET}")
        print(f"⏰ {BOLD}سشن و کیل‌زون معاملاتی:{RESET} {CYAN}{smc.get('killzone', {}).get('killzone', '')}{RESET} ({smc.get('killzone', {}).get('active_session', '')})")
        print(f"📐 {BOLD}وضعیت واگرایی (RSI):{RESET} {YELLOW}{smc.get('divergence', {}).get('desc', '')}{RESET}")
        print(f"🧲 {BOLD}مرکز تراکم حجم نقدینگی (POC):{RESET} {CYAN}{format_price(poc)}{RESET}")
        print(f"📦 {BOLD}استخر نقدینگی سقف (BSL Target):{RESET} {format_price(smc.get('bsl_pool_target'))}")
        print(f"📦 {BOLD}استخر نقدینگی کف (SSL Target):{RESET} {format_price(smc.get('ssl_pool_target'))}")

        if fvgs:
            print(f"⚡ {BOLD}خلاءهای نقدینگی فعال (Unmitigated FVGs):{RESET}")
            for f in fvgs:
                col = GREEN if f['type'] == 'BULLISH_FVG' else RED
                print(f"   • {col}{f['title']}:{RESET} {format_price(f['bottom'])} تا {format_price(f['top'])} (فاصله: {f['distance_pct']}%)")
        else:
            print(f"⚡ {BOLD}خلاء نقدینگی فعال:{RESET} در حال حاضر FVG پرنشده نزدیک دیده نمی‌شود.")

        if sweeps:
            latest_sw = sweeps[-1]
            print(f"🎣 {BOLD}آخرین شکار نقدینگی (Liquidity Sweep):{RESET} {RED}{latest_sw['title']}{RESET} در سطح {format_price(latest_sw['level_swept'])}")

        if derivatives.get("has_data"):
            print(f"📊 {BOLD}قراردادهای باز مشتقه (Open Interest):{RESET} {derivatives.get('open_interest_fmt')} | {BOLD}فاندینگ ریت:{RESET} {derivatives.get('funding_rate_fmt')} ({derivatives.get('market_crowd')})")

    # Institutional Matrix & VWAP / CVD Section
    if args.mode in ["all", "inst"]:
        print(f"\n{MAGENTA}{BOLD}------------------------------------------------------------------------{RESET}")
        print(f"{MAGENTA}{BOLD}🏛️ ماتریس نهادی، حجم وزنی (VWAP) و جریان سفارشات تهاجمی (CVD):{RESET}")
        print(f"  🌊 {BOLD}میانگین وزنی روزانه (VWAP):{RESET} {format_price(vwap_cvd.get('vwap'))} ({vwap_cvd.get('vwap_position')})")
        print(f"  📈 {BOLD}محدوده ارزش (Value Area 70%):{RESET} سقف (VAH): {format_price(vwap_cvd.get('vah'))} | کف (VAL): {format_price(vwap_cvd.get('val'))}")
        print(f"  ⚡ {BOLD}جریان سفارشات تهاجمی (CVD Trend):{RESET} {CYAN}{vwap_cvd.get('cvd_trend')}{RESET} (دلتای آخرین کندل: {vwap_cvd.get('delta_latest', 0):+.2f})")
        if matrix:
            print(f"  🧩 {BOLD}ماتریس ۴ گانه مشتقات (Price vs OI):{RESET} {YELLOW}{matrix.get('regime')} ({matrix.get('regime_fa')}){RESET}")
            print(f"     {DIM}{matrix.get('implication')}{RESET}")

        if inst_table:
            print(f"\n{BOLD}📋 جدول سیگنال فوق‌حرفه‌ای سازمانی (Institutional Signal Table):{RESET}")
            print(f"{DIM}{'-'*72}{RESET}")
            for idx, row in enumerate(inst_table):
                status_color = GREEN if row['status'] == 'PASS' else (YELLOW if row['status'] == 'WARN' else (RED if row['status'] == 'DANGER' else BLUE))
                print(f" {idx+1:2d}. {BOLD}{row['param']}:{RESET} {row['val']} [{status_color}{row['status']}{RESET}]")
            print(f"{DIM}{'-'*72}{RESET}")

    # Scalp Section
    if args.mode in ["all", "scalp"]:
        action_col = GREEN if "BUY" in scalp.get("action_code", "") else (RED if "SELL" in scalp.get("action_code", "") else YELLOW)
        print(f"\n{CYAN}{BOLD}------------------------------------------------------------------------{RESET}")
        print(f"{CYAN}{BOLD}⚡ ستاپ معاملاتی اسکالپینگ (Scalp Setup - تایم‌فریم ۵ و ۱۵ دقیقه):{RESET}")
        print(f"📌 {BOLD}سیگنال لحظه‌ای:{RESET} {action_col}{BOLD}{scalp.get('action')}{RESET}")
        print(f"🔹 {BOLD}محدوده ورود بهینه:{RESET} {scalp.get('entry_zone')}")
        print(f"🛑 {BOLD}حد ضرر (Stop Loss):{RESET} {RED}{BOLD}{format_price(scalp.get('stop_loss'))} (-{scalp.get('stop_loss_pct')}%) {RESET}")
        print(f"🎯 {BOLD}تارگت اول (TP 1):{RESET} {GREEN}{format_price(scalp.get('tp1'))} (+{scalp.get('tp1_pct')}%) {RESET}")
        print(f"🎯 {BOLD}تارگت دوم (TP 2):{RESET} {GREEN}{format_price(scalp.get('tp2'))} (+{scalp.get('tp2_pct')}%) {RESET}")
        print(f"🎯 {BOLD}تارگت سوم (TP 3):{RESET} {GREEN}{format_price(scalp.get('tp3'))} (+{scalp.get('tp3_pct')}%) {RESET}")
        print(f"⚖️ {BOLD}ریسک به ریوارد:{RESET} {scalp.get('risk_reward')} | {BOLD}اهرم پیشنهادی:{RESET} {scalp.get('suggested_leverage')}")
        print(f"⏳ {BOLD}افق زمانی:{RESET} {scalp.get('estimated_duration')} | {BOLD}احتمال موفقیت:{RESET} {scalp.get('confidence')}")

    # Swing Section
    if args.mode in ["all", "swing"]:
        swing_action_col = GREEN if "BUY" in swing.get("action_code", "") else (RED if "SELL" in swing.get("action_code", "") else YELLOW)
        print(f"\n{MAGENTA}{BOLD}------------------------------------------------------------------------{RESET}")
        print(f"{MAGENTA}{BOLD}🌊 ستاپ معاملاتی سوئینگ تریدینگ (Swing Setup - تایم‌فریم ۴ ساعته و روزانه):{RESET}")
        print(f"📌 {BOLD}استراتژی کلان:{RESET} {swing_action_col}{BOLD}{swing.get('action')}{RESET}")
        print(f"🔹 {BOLD}ورود در اصلاح (POC):{RESET} {swing.get('pullback_entry')}")
        print(f"🔹 {BOLD}ورود در بریک‌اوت:{RESET} {swing.get('breakout_entry')}")
        print(f"🛑 {BOLD}حد ضرر ساختاری (SL):{RESET} {RED}{BOLD}{format_price(swing.get('stop_loss'))} (-{swing.get('stop_loss_pct')}%) {RESET}")
        print(f"🎯 {BOLD}تارگت اول (سقف قبلی):{RESET} {GREEN}{format_price(swing.get('target1'))} (+{swing.get('target1_pct')}%) {RESET}")
        print(f"🎯 {BOLD}تارگت دوم (فیبو 1.272):{RESET} {GREEN}{format_price(swing.get('target2'))} (+{swing.get('target2_pct')}%) {RESET}")
        print(f"🎯 {BOLD}تارگت سوم (فیبو 1.618):{RESET} {GREEN}{format_price(swing.get('target3'))} (+{swing.get('target3_pct')}%) {RESET}")
        print(f"⚖️ {BOLD}ریسک به ریوارد:{RESET} {swing.get('risk_reward')} | {BOLD}مدت نگهداری:{RESET} {swing.get('holding_period')}")
        print(f"🛡️ {BOLD}قانون مدیریت سرمایه:{RESET} {swing.get('capital_risk_advice')}")

    # Trade Quality & Confluence Checklist
    tq = data.get("trade_quality", {})
    if tq:
        grade_col = YELLOW if tq.get("grade") == "A+" else (GREEN if tq.get("grade") == "A" else RED)
        print(f"\n{YELLOW}{BOLD}------------------------------------------------------------------------{RESET}")
        print(f"{YELLOW}{BOLD}👑 ارزیابی دقت و درجه سیگنال (Signal Quality Shield):{RESET}")
        print(f"⭐ {BOLD}درجه سیگنال:{RESET} {grade_col}{BOLD}{tq.get('grade_title')}{RESET}")
        print(f"💡 {BOLD}توصیه اجرایی:{RESET} {tq.get('action_advice')} (امتیاز تاییدیه: {tq.get('score')}/100)")
        print(f"📋 {BOLD}چک‌لیست ۶ گانه ورود امن:{RESET}")
        for c in tq.get("checklist", []):
            mark = f"{GREEN}✔{RESET}" if c["passed"] else f"{RED}✖{RESET}"
            print(f"   [{mark}] {c['item']}: {DIM}{c['detail']}{RESET}")

    # Precision Suite Mode (Liquidations, Whale Flow, Calendar Shield, Options, Spoofing)
    if args.mode in ["all", "precision"]:
        from institutional_addons import (
            LiquidationHeatmapEngine, WhaleFlowEngine, EconomicCalendarEngine,
            OptionsEngine, OrderbookDepthSpoofingEngine
        )
        h24 = float(ticker.get("high_24h") or price * 1.02)
        l24 = float(ticker.get("low_24h") or price * 0.98)
        base_coin = data.get("base_coin", "BTC")

        liq = LiquidationHeatmapEngine.calculate_clusters(sym, price, h24, l24)
        whales = WhaleFlowEngine.get_whale_metrics()
        macro_cal = EconomicCalendarEngine.get_macro_shield_status()
        options = OptionsEngine.get_options_analytics(base_coin)
        depth_spoof = OrderbookDepthSpoofingEngine.scan_depth_and_spoofing(sym)

        print(f"\n{CYAN}{BOLD}========================================================================{RESET}")
        print(f"{CYAN}{BOLD}💎 رادار ۵ گانه ارتقای دقت سازمانی (Hedge-Fund Precision Suite):{RESET}")

        # 1. Liquidation Clusters
        casc_col = RED if liq.get("cascade_warning") else GREEN
        print(f"\n{BOLD}1. 🧲 خوشه‌های لیکوئیدیشن اهرم‌های بالا (Coinglass Style Liquidations):{RESET}")
        print(f"  🔻 {BOLD}مجموع استاپ‌های لانگ در خطر:{RESET} {RED}{liq.get('total_long_liq_fmt')}{RESET} | 🔺 {BOLD}شورت‌ها:{RESET} {GREEN}{liq.get('total_short_liq_fmt')}{RESET}")
        print(f"  🧲 {BOLD}قیمت آهنربای جذب نقدینگی (Magnet Price):{RESET} {CYAN}{format_price(liq.get('magnet_price'))}{RESET} (فاصله: {liq.get('magnet_distance_pct'):+.2f}%)")
        print(f"  ⚡ {BOLD}وضعیت آبشاری لیکوئیدیتی:{RESET} {casc_col}{liq.get('cascade_status')}{RESET}")

        # 2. Whale Netflow
        wf_col = GREEN if "خروج" in whales.get("flow_status","") else RED
        print(f"\n{BOLD}2. 🐋 رادار ورود/خروج نهنگ‌ها به صرافی (Whale Exchange Netflow):{RESET}")
        print(f"  🌊 {BOLD}جریان خالص ۲۴ ساعته صرافی‌ها:{RESET} {wf_col}{whales.get('netflow_fmt')}{RESET}")
        print(f"  🧭 {BOLD}سنتیمنت نهنگ‌ها:{RESET} {wf_col}{whales.get('flow_status')}{RESET} (امتیاز انباشت: {whales.get('accumulation_score')}/100)")

        # 3. Macro Economic Calendar & Shield
        sh_col = GREEN if "باثبات" in macro_cal.get("shield_state","") else (YELLOW if "ریسک بالا" in macro_cal.get("shield_state","") else RED)
        print(f"\n{BOLD}3. 🏛️ تقویم رویدادهای ماکرو و فیوز کلان (Economic Shield & Circuit Breaker):{RESET}")
        print(f"  📅 {BOLD}رویداد بعدی کلان آمریکا:{RESET} {YELLOW}{macro_cal.get('next_event')}{RESET}")
        print(f"  ⏳ {BOLD}شمارش معکوس زنده:{RESET} {macro_cal.get('countdown_fmt')} (پیش‌بینی: {macro_cal.get('forecast')} | قبلی: {macro_cal.get('previous')})")
        print(f"  🛡️ {BOLD}وضعیت فیوز معاملاتی:{RESET} {sh_col}{macro_cal.get('shield_state')}{RESET}")

        # 4. Deribit Options & Max Pain
        pcr_col = GREEN if "صعودی" in options.get("pcr_sentiment","") else (RED if "نزولی" in options.get("pcr_sentiment","") else YELLOW)
        print(f"\n{BOLD}4. 📊 بازار آپشن‌ها و قیمت ماکس‌پین دریبیت (Deribit Options & Max Pain):{RESET}")
        print(f"  🎯 {BOLD}قیمت ماکس‌پین قراردادها (Max Pain Strike):{RESET} {CYAN}{format_price(options.get('max_pain_strike'))}{RESET}")
        print(f"  ⚖️ {BOLD}نسبت اختیار فروش به خرید (Put/Call Ratio):{RESET} {options.get('pcr_ratio')} [{pcr_col}{options.get('pcr_sentiment')}{RESET}]")
        print(f"  📦 {BOLD}کل حجم قراردادهای باز:{RESET} کال‌ها {options.get('total_calls_oi'):,.0f} | پوت‌ها {options.get('total_puts_oi'):,.0f}")

        # 5. Orderbook Depth & Spoofing
        spf_col = GREEN if "واقعی" in depth_spoof.get("spoofing_status","") else RED
        print(f"\n{BOLD}5. 🧱 عمق ۱۰۰ پله اردربوک و ردیاب سفارشات فیک (Depth & Spoofing Detector):{RESET}")
        print(f"  ⚖️ {BOLD}توازن عمق دفتر سفارشات:{RESET} تقاضا (Bid) {depth_spoof.get('bid_percentage')}% | عرضه (Ask) {depth_spoof.get('ask_percentage')}% ({depth_spoof.get('depth_bias')})")
        print(f"  🛡️ {BOLD}آشکارساز سفارش‌گذاری جعلی:{RESET} {spf_col}{depth_spoof.get('spoofing_status')}{RESET}")
        if depth_spoof.get("buy_wall"):
            print(f"  🟢 {BOLD}بزرگ‌ترین دیواره خرید:{RESET} {format_price(depth_spoof['buy_wall']['price'])} (${depth_spoof['buy_wall']['val_usd']:,.0f} در فاصله {depth_spoof['buy_wall']['dist_pct']}%)")
        if depth_spoof.get("sell_wall"):
            print(f"  🔴 {BOLD}بزرگ‌ترین دیواره فروش:{RESET} {format_price(depth_spoof['sell_wall']['price'])} (${depth_spoof['sell_wall']['val_usd']:,.0f} در فاصله {depth_spoof['sell_wall']['dist_pct']}%)")

    # Alerts
    alerts = data.get("verdict", {}).get("danger_alerts", [])

    if alerts:
        print(f"\n{RED}{BOLD}🚨 هشدارهای اضطراری و نقدینگی:{RESET}")
        for a in alerts:
            print(f"  • {RED}{a}{RESET}")

    # Alpha Leaders Matrix Mode
    if args.mode in ["all", "alpha"]:
        from institutional_addons import AlphaCorrelationEngine
        alpha_data = AlphaCorrelationEngine.get_leaders_alpha_matrix()
        print(f"\n{YELLOW}{BOLD}------------------------------------------------------------------------{RESET}")
        print(f"{YELLOW}{BOLD}🗺️ ماتریس مقایسه‌ای لیدرهای بازار و ضریب آلفا (Alpha & Relative Strength Matrix):{RESET}")
        print(f"🪙 {BOLD}بیت‌کوین مرجع:{RESET} {format_price(alpha_data.get('btc_price'))} ({alpha_data.get('btc_change_24h'):+.2f}%)")
        leaders_list = alpha_data.get("leaders", [])
        if leaders_list:
            print(f"{DIM}{'نماد':<8} | {'قیمت لحظه‌ای':<14} | {'نوسان ۲۴h':<10} | {'ضریب آلفا vs BTC':<18} | {'وضعیت لیدری'}{RESET}")
            print(f"{DIM}{'-'*75}{RESET}")
            for l in leaders_list:
                col = GREEN if l['alpha_vs_btc'] > 0 else RED
                print(f" {BOLD}{l['symbol']:<7}{RESET} | {l['price_fmt']:<14} | {l['change_24h']:+6.2f}%    | {col}{l['alpha_vs_btc']:+6.2f}%{RESET}            | {l['status']}")
            print(f"{DIM}{'-'*75}{RESET}")

    # Kelly Risk & Position Sizing Mode
    if args.mode in ["all", "risk"]:
        from institutional_addons import KellyRiskEngine
        scalp = data.get("scalp_setup", {})
        e_price = float(scalp.get("entry_price") or price)
        sl_price = float(scalp.get("stop_loss") or (price * 0.985 if "BUY" in scalp.get("direction", "BUY") else price * 1.015))
        tp_price = float(scalp.get("tp2") or scalp.get("tp1") or (price * 1.025 if "BUY" in scalp.get("direction", "BUY") else price * 0.975))
        win_r = float(data.get("backtest", {}).get("win_rate_pct", 55.0))
        
        kelly_data = KellyRiskEngine.calculate_risk_and_kelly(
            balance=args.balance,
            risk_pct=args.risk_pct,
            entry_price=e_price,
            stop_loss=sl_price,
            take_profit=tp_price,
            win_rate_pct=win_r,
            direction=scalp.get("direction", "LONG")
        )

        print(f"\n{GREEN}{BOLD}------------------------------------------------------------------------{RESET}")
        print(f"{GREEN}{BOLD}🧮 ماشین‌حساب مدیریت سرمایه نهادی و معیار کِلی (Kelly Criterion Calculator):{RESET}")
        print(f"  💰 {BOLD}سرمایه کل حساب:{RESET} ${args.balance:,.2f} | {BOLD}سقف ریسک دلاری:{RESET} {RED}{kelly_data.get('dollar_risk_fmt')}{RESET} ({args.risk_pct}%)")
        print(f"  🎯 {BOLD}فاصله تا استاپ‌لاس:{RESET} {kelly_data.get('sl_distance_pct')}% | {BOLD}سود هدف (TP):{RESET} {GREEN}{kelly_data.get('dollar_reward_fmt')}{RESET} ({kelly_data.get('rr_fmt')})")
        print(f"  📦 {BOLD}حجم دقیق معامله (Position Size):{RESET} {CYAN}{BOLD}{kelly_data.get('position_size_fmt')}{RESET} ({kelly_data.get('coin_units')} {data.get('base_coin')})")
        print(f"  🛡️ {BOLD}حداکثر اهرم امن (Safe Leverage):{RESET} {YELLOW}{kelly_data.get('safe_leverage_fmt')}{RESET} (مارجین درگیر: {kelly_data.get('margin_fmt')})")
        print(f"  📐 {BOLD}درصد بهینه معیار کِلی (Kelly Half %):{RESET} {GREEN}{kelly_data.get('kelly_half_pct')}%{RESET} (تخصیص بهینه: ${kelly_data.get('kelly_half_dollar'):,.2f})")
        print(f"  💡 {BOLD}ارزیابی ریاضی معامله:{RESET} {kelly_data.get('advice')}")

    # Golden 6 Vital Noise-Free Filters Mode
    if args.mode in ["all", "golden"]:
        from institutional_addons import GoldenSixCoreEngine
        scalp = data.get("scalp_setup", {})
        dir_val = "LONG" if "BUY" in scalp.get("action", "BUY") or "LONG" in scalp.get("direction", "LONG") else "SHORT"
        golden = GoldenSixCoreEngine.evaluate(
            symbol=data.get("base_coin", "BTC"),
            current_price=float(price),
            direction=dir_val
        )

        print(f"\n{CYAN}{BOLD}========================================================================{RESET}")
        print(f"{CYAN}{BOLD}⭐ فیلترهای ۶ گانه طلایی آنچین و مشتقات (The 6 Vital Noise-Free Filters):{RESET}")
        print(f"  🏆 {BOLD}امتیاز نهایی سازمانی:{RESET} {GREEN if golden['pass_count']>=5 else (YELLOW if golden['pass_count']==4 else RED)}{golden['pass_count']}/6 ({golden['grade']}){RESET} | {BOLD}جهت ارزیابی:{RESET} {dir_val}")
        print(f"  🧭 {BOLD}حکم نهایی:{RESET} {golden['verdict_fa']}")
        print(f"  🎯 {BOLD}توصیه اجرایی:{RESET} {golden['action_fa']}")
        print(f"{DIM}{'-'*75}{RESET}")
        for f in golden["filters"]:
            status_icon = f"{GREEN}✅ PASS{RESET}" if f["passed"] else f"{RED}❌ FAIL{RESET}"
            print(f"  {f['id']}. {BOLD}{f['name_fa']:<35}{RESET} [{status_icon}] {DIM}({f['value_display']}){RESET}")
            print(f"     {DIM}↳ {f['description']}{RESET}")
        print(f"{CYAN}{BOLD}========================================================================{RESET}")

    # Telegram Dispatch
    if args.telegram:
        import os
        from institutional_addons import TelegramDispatcher
        cfg_file = os.path.join(os.path.dirname(__file__), "telegram_config.json")
        bot_token = ""
        chat_id = ""
        if os.path.exists(cfg_file):
            try:
                with open(cfg_file, "r") as f:
                    cfg = json.load(f)
                    bot_token = cfg.get("bot_token", "")
                    chat_id = cfg.get("chat_id", "")
            except Exception:
                pass
        
        tg_res = TelegramDispatcher.send_to_telegram(bot_token, chat_id, data)
        print(f"\n{CYAN}{BOLD}📱 وضعیت ارسال به تلگرام:{RESET}")
        if tg_res.get("simulated"):
            print(f"{YELLOW}ℹ️ {tg_res.get('message')}{RESET}")
            print(f"{DIM}{tg_res.get('preview_text')}{RESET}")
        elif tg_res.get("success"):
            print(f"{GREEN}✔ سیگنال با موفقیت به چنل/گروه تلگرام ارسال شد! (Message ID: {tg_res.get('telegram_message_id')}){RESET}")
        else:
            print(f"{RED}✖ خطا در ارسال تلگرام: {tg_res.get('error')}{RESET}")

    print(f"\n{DIM}زمان تحلیل: {data.get('analyzed_at')}{RESET}")
    print(f"{CYAN}{BOLD}========================================================================{RESET}\n")

if __name__ == "__main__":
    run_cli()
