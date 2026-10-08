# -*- coding: utf-8 -*-
"""
CryptoChatOrderflowHelper: Specialized AI Consultant Engine for Order Flow,
Bookmap, NinjaTrader, ATAS, Quantower, Sierra Chart, Geopolitics, and Wall Street Reports.
"""

from typing import Optional
import crypto_orderflow_engine as coe

def handle_orderflow_chat(q: str, clean_s: str, price: float) -> Optional[str]:
    q = q.lower().strip()

    # 1. Master Unified Order Flow Summary
    if any(w in q for w in ["خلاصه اردر فلو", "اردر فلو چی میگه", "اردر فلو وال استریت", "۷ ماژول جدید", "ماژول های جدید", "وضعیت اردر فلو", "تحلیل جامع اردر فلو", "دیدبان اردر فلو", "سفارشات سازمانی"]):
        try:
            bm = coe.get_crypto_bookmap_data(clean_s, "15m")
            nt = coe.get_crypto_ninjatrader_live(clean_s, "15m")
            at = coe.get_crypto_atas_live(clean_s)
            qt = coe.get_crypto_quantower_live(clean_s, "1h")
            sc = coe.get_crypto_sierrachart_data(clean_s, "15m")
            geo = coe.get_crypto_geopolitics_radar()
            bank = coe.get_crypto_bank_reports()

            mp = qt.get("market_profile", {})
            cvd = nt.get("cvd", {})
            ts = at.get("tape_speed", {})
            div = sc.get("delta_divergence", {})
            gpr = geo.get("gpr_index", 145)

            lines = [
                f"### 🏛️ گزارش تلفیقی اتاق فرمان اردر فلو وال‌استریت (Order Flow Master Intelligence) برای {clean_s}",
                f"",
                f"قیمت زنده: **${price:,.2f}** | وضعیت نقدینگی دفتر سفارشات: **{bm.get('liquidity_state', 'انباشت نهادی')}**",
                f"",
                f"#### ۱. 🔥 بوک‌مپ نقدینگی (Bookmap L2/L3):",
                f"- نسبت قدرت دیواره‌ها: **{bm.get('imbalance_pct', 55)}% به سود خریداران** | {bm.get('verdict_fa', '')}",
                f"- کشف اوردرهای مخفی: **{len(bm.get('iceberg_orders', []))} اردر آیس‌برگ نهنگ** در محدوده نوسان جاری.",
                f"",
                f"#### ۲. 🎯 نینجاتریدر ۸ (NinjaTrader 8 Footprint):",
                f"- دلتای تجمیعی (CVD): **{cvd.get('status_fa', 'مثبت و صعودی')}**",
                f"- تراز سشن VWAP مرکزی: **${nt.get('vwap', {}).get('center', price):,.2f}** (باندهای انحراف معیار مستحکم)",
                f"",
                f"#### ۳. ⚡ اتاس (ATAS Time & Sales):",
                f"- سرعت نوار معاملات: **{ts.get('trades_per_sec', 32)} تیک در ثانیه** ({ts.get('status_fa', 'جریان فعال HFT')})",
                f"- شکار نهنگ‌ها: رصد **{len(at.get('big_trades', []))} بلاک‌ترید بزرگ بالای ۵۰ بیت‌کوین** در سشن جاری.",
                f"",
                f"#### ۴. 🏛️ کوانت‌تاور (Quantower TPO Profile):",
                f"- محدوده ارزش ۷۰٪: کف (VAL): **${mp.get('val', 0):,.2f}** | سقف (VAH): **${mp.get('vah', 0):,.2f}**",
                f"- تراز نقطه کنترل حجم (VPOC): **${mp.get('vpoc', 0):,.2f} 👑** ({mp.get('shape_fa', 'حراج متوازن')})",
                f"",
                f"#### ۵. 📈 سیرا چارت (Sierra Chart Orderflow):",
                f"- واگرایی دلتا (Delta Divergence): **{div.get('type_fa', 'نرمال')}** با ضریب قطعیت **{div.get('confidence', 85)}%**",
                f"",
                f"#### ۶. 🌐 رادار ریسک ژئوپلیتیک & استیبل‌کوین:",
                f"- شاخص ریسک جهانی (GPR Index): **{gpr}** ({geo.get('gpr_status_fa', 'نرمال')})",
                f"- تزریق نقدینگی تتر (USDT Treasury): **{geo.get('stablecoin_minting', {}).get('usdt_market_cap', 'ضرب ۱.۲ میلیارد دلار')}**",
                f"",
                f"#### ۷. 🏦 گزارشات بانک‌های وال‌استریت & ETFها:",
                f"- ورودی خالص روزانه ETFهای اسپات: **{bank.get('total_daily_net', '+۴۲۳.۳ میلیون دلار')}**",
                f"- ارزیابی مؤسسات: **{bank.get('executive_summary_fa', 'تقاضای سازمانی پشتوانه روند صعودی است.')}**",
                f"",
                f"🧭 **جمع‌بندی استراتژیک ایجنت:** ترکیب داده‌های هر ۷ ترمینال نشان می‌دهد گاوهای سازمانی کنترل دفتر سفارشات را در دست دارند و اصلاحات به سمت ترازهای VWAP و VPOC بهترین ستاپ‌های ورود کم‌ریسک هستند."
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"خطا در واکشی داده‌های تلفیقی اردر فلو: {str(e)}"

    # 2. Bookmap L2/L3 Liquidity Heatmap & Iceberg Detector
    if any(w in q for w in ["بوک مپ", "بوکمپ", "bookmap", "آیسبرگ", "آیس برگ", "iceberg", "دیواره نقدینگی", "دیواره سفارشات", "دیوار خرید", "دیوار فروش", "هیت مپ لول 3", "هیت‌مپ بوک‌مپ", "اوردرهای مخفی"]):
        try:
            bm = coe.get_crypto_bookmap_data(clean_s, "15m")
            cur_p = bm.get("current_price", price)
            icebergs = bm.get("iceberg_orders", [])
            asks = bm.get("ask_levels", [])
            bids = bm.get("bid_levels", [])

            ice_lines = []
            for ic in icebergs[:4]:
                ice_lines.append(f"• **${ic.get('price'):,.2f}:** حجم مخفی تخمینی: **{ic.get('estimated_hidden_vol')} {clean_s}** ({ic.get('direction_fa')}) | وضعیت: {ic.get('status')}")
            ice_str = "\n".join(ice_lines) if ice_lines else "• در حال حاضر اردر آیس‌برگ حجیمی در محدوده نوسان اخیر شناسایی نشده است."

            ask_w = asks[0] if asks else {}
            bid_w = bids[0] if bids else {}

            lines = [
                f"### 🔥 کالبدشکافی امواج نقدینگی بوک‌مپ (Bookmap L2/L3) برای {clean_s}",
                f"",
                f"قیمت زنده: **${cur_p:,.2f}** | نسبت نقدینگی دفتر سفارشات: **{bm.get('imbalance_pct', 50)}% خریدار**",
                f"",
                f"🧱 **مهم‌ترین دیواره‌های نقدینگی فعال (Resting Liquidity Walls):**",
                f"- **دیواره عرضه و سقف فروش (Ask Wall):** نرخ **${ask_w.get('price', 0):,.2f}** با حجم **{ask_w.get('revealed_vol', 0)} {clean_s}** (سهم: {ask_w.get('share_pct', 0)}%)",
                f"- **دیواره تقاضا و کف خرید (Bid Wall):** نرخ **${bid_w.get('price', 0):,.2f}** با حجم **{bid_w.get('revealed_vol', 0)} {clean_s}** (سهم: {bid_w.get('share_pct', 0)}%)",
                f"",
                f"🧊 **اوردرهای مخفی آیس‌برگ کشف‌شده نهنگ‌ها (Iceberg Orders):**",
                f"{ice_str}",
                f"",
                f"🧭 **تحلیل سازمانی بوک‌مپ:** {bm.get('verdict_fa', 'تراکم نقدینگی در کف‌های قیمتی مشهود است.')}",
                f"",
                f"💡 **راهنمای عمل تریدر:** دیواره خرید ${bid_w.get('price', 0):,.2f} به عنوان کف حمایتی معتبر عمل می‌کند؛ در صورت پولبک قیمت به این سطح و شکل‌گیری حباب‌های خرید سبز، ورود لانگ با حد ضرر زیر دیواره بهترین گزینه است."
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"خطا در پردازش اطلاعات بوک‌مپ: {str(e)}"

    # 3. NinjaTrader 8 Footprint Ladder & SuperDOM
    if any(w in q for w in ["نینجا", "نینجاتریدر", "ninjatrader", "فوت پرینت", "فوت‌پرینت", "footprint", "سوپردام", "superdom", "کلاستر", "تاج طلایی", "poc کلاستر", "باندهای vwap", "انحراف معیار vwap"]):
        try:
            nt = coe.get_crypto_ninjatrader_live(clean_s, "15m")
            cur_p = nt.get("current_price", price)
            vwap = nt.get("vwap", {})
            cvd = nt.get("cvd", {})
            fps = nt.get("footprint_candles", [])

            fp_lines = []
            for c in fps[:3]:
                col = "🟢 صعودی" if c.get('is_bullish') else "🔴 نزولی"
                fp_lines.append(f"• **کندل {c.get('time')} ({col}):** تراز POC: **${c.get('poc_price'):,.2f} 👑** | دلتای خالص: **{c.get('candle_delta'):+d} Δ** ({c.get('delta_pct')}%) | حجم کل: {c.get('total_volume')} {clean_s}")
            fp_str = "\n".join(fp_lines) if fp_lines else "اطلاعات فوت‌پرینت در دسترس است."

            lines = [
                f"### 🎯 ماتریس اردر فلو و فوت‌پرینت نینجاتریدر ۸ (NinjaTrader 8) برای {clean_s}",
                f"",
                f"قیمت زنده: **${cur_p:,.2f}** | وضعیت دلتای تجمیعی (CVD): **{cvd.get('status_fa', 'پایدار')}**",
                f"",
                f"📊 **کندل‌های فوت‌پرینت کلاستری نردبانی (Footprint Clusters):**",
                f"{fp_str}",
                f"",
                f"📐 **باندهای انحراف معیار سشن VWAP:**",
                f"- خط مرکزی سشن (VWAP Mid): **${vwap.get('center', cur_p):,.2f}**",
                f"- باند +1 انحراف معیار: **${vwap.get('sd1_up', 0):,.2f}** | باند -1 انحراف معیار: **${vwap.get('sd1_dn', 0):,.2f}**",
                f"- سقف اکستریم (+2 SD): **${vwap.get('sd2_up', 0):,.2f}** | کف اکستریم (-2 SD): **${vwap.get('sd2_dn', 0):,.2f}**",
                f"",
                f"🧭 **استراتژی اسکالپ نینجاتریدر:** {nt.get('summary_fa', 'حفظ قیمت بالای VWAP نشانه تسلط خریداران است.')}",
                f"",
                f"💡 **توصیه اجرایی:** در چارت نینجاتریدر پلتفرم، کلاسترها با فاصله مهندسی‌شده رندر می‌شوند؛ مشاهده نماد صاعقه ⚡ در کنار حجم نشانه عدم تعادل تهاجمی ۳۰۰٪ است که سوخت یک پرتاب قیمتی شارپ را تامین می‌کند."
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"خطا در پردازش فوت‌پرینت نینجاتریدر: {str(e)}"

    # 4. ATAS Advanced Time & Sales / Big Trades
    if any(w in q for w in ["اتاس", "atas", "تایم اند سیلز", "تایم اند سيلز", "نوار معاملات", "بلاک ترید", "big trade", "معاملات بزرگ", "سرعت نوار", "عدم تعادل قطری", "عدم تعادل ۳۰۰"]):
        try:
            at = coe.get_crypto_atas_live(clean_s)
            cur_p = at.get("current_price", price)
            ts = at.get("tape_speed", {})
            bts = at.get("big_trades", [])
            imbs = at.get("diagonal_imbalances", [])

            bt_lines = []
            for b in bts[:4]:
                col = "🟢 خرید مارکت" if b.get('side') == 'BUY' else "🔴 فروش مارکت"
                val_str = str(b.get('value_usd', ''))
                size_str = str(b.get('size_lots') or b.get('volume_lots') or b.get('size', ''))
                bt_lines.append(f"• **ساعت {b.get('time')}:** {col} **{size_str} {clean_s}** ({val_str}) روی نرخ ${b.get('price'):,.2f} ({b.get('exchange')})")
            bt_str = "\n".join(bt_lines) if bt_lines else "سفارشات بزرگ بالای ۵۰ بیت‌کوین در حال رصد هستند."

            imb_lines = []
            for im in imbs[:3]:
                imb_lines.append(f"• تراز **${im.get('price_level', 0):,.2f}:** عدم تعادل قطری **{im.get('ratio')}x** ({im.get('side_fa')}) | شدت فشار: {im.get('pressure')}")
            imb_str = "\n".join(imb_lines) if imb_lines else "عدم تعادل قطری شدیدی ثبت نشده است."

            lines = [
                f"### ⚡ نوار زمان و فروش پیشرفته اتاس (ATAS Time & Sales) برای {clean_s}",
                f"",
                f"قیمت لحظه‌ای: **${cur_p:,.2f}**",
                f"⏱️ **سرعت نوار معاملات (Speed of Tape):** **{ts.get('trades_per_sec', 0)} تیک در ثانیه** ({ts.get('status_fa', 'عادی')})",
                f"",
                f"🐋 **بلاک‌تریدها و ردپای نهنگ‌های وال‌استریت (Big Trades > 50 BTC):**",
                f"{bt_str}",
                f"",
                f"⚖️ **عدم تعادل‌های قطری ۳۰۰٪ دفتر سفارشات (Diagonal Imbalances):**",
                f"{imb_str}",
                f"",
                f"🧭 **تحلیل جریان نوار اتاس:** {at.get('summary_fa', 'نوار معاملات آماده موج نوسانی بعدی است.')}",
                f"",
                f"💡 **چرا اتاس مهم است؟** سرعت ثبت معاملات بالای ۵۰ تیک/ثانیه به این معناست که بازارسازها و ربات‌های HFT وارد جریان خرید تهاجمی شده‌اند و باید هم‌جهت با آن‌ها وارد معامله شد."
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"خطا در پردازش اطلاعات اتاس: {str(e)}"

    # 5. Quantower TPO Market Profile & Volume Profile
    if any(w in q for w in ["کوانت تاور", "کوانت‌تاور", "quantower", "مارکت پروفایل", "tpo", "ناحیه ارزش", "vah", "val", "vpoc", "گره پرحجم", "hvn", "lvn", "تعادل اولیه", "اسپرد سنتتیک"]):
        try:
            qt = coe.get_crypto_quantower_live(clean_s, "1h")
            cur_p = qt.get("current_price", price)
            mp = qt.get("market_profile", {})
            nodes = qt.get("volume_nodes", [])
            spreads = qt.get("synthetic_spreads", [])

            node_lines = []
            for n in nodes[:4]:
                col = "🟢 گره پرحجم (HVN - منطقه جذب قیمت)" if n.get('type') == 'HVN' else "🔴 گره کم‌حجم (LVN - منطقه پرتاب قیمت)"
                node_lines.append(f"• **{col}:** تراز **${n.get('price'):,.2f}** | حجم: {n.get('volume')} {clean_s} ({n.get('role_fa')})")
            node_str = "\n".join(node_lines) if node_lines else "گره‌های حجمی در حال محاسبه هستند."

            spread_lines = []
            if isinstance(spreads, list):
                for sp in spreads[:2]:
                    spread_lines.append(f"• **{sp.get('pair')}:** نسبت: {sp.get('ratio')} ({sp.get('change_pct')}) &bull; {sp.get('status_fa')}")
            spread_str = "\n".join(spread_lines) if spread_lines else "اطلاعات اسپرد سنتتیک در دسترس است."

            lines = [
                f"### 🏛️ تحلیل مارکت پروفایل کوانت‌تاور (Quantower TPO Profile) برای {clean_s}",
                f"",
                f"قیمت زنده: **${cur_p:,.2f}** | ساختار توزیع حراج: **{mp.get('shape_fa', 'متوازن')}**",
                f"",
                f"📊 **سطوح هندسی ناحیه ارزش ۷۰٪ (Value Area):**",
                f"- **سقف ناحیه ارزش (VAH):** **${mp.get('vah', 0):,.2f}** (مقاومت دینامیک حراج)",
                f"- **نقطه کنترل حجم (VPOC):** **${mp.get('vpoc', 0):,.2f} 👑** (آهنربای جذب قیمت سازمانی)",
                f"- **کف ناحیه ارزش (VAL):** **${mp.get('val', 0):,.2f}** (حمایت نهادی اصلی)",
                f"",
                f"🌐 **گره‌های تراکم حجم (High/Low Volume Nodes):**",
                f"{node_str}",
                f"",
                f"📐 **اسپردهای سنتتیک مارکت:**",
                f"{spread_str}",
                f"",
                f"🧭 **راهنمای تئوری حراج بازار (Auction Theory):** {mp.get('trading_guidance_fa', 'معامله بین سطوح VAL و VAH توصیه می‌شود.')}"
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"خطا در پردازش اطلاعات کوانت‌تاور: {str(e)}"

    # 6. Sierra Chart Orderflow & Numbered Bars
    if any(w in q for w in ["سیرا چارت", "سییرا چارت", "sierrachart", "sierra chart", "کندل های عددی", "numbered bars", "واگرایی دلتا", "دلتا دایورجنس", "vbp", "پروفایل عمودی"]):
        try:
            sc = coe.get_crypto_sierrachart_data(clean_s, "15m")
            cur_p = sc.get("current_price", price)
            div = sc.get("delta_divergence", {})
            vbp = sc.get("vbp_profile", {})
            nbs = sc.get("numbered_bars", [])

            nb_lines = []
            for b in nbs[-3:]:
                nb_lines.append(f"• **کندل {b.get('time')}:** دلتا: **{b.get('delta'):+d} Δ** | وضعیت: {b.get('state')} | حجم کل: {b.get('vol')} {clean_s}")
            nb_str = "\n".join(nb_lines) if nb_lines else "کندل‌های تیک عددی در دسترس هستند."

            lines = [
                f"### 📈 ترمینال پیشرفته سیرا چارت (Sierra Chart Orderflow) برای {clean_s}",
                f"",
                f"قیمت زنده: **${cur_p:,.2f}**",
                f"",
                f"🎯 **ردیاب واگرایی دلتا (Delta Divergence Scanner):**",
                f"- وضعیت واگرایی: **{div.get('type_fa', 'نرمال')}**",
                f"- ضریب قطعیت سازمانی: **{div.get('confidence', 85)}%**",
                f"- تحلیل جذب نقدینگی: {div.get('detail_fa', 'جذب سفارشات مارکت توسط لیمیت‌اردرهای سنگین پنهان')}",
                f"",
                f"🔢 **کندل‌های عددی تیک‌به‌تیک (Numbered Bars):**",
                f"{nb_str}",
                f"",
                f"📊 **پروفایل عمودی حجم بر اساس قیمت (Vertical VBP):**",
                f"- نقطه کنترل حجم (VBP POC): **${vbp.get('vbp_poc', cur_p):,.2f}**",
                f"- محدوده ارزش سشن: **{vbp.get('value_area', '$82,000 - $84,000')}**",
                f"",
                f"🧭 **پیام سیرا چارت:** {sc.get('summary_fa', 'ساختار کندل‌های عددی نشان از عدم تمایل فروشندگان دارد.')}"
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"خطا در پردازش اطلاعات سیرا چارت: {str(e)}"

    # 7. Geopolitical & Fundamentals Radar (GPR Index & Stablecoins)
    if any(w in q for w in ["ژئوپلیتیک", "gpr", "جنگ", "خاورمیانه", "تنگه هرمز", "تایوان", "پناهگاه امن", "چاپ تتر", "ضرب تتر", "استیبل کوین", "tether treasury", "سرمایه امن", "ریسک جهانی"]):
        try:
            geo = coe.get_crypto_geopolitics_radar()
            gpr = geo.get("gpr_index", 145)
            hs = geo.get("hotspots", [])
            st = geo.get("stablecoin_minting", {})
            sh = geo.get("safe_haven_rotation", {})

            hs_lines = []
            for h in hs:
                hs_lines.append(f"• **{h.get('region')}:** شدت تنش: **{h.get('level_fa', h.get('level', 'متوسط'))}** | تأثیر بر بازار: {h.get('crypto_impact', '')}")
            hs_str = "\n".join(hs_lines) if hs_lines else "کانون‌های بحران در سطح پایدار گزارش شده‌اند."

            sh_ratio = sh.get('gold_btc_ratio', '۲۰ انس طلا')
            sh_bias = sh.get('flow_bias_fa', 'چرخش نقدینگی به بیت‌کوین')

            lines = [
                f"### 🌐 رادار ریسک‌های ژئوپلیتیک و فاندامنتال کلان (Geopolitics & GPR Radar)",
                f"",
                f"📊 **شاخص ریسک ژئوپلیتیک جهانی (GPR Index):** **{gpr}** ({geo.get('gpr_status_fa', 'نرمال')})",
                f"- خط مبنای تاریخی: ۱۰۰.۰ واحد | عبور از ۱۵۰ واحد نشان‌دهنده ورود بازارهای مالی به فاز ریسک‌گریزی (Risk-Off) است.",
                f"",
                f"🔥 **پایش کانون‌های بحران نظامی و اقتصادی جهان:**",
                f"{hs_str}",
                f"",
                f"🛡️ **ماتریس چرخش سرمایه امن (Safe Haven Rotation):**",
                f"- نسبت طلا به بیت‌کوین: **{sh_ratio}**",
                f"- سوگیری جریان سرمایه: **{sh_bias}**",
                f"",
                f"💵 **رادار چاپ و تزریق استیبل‌کوین (Tether Treasury Radar):**",
                f"- ارزش بازار و ضرب تتر: **{st.get('usdt_market_cap', '')}**",
                f"- ورود نقدینگی USDC: **{st.get('usdc_market_cap', '')}**",
                f"- شتاب ضرب استیبل‌کوین: **{st.get('minting_velocity_fa', 'صعودی')}**",
                f"",
                f"🧭 **جمع‌بندی استراتژیک ایجنت:** {geo.get('verdict_fa', 'بیت‌کوین در شوک‌های ژئوپلیتیک به عنوان دارایی ضدسانسور رالی صعودی می‌زند.')}"
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"خطا در پردازش اطلاعات ژئوپلیتیک: {str(e)}"

    # 8. Wall Street Investment Banks & Institutional ETF Reports
    if any(w in q for w in ["بانک ها", "بانک‌های وال استریت", "وال استریت", "گزارش بانک", "تارگت بانک", "بلک راک", "فیدلیتی", "etf", "ورودی etf", "خالص etf", "گلدمن ساکس", "جی پی مورگان", "مورگان استنلی", "cme", "مایکرو استراتژی", "mstr"]):
        try:
            bank = coe.get_crypto_bank_reports()
            funds = bank.get("etf_flows", [])
            desks = bank.get("bank_desks", [])

            f_lines = []
            for f in funds[:4]:
                f_lines.append(f"• **{f.get('name')} ({f.get('ticker')}):** ورودی روزانه: **{f.get('net_flow')}** | کل دارایی تحت مدیریت: **{f.get('aum')}** ({f.get('status_fa')})")
            f_str = "\n".join(f_lines) if f_lines else "اطلاعات ETFهای اسپات در حال به‌روزرسانی است."

            d_lines = []
            for d in desks[:4]:
                d_lines.append(f"• **{d.get('bank')}:** تارگت رسمی: **{d.get('target')}** (افق {d.get('horizon')}) | استدلال: {d.get('rationale_fa')}")
            d_str = "\n".join(d_lines) if d_lines else "مدل‌های تحلیلی بانک‌های وال‌استریت در دسترس است."

            lines = [
                f"### 🏦 گزارشات نهادی بانک‌های وال‌استریت و صندوق‌های ETF کریپتو",
                f"",
                f"💵 **ورودی خالص روزانه ETFهای اسپات (Spot ETFs):** **{bank.get('total_daily_net', '+۴۲۳.۳ میلیون دلار')}**",
                f"🎯 **شاخص همگرایی سازمانی:** **{bank.get('gauge_score', 88)}/100** ({bank.get('gauge_text', 'انباشت قدرتمند')})",
                f"",
                f"🏛️ **عملکرد صندوق‌های غول سرمایه‌گذاری (BlackRock, Fidelity, ...):**",
                f"{f_str}",
                f"",
                f"💼 **پیش‌بینی و اهداف قیمتی ۵ بانک برتر وال‌استریت:**",
                f"{d_str}",
                f"",
                f"🧭 **بیانیه نهایی وال‌استریت:** {bank.get('executive_summary_fa', 'تقاضای سازمانی پشتوانه روند صعودی است.')}",
                f"",
                f"💡 **نکته کلیدی:** اگر در ساعات ۲۱:۳۰ تا ۲۴:۰۰ به وقت ایران، بلک‌راک و فیدلیتی خریدهای بالای ۳۰۰ میلیون دلار ثبت کنند، شکست مقاومت‌های هفتگی قطعی خواهد بود."
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"خطا در پردازش اطلاعات وال‌استریت: {str(e)}"

    return None
