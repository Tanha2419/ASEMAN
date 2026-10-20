# -*- coding: utf-8 -*-
"""
CryptoChatOrderflowHelper: Specialized AI Consultant Engine for Order Flow,
Bookmap, NinjaTrader, ATAS, Quantower, Sierra Chart, Geopolitics, and Wall Street Reports.

Fail-Closed contract: every number printed here must come from a live engine
response. When an engine returns ok=False, the Persian error message of that
engine is shown instead of any fabricated fallback number.
"""

from typing import Optional
import crypto_orderflow_engine as coe


def _fail_msg(block: dict, label: str) -> str:
    if isinstance(block, dict) and block.get("message"):
        return str(block["message"])
    return f"دادهٔ زندهٔ {label} در دسترس نیست؛ عدد ساختگی نمایش داده نمی‌شود (Fail-Closed)."


def _ok(block) -> bool:
    return isinstance(block, dict) and block.get("ok") is True


def _px(v, dec: int = 2, suffix: str = "") -> str:
    if isinstance(v, (int, float)):
        return f"${v:,.{dec}f}{suffix}"
    return "نامشخص"


def _num(v, suffix: str = "") -> str:
    if isinstance(v, (int, float)):
        return f"{v:,.2f}{suffix}"
    return "نامشخص"


def _txt(block: dict, key: str, default: str = "نامشخص") -> str:
    val = block.get(key) if isinstance(block, dict) else None
    return str(val) if val not in (None, "", "-") else default


def handle_orderflow_chat(q: str, clean_s: str, price: float) -> Optional[str]:
    q = q.lower().strip()
    price_ok = isinstance(price, (int, float)) and price > 0

    # 0. Specialized Deep Swing Trading Intelligence & Methodology
    if any(w in q for w in ["سوئینگ", "سوئینگ تریدینگ", "سیگنال سوئینگ", "سوئینگ از چی میاد", "از چه چیزی میاد", "بررسی سوئینگ", "چهار ساعته", "4 ساعته", "4h", "تفاوت اسکالپ و سوئینگ", "اختلاف اسکالپ"]):
        try:
            import crypto_unified_signals as cus
            unif = cus.get_unified_signals(clean_s) or {}
            sw = unif.get("swing", {}) or {}
            sc = unif.get("scalp", {}) or {}
            align = unif.get("alignment_status", "نامشخص")
            align_note = unif.get("alignment_note", "")
            conflict = unif.get("conflict_engine", {}) or {}

            qt = coe.get_crypto_quantower_live(clean_s, "4h")
            qt_ok = _ok(qt)
            mp = qt.get("market_profile", {}) if qt_ok else {}

            bank = coe.get_crypto_bank_reports()
            geo = coe.get_crypto_geopolitics_radar()

            bank_line = (
                f"ورودی خالص روزانهٔ ETFهای اسپات: **{bank.get('total_daily_net')}**"
                if _ok(bank) and bank.get("total_daily_net")
                else _fail_msg(bank, "جریان ETF و گزارش بانک‌های وال‌استریت")
            )
            geo_line = (
                f"ارزش بازار و ضرب تتر: **{geo.get('stablecoin_minting', {}).get('usdt_market_cap', 'نامشخص')}**"
                if _ok(geo)
                else _fail_msg(geo, "خزانهٔ تتر و رادار ژئوپلیتیک")
            )
            value_area_line = (
                f"قیمت در کف ارزش (VAL: ${mp.get('val'):,.1f}) یا سقف ارزش (VAH: ${mp.get('vah'):,.1f}) قرار دارد."
                if qt_ok and isinstance(mp.get("val"), (int, float)) and isinstance(mp.get("vah"), (int, float))
                else "دادهٔ پروفایل TPO ۴ ساعته در دسترس نیست (Fail-Closed)."
            )

            return f"""### 🏛️ تحلیل جامع و فوق‌تخصصی سوئینگ تریدینگ (۴ ساعته) برای {clean_s}

#### ❓ سیگنال سوئینگ تریدینگ از چه چیزی می‌آید و خاستگاه آن چیست؟
سیگنال سوئینگ تریدینگ (Swing Trading) بر خلاف اسکالپ که به نوسانات چند دقیقه‌ای وابسته است، از **تجمیع ۶ مؤلفه ساختاری و کلان ۴ ساعته و چندروزه** سرچشمه می‌گیرد:
1. **مارکت پروفایل ۴ ساعته کوانت‌تاور (Quantower 4H TPO):** {value_area_line}
2. **عدم تعادل اردر فلو و دلتای تجمیعی:** داده از کندل‌های واقعی بایننس محاسبه می‌شود و وضعیت آن در بخش Footprint همین چت قابل مشاهده است.
3. **جریان نقدینگی صندوق‌های ETF اسپات:** {bank_line}
4. **مدل‌های اجماع بانک‌های وال‌استریت:** تا اتصال فید معتبر، هیچ تارگت بانکی به‌صورت عددی ارائه نمی‌شود.
5. **رادار ژئوپلیتیک و خزانه‌داری تتر:** {geo_line}
6. **میانگین وزنی سشن‌های ۴ ساعته (4H VWAP & Trend):** در بخش نینجاتریدر با کلیدهای واقعی `mid/upper_1sd/lower_1sd` نمایش داده می‌شود.

---

#### 🎯 مشخصات دقیق سیگنال سوئینگ زنده {clean_s} (تایم‌فریم ۴ ساعته):
• **جهت معامله سوئینگ:** {sw.get('direction_emoji', '⏸️')} **{sw.get('direction', 'نامشخص')}**
• **نقطه ورود بهینه (Entry):** <code>{sw.get('entry_fmt', '—')}</code>
• **حد ضرر ساختاری (SL):** <code>{sw.get('stop_loss_fmt', '—')}</code>
• **تارگت اول میان‌مدت (TP1):** <code>{sw.get('tp1_fmt', '—')}</code>
• **تارگت دوم ماکرو (TP2):** <code>{sw.get('tp2_fmt', '—')}</code>
• **مدت زمان نگهداری (Holding Horizon):** <code>{sw.get('holding_duration', '—')}</code>
• **ساعت و تاریخ صدور:** <code>{sw.get('date_tehran', '—')} ساعت {sw.get('time_tehran', '—')} (ایران 🇮🇷)</code>
• **کامل بودن داده‌های ورودی:** **{conflict.get('badge', 'نامشخص')}** ({conflict.get('conflict_count', 'نامشخص')} ابزار ناهم‌سو)

---

#### ⚖️ تطبیق و هم‌راستایی با سیگنال اسکالپ (Scalp vs Swing):
• **وضعیت همسویی:** **{align}**
• **تحلیل تطبیقی:** {align_note or 'نامشخص'}
• **سیگنال اسکالپ همزمان (۱۵ دقیقه‌ای):** {sc.get('direction_emoji', '⏸️')} <b>{sc.get('direction', 'نامشخص')}</b> در <code>{sc.get('entry_fmt', '—')}</code> (با افق زمانی {sc.get('holding_duration', '—')})
• **راهبرد عملیاتی تریدر:** اگر اسکالپ با سوئینگ هم‌جهت باشد، با حجم کامل و حداکثر اعتماد وارد شوید؛ اگر اسکالپ خلاف سوئینگ باشد، آن معامله صرفاً یک نوسان اصلاحی سریع و موقت است. اگر یکی از دو سیگنال NO_TRADE باشد، طبق اصول Fail-Closed هیچ معامله‌ای پیشنهاد نمی‌شود.
"""
        except Exception as e:
            return f"خطا در استخراج تحلیل سوئینگ: {e}"

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

            mp = qt.get("market_profile", {}) if _ok(qt) else {}
            cvd = nt.get("cvd", {}) if _ok(nt) else {}
            ts = at.get("tape_speed", {}) if _ok(at) else {}
            div = sc.get("delta_divergence", {}) if _ok(sc) else {}

            bm_txt = (
                f"- نسبت قدرت دیواره‌ها: **{bm.get('imbalance_pct')}% به سود خریداران** | {bm.get('verdict_fa', '')}\n"
                f"- تعداد سفارش‌های آیس‌برگ (برآوردی از دیواره‌های واقعی): **{len(bm.get('iceberg_orders', []))} مورد**"
                if _ok(bm) else f"- {_fail_msg(bm, 'بوک‌مپ')}"
            )
            nt_txt = (
                f"- دلتای تجمیعی (CVD): **{cvd.get('status_fa', 'نامشخص')}**\n"
                f"- خط مرکزی VWAP: **{_px(nt.get('vwap', {}).get('mid'))}** | باند +1SD: **{_px(nt.get('vwap', {}).get('upper_1sd'))}**"
                if _ok(nt) else f"- {_fail_msg(nt, 'نینجاتریدر')}"
            )
            at_txt = (
                f"- سرعت نوار معاملات: **{ts.get('trades_per_sec', 'نامشخص')} تیک در ثانیه** ({ts.get('status_fa', 'نامشخص')})\n"
                f"- بلاک‌تریدهای واقعی ثبت‌شده: **{len(at.get('big_trades', []))} مورد**"
                if _ok(at) else f"- {_fail_msg(at, 'اتاس')}"
            )
            qt_txt = (
                f"- محدوده ارزش ۷۰٪: کف (VAL): **{_px(mp.get('val'))}** | سقف (VAH): **{_px(mp.get('vah'))}**\n"
                f"- نقطه کنترل حجم (VPOC): **{_px(mp.get('vpoc'))}** ({mp.get('shape_fa', 'نامشخص')})"
                if _ok(qt) else f"- {_fail_msg(qt, 'کوانت‌تاور')}"
            )
            sc_txt = (
                f"- واگرایی دلتا: **{div.get('type_fa', 'نامشخص')}** با ضریب قطعیت **{div.get('confidence', 'نامشخص')}%**"
                if _ok(sc) else f"- {_fail_msg(sc, 'سیرا چارت')}"
            )
            geo_txt = (
                f"- شاخص ریسک جهانی (GPR): **{geo.get('gpr_index', 'نامشخص')}** ({geo.get('gpr_status_fa', 'نامشخص')})"
                if _ok(geo) else f"- {_fail_msg(geo, 'رادار ژئوپلیتیک')}"
            )
            bank_txt = (
                f"- ورودی خالص روزانه ETFهای اسپات: **{bank.get('total_daily_net')}**\n"
                f"- ارزیابی مؤسسات: **{bank.get('executive_summary_fa', 'نامشخص')}**"
                if _ok(bank) else f"- {_fail_msg(bank, 'جریان ETF و گزارش بانکی')}"
            )

            lines = [
                f"### 🏛️ گزارش تلفیقی اتاق فرمان اردر فلو وال‌استریت (Order Flow Master Intelligence) برای {clean_s}",
                f"",
                f"قیمت زنده: **{_px(price) if price_ok else 'نامشخص'}**",
                f"",
                f"#### ۱. 🔥 بوک‌مپ نقدینگی (Bookmap L2/L3):",
                bm_txt,
                f"",
                f"#### ۲. 🎯 نینجاتریدر ۸ (NinjaTrader 8 Footprint):",
                nt_txt,
                f"",
                f"#### ۳. ⚡ اتاس (ATAS Time & Sales):",
                at_txt,
                f"",
                f"#### ۴. 🏛️ کوانت‌تاور (Quantower TPO Profile):",
                qt_txt,
                f"",
                f"#### ۵. 📈 سیرا چارت (Sierra Chart Orderflow):",
                sc_txt,
                f"",
                f"#### ۶. 🌐 رادار ریسک ژئوپلیتیک & استیبل‌کوین:",
                geo_txt,
                f"",
                f"#### ۷. 🏦 گزارشات بانک‌های وال‌استریت & ETFها:",
                bank_txt,
                f"",
                f"🧭 **جمع‌بندی ایجنت:** تنها ابزارهایی که پاسخ `ok=True` داده‌اند در تصمیم‌گیری لحاظ می‌شوند. اگر بخشی «در دسترس نیست» گزارش شده باشد، هیچ نتیجه‌گیری از آن گرفته نمی‌شود (Fail-Closed)."
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"خطا در واکشی داده‌های تلفیقی اردر فلو: {str(e)}"

    # 2. Bookmap L2/L3 Liquidity Heatmap & Iceberg Detector
    if any(w in q for w in ["بوک مپ", "بوکمپ", "bookmap", "آیسبرگ", "آیس برگ", "iceberg", "دیواره نقدینگی", "دیواره سفارشات", "دیوار خرید", "دیوار فروش", "هیت مپ لول 3", "هیت‌مپ بوک‌مپ", "اوردرهای مخفی"]):
        try:
            bm = coe.get_crypto_bookmap_data(clean_s, "15m")
            if not _ok(bm):
                return f"### 🔥 بوک‌مپ (Bookmap L2/L3)\n\n{_fail_msg(bm, 'بوک‌مپ')}"
            cur_p = bm.get("current_price", price if price_ok else None)
            icebergs = bm.get("iceberg_orders", [])
            asks = bm.get("ask_levels", [])
            bids = bm.get("bid_levels", [])

            ice_lines = []
            for ic in icebergs[:4]:
                ice_lines.append(
                    f"• **{_px(ic.get('price'))}:** حجم پنهان **برآوردی**: **{_num(ic.get('estimated_hidden_vol'))} {clean_s}** "
                    f"({ic.get('direction_fa', 'نامشخص')}) | وضعیت: {ic.get('status', 'نامشخص')}"
                )
            ice_str = "\n".join(ice_lines) if ice_lines else "• در حال حاضر دیوارهٔ واقعی با حجم پنهان قابل‌تخمین در دفتر سفارشات بایننس یافت نشده است."

            ask_w = asks[0] if asks else {}
            bid_w = bids[0] if bids else {}

            return (
                f"### 🔥 کالبدشکافی امواج نقدینگی بوک‌مپ (Bookmap L2/L3) برای {clean_s}\n\n"
                f"قیمت زنده: **{_px(cur_p)}** | نسبت نقدینگی دفتر سفارشات: **{bm.get('imbalance_pct', 'نامشخص')}% خریدار**\n\n"
                f"🧱 **مهم‌ترین دیواره‌های نقدینگی واقعی (Resting Liquidity Walls):**\n"
                f"- **دیواره عرضه (Ask Wall):** نرخ **{_px(ask_w.get('price'))}** با حجم **{_num(ask_w.get('volume_lots'))} {clean_s}** (سهم: {ask_w.get('share_pct', 'نامشخص')}%)\n"
                f"- **دیواره تقاضا (Bid Wall):** نرخ **{_px(bid_w.get('price'))}** با حجم **{_num(bid_w.get('volume_lots'))} {clean_s}** (سهم: {bid_w.get('share_pct', 'نامشخص')}%)\n\n"
                f"🧊 **سفارش‌های پنهان (تخمینی — نه اندازه‌گیری زنده):**\n{ice_str}\n\n"
                f"🧭 **تحلیل سازمانی بوک‌مپ:** {bm.get('verdict_fa', 'نامشخص')}\n\n"
                f"💡 **راهنمای عمل تریدر:** تنها در صورت وجود دیوارهٔ خرید واقعی در دفتر سفارشات، می‌توان آن را به‌عنوان کف حمایتی در نظر گرفت؛ در غیر این صورت ورود بر پایهٔ سطح ناموجود ممنوع است."
            )
        except Exception as e:
            return f"خطا در پردازش اطلاعات بوک‌مپ: {str(e)}"

    # 3. NinjaTrader 8 Footprint Ladder & SuperDOM
    if any(w in q for w in ["نینجا", "نینجاتریدر", "ninjatrader", "فوت پرینت", "فوت‌پرینت", "footprint", "سوپردام", "superdom", "کلاستر", "تاج طلایی", "poc کلاستر", "باندهای vwap", "انحراف معیار vwap"]):
        try:
            nt = coe.get_crypto_ninjatrader_live(clean_s, "15m")
            if not _ok(nt):
                return f"### 🎯 فوت‌پرینت نینجاتریدر ۸\n\n{_fail_msg(nt, 'نینجاتریدر')}"
            cur_p = nt.get("current_price", price if price_ok else None)
            vwap = nt.get("vwap", {}) or {}
            cvd = nt.get("cvd", {}) or {}
            fps = nt.get("footprint_candles", [])

            fp_lines = []
            for c in fps[:3]:
                col = "🟢 صعودی" if c.get('is_bullish') else "🔴 نزولی"
                delta = c.get('candle_delta')
                delta_str = f"{delta:+d}" if isinstance(delta, int) else "نامشخص"
                fp_lines.append(
                    f"• **کندل {c.get('time', 'نامشخص')} ({col}):** تراز POC: **{_px(c.get('poc_price'))}** | "
                    f"دلتای خالص: **{delta_str} Δ** ({c.get('delta_pct', 'نامشخص')}%) | حجم کل: {_num(c.get('total_volume'))} {clean_s}"
                )
            fp_str = "\n".join(fp_lines) if fp_lines else "دادهٔ فوت‌پرینت کندلی در دسترس نیست (Fail-Closed)."
            vwap_available = bool(vwap.get("available"))

            return (
                f"### 🎯 ماتریس اردر فلو و فوت‌پرینت نینجاتریدر ۸ (NinjaTrader 8) برای {clean_s}\n\n"
                f"قیمت زنده: **{_px(cur_p)}** | وضعیت دلتای تجمیعی (CVD): **{cvd.get('status_fa', 'نامشخص')}**\n\n"
                f"📊 **کندل‌های فوت‌پرینت کلاستری (از کندل‌های واقعی بایننس):**\n{fp_str}\n\n"
                f"📐 **باندهای انحراف معیار VWAP:**\n"
                + (
                    f"- خط مرکزی: **{_px(vwap.get('mid'))}**\n"
                    f"- باند +1SD: **{_px(vwap.get('upper_1sd'))}** | باند -1SD: **{_px(vwap.get('lower_1sd'))}**\n"
                    f"- باند +2SD: **{_px(vwap.get('upper_2sd'))}** | باند -2SD: **{_px(vwap.get('lower_2sd'))}**\n"
                    if vwap_available else
                    "- دادهٔ کندلی کافی برای محاسبهٔ VWAP در دسترس نیست (Fail-Closed).\n"
                )
                + f"\n🧭 **استراتژی نینجاتریدر:** {nt.get('summary_fa', 'نامشخص')}\n\n"
                f"💡 **توصیه اجرایی:** عدم تعادل‌های فوت‌پرینت فقط زمانی معتبرند که از کندل‌های واقعی استخراج شده باشند؛ در این گزارش هیچ عدد نمایشیِ ساختگی رندر نمی‌شود."
            )
        except Exception as e:
            return f"خطا در پردازش فوت‌پرینت نینجاتریدر: {str(e)}"

    # 4. ATAS Advanced Time & Sales / Big Trades
    if any(w in q for w in ["اتاس", "atas", "تایم اند سیلز", "تایم اند سيلز", "نوار معاملات", "بلاک ترید", "big trade", "معاملات بزرگ", "سرعت نوار", "عدم تعادل قطری", "عدم تعادل ۳۰۰"]):
        try:
            at = coe.get_crypto_atas_live(clean_s)
            if not _ok(at):
                return f"### ⚡ نوار زمان و فروش اتاس (ATAS)\n\n{_fail_msg(at, 'اتاس')}"
            cur_p = at.get("current_price", price if price_ok else None)
            ts = at.get("tape_speed", {}) or {}
            bts = at.get("big_trades", [])
            imbs = at.get("diagonal_imbalances", [])

            bt_lines = []
            for b in bts[:4]:
                col = "🟢 خرید مارکت" if b.get('side') == 'BUY' else "🔴 فروش مارکت"
                size = b.get('size_lots', b.get('volume_lots'))
                bt_lines.append(
                    f"• **ساعت {b.get('time', 'نامشخص')}:** {col} **{_num(size)} {clean_s}** ({b.get('value_usd', 'نامشخص')}) "
                    f"روی نرخ {_px(b.get('price'))} ({b.get('exchange', 'نامشخص')})"
                )
            bt_str = "\n".join(bt_lines) if bt_lines else "بلاک‌ترید واقعی در نمونهٔ معاملات اخیر بایننس ثبت نشده است."

            imb_lines = []
            for im in imbs[:3]:
                imb_lines.append(f"• تراز **{_px(im.get('price_level'))}:** عدم تعادل **{im.get('ratio', 'نامشخص')}x** ({im.get('side_fa', 'نامشخص')}) | شدت: {im.get('pressure', 'نامشخص')}")
            imb_str = "\n".join(imb_lines) if imb_lines else "عدم تعادل قطری از تیک‌دادهٔ زندهٔ کافی قابل استخراج نیست (Fail-Closed)."

            return (
                f"### ⚡ نوار زمان و فروش پیشرفته اتاس (ATAS Time & Sales) برای {clean_s}\n\n"
                f"قیمت لحظه‌ای: **{_px(cur_p)}**\n"
                f"⏱️ **سرعت نوار معاملات:** **{ts.get('trades_per_sec', 'نامشخص')} تیک در ثانیه** ({ts.get('status_fa', 'نامشخص')})\n\n"
                f"🐋 **بلاک‌تریدهای واقعی (از aggTrades بایننس):**\n{bt_str}\n\n"
                f"⚖️ **عدم تعادل‌های قطری:**\n{imb_str}\n\n"
                f"🧭 **تحلیل جریان نوار اتاس:** {at.get('summary_fa', 'نامشخص')}\n\n"
                f"💡 **چرا اتاس مهم است؟** سرعت و اندازهٔ معاملات واقعی، شدت تهاجم خریداران/فروشندگان را نشان می‌دهد؛ در این گزارش فقط تیک‌دادهٔ واقعی بایننس منعکس می‌شود."
            )
        except Exception as e:
            return f"خطا در پردازش اطلاعات اتاس: {str(e)}"

    # 5. Quantower TPO Market Profile & Volume Profile
    if any(w in q for w in ["کوانت تاور", "کوانت‌تاور", "quantower", "مارکت پروفایل", "tpo", "ناحیه ارزش", "vah", "val", "vpoc", "گره پرحجم", "hvn", "lvn", "تعادل اولیه", "اسپرد سنتتیک"]):
        try:
            qt = coe.get_crypto_quantower_live(clean_s, "1h")
            if not _ok(qt):
                return f"### 🏛️ مارکت پروفایل کوانت‌تاور (Quantower TPO)\n\n{_fail_msg(qt, 'کوانت‌تاور')}"
            cur_p = qt.get("current_price", price if price_ok else None)
            mp = qt.get("market_profile", {}) or {}
            spreads = qt.get("synthetic_spreads", [])

            spread_lines = []
            if isinstance(spreads, list):
                for sp in spreads[:2]:
                    spread_lines.append(f"• **{sp.get('pair', 'نامشخص')}:** نسبت: {sp.get('ratio', 'نامشخص')} ({sp.get('change_pct', 'نامشخص')}) &bull; {sp.get('status_fa', 'نامشخص')}")
            spread_str = "\n".join(spread_lines) if spread_lines else qt.get("spreads_note", "اطلاعات اسپرد سنتتیک در دسترس نیست (Fail-Closed).")

            return (
                f"### 🏛️ تحلیل مارکت پروفایل کوانت‌تاور (Quantower TPO Profile) برای {clean_s}\n\n"
                f"قیمت زنده: **{_px(cur_p)}** | ساختار توزیع حراج: **{mp.get('shape_fa', 'نامشخص')}**\n\n"
                f"📊 **سطوح ناحیه ارزش ۷۰٪ (از کندل‌های واقعی {qt.get('timeframe', '1h')}):**\n"
                f"- **سقف ناحیه ارزش (VAH):** **{_px(mp.get('vah'))}**\n"
                f"- **نقطه کنترل حجم (VPOC):** **{_px(mp.get('vpoc'))}**\n"
                f"- **کف ناحیه ارزش (VAL):** **{_px(mp.get('val'))}**\n\n"
                f"📐 **اسپردهای سنتتیک مارکت:**\n{spread_str}\n\n"
                f"🧭 **راهنمای تئوری حراج بازار:** {mp.get('trading_guidance_fa', 'نامشخص')}"
            )
        except Exception as e:
            return f"خطا در پردازش اطلاعات کوانت‌تاور: {str(e)}"

    # 6. Sierra Chart Orderflow & Numbered Bars
    if any(w in q for w in ["سیرا چارت", "سییرا چارت", "sierrachart", "sierra chart", "کندل های عددی", "numbered bars", "واگرایی دلتا", "دلتا دایورجنس", "vbp", "پروفایل عمودی"]):
        try:
            sc = coe.get_crypto_sierrachart_data(clean_s, "15m")
            if not _ok(sc):
                return f"### 📈 سیرا چارت (Sierra Chart Orderflow)\n\n{_fail_msg(sc, 'سیرا چارت')}"
            cur_p = sc.get("current_price", price if price_ok else None)
            div = sc.get("delta_divergence", {}) or {}
            vbp = sc.get("vbp_profile", {}) or {}
            nbs = sc.get("numbered_bars", [])

            nb_lines = []
            for b in nbs[-3:]:
                delta = b.get('delta')
                delta_str = f"{delta:+d}" if isinstance(delta, int) else "نامشخص"
                nb_lines.append(f"• **کندل {b.get('time', 'نامشخص')}:** دلتا: **{delta_str} Δ** | وضعیت: {b.get('state', 'نامشخص')} | حجم کل: {_num(b.get('vol'))} {clean_s}")
            nb_str = "\n".join(nb_lines) if nb_lines else "کندل عددی کافی در دسترس نیست (Fail-Closed)."

            return (
                f"### 📈 ترمینال پیشرفته سیرا چارت (Sierra Chart Orderflow) برای {clean_s}\n\n"
                f"قیمت زنده: **{_px(cur_p)}**\n\n"
                f"🎯 **ردیاب واگرایی دلتا (از کندل‌های واقعی):**\n"
                f"- وضعیت واگرایی: **{div.get('type_fa', 'نامشخص')}**\n"
                f"- ضریب قطعیت: **{div.get('confidence', 'نامشخص')}%**\n\n"
                f"🔢 **کندل‌های عددی (Numbered Bars):**\n{nb_str}\n\n"
                f"📊 **پروفایل عمودی حجم (VBP):**\n"
                f"- نقطه کنترل حجم (VBP POC): **{_px(vbp.get('vbp_poc'))}**\n"
                f"- محدوده ارزش سشن: **{vbp.get('value_area', 'نامشخص')}**\n\n"
                f"🧭 **پیام سیرا چارت:** {sc.get('summary_fa', 'نامشخص')}"
            )
        except Exception as e:
            return f"خطا در پردازش اطلاعات سیرا چارت: {str(e)}"

    # 7. Geopolitical & Fundamentals Radar (GPR Index & Stablecoins)
    if any(w in q for w in ["ژئوپلیتیک", "gpr", "جنگ", "خاورمیانه", "تنگه هرمز", "تایوان", "پناهگاه امن", "چاپ تتر", "ضرب تتر", "استیبل کوین", "tether treasury", "سرمایه امن", "ریسک جهانی"]):
        try:
            geo = coe.get_crypto_geopolitics_radar()
            if not _ok(geo):
                return (
                    f"### 🌐 رادار ریسک‌های ژئوپلیتیک و فاندامنتال کلان (Geopolitics & GPR Radar)\n\n"
                    f"⛔ {_fail_msg(geo, 'شاخص GPR و خزانهٔ استیبل‌کوین')}\n\n"
                    f"📌 **توضیح آموزشی:** شاخص GPR یک سنجهٔ پژوهشی است که تنش‌های ژئوپلیتیک را می‌سنجد و خط مبنای تاریخی آن ۱۰۰ واحد است؛ "
                    f"اما تا اتصال یک فید معتبر، هیچ مقدار زنده یا کانون بحرانی ساختگی در این بخش نمایش داده نمی‌شود."
                )

            gpr = geo.get("gpr_index")
            hs = geo.get("hotspots", [])
            st = geo.get("stablecoin_minting", {})
            sh = geo.get("safe_haven_rotation", {})
            hs_lines = [f"• **{h.get('region', 'نامشخص')}:** شدت تنش: **{h.get('level_fa', h.get('level', 'نامشخص'))}** | تأثیر: {h.get('crypto_impact', 'نامشخص')}" for h in hs]
            hs_str = "\n".join(hs_lines) if hs_lines else "کانون بحرانی از فید زنده گزارش نشده است."

            return (
                f"### 🌐 رادار ریسک‌های ژئوپلیتیک و فاندامنتال کلان (Geopolitics & GPR Radar)\n\n"
                f"📊 **شاخص ریسک ژئوپلیتیک جهانی (GPR Index):** **{gpr if gpr is not None else 'نامشخص'}** ({geo.get('gpr_status_fa', 'نامشخص')})\n\n"
                f"🔥 **کانون‌های بحران:**\n{hs_str}\n\n"
                f"🛡️ **چرخش سرمایه امن:**\n"
                f"- نسبت طلا به بیت‌کوین: **{sh.get('gold_btc_ratio', 'نامشخص')}**\n"
                f"- سوگیری جریان: **{sh.get('flow_bias_fa', 'نامشخص')}**\n\n"
                f"💵 **خزانهٔ استیبل‌کوین:**\n"
                f"- ارزش بازار تتر: **{st.get('usdt_market_cap', 'نامشخص')}**\n"
                f"- ارزش بازار USDC: **{st.get('usdc_market_cap', 'نامشخص')}**\n"
                f"- شتاب ضرب: **{st.get('minting_velocity_fa', 'نامشخص')}**\n\n"
                f"🧭 **جمع‌بندی:** {geo.get('verdict_fa', 'نامشخص')}"
            )
        except Exception as e:
            return f"خطا در پردازش اطلاعات ژئوپلیتیک: {str(e)}"

    # 8. Wall Street Investment Banks & Institutional ETF Reports
    if any(w in q for w in ["بانک ها", "بانک‌های وال استریت", "وال استریت", "گزارش بانک", "تارگت بانک", "بلک راک", "فیدلیتی", "etf", "ورودی etf", "خالص etf", "گلدمن ساکس", "جی پی مورگان", "مورگان استنلی", "cme", "مایکرو استراتژی", "mstr"]):
        try:
            bank = coe.get_crypto_bank_reports()
            if not _ok(bank):
                return (
                    f"### 🏦 گزارشات نهادی بانک‌های وال‌استریت و صندوق‌های ETF کریپتو\n\n"
                    f"⛔ {_fail_msg(bank, 'جریان ETF و گزارش میزهای بانکی')}\n\n"
                    f"📌 **توضیح آموزشی:** تارگت‌های بانکی و جریان روزانهٔ ETF تنها با فید رسمی قابل استناد هستند؛ "
                    f"تا اتصال آن، هیچ عدد یا تارگتی در این بخش ارائه نمی‌شود (Fail-Closed)."
                )

            funds = bank.get("etf_flows", [])
            desks = bank.get("bank_desks", [])
            f_lines = [f"• **{f.get('name', 'نامشخص')} ({f.get('ticker', 'نامشخص')}):** ورودی روزانه: **{f.get('net_flow', 'نامشخص')}** | AUM: **{f.get('aum', 'نامشخص')}** ({f.get('status_fa', 'نامشخص')})" for f in funds[:4]]
            d_lines = [f"• **{d.get('bank', 'نامشخص')}:** تارگت: **{d.get('target', 'نامشخص')}** (افق {d.get('horizon', 'نامشخص')}) | {d.get('rationale_fa', 'نامشخص')}" for d in desks[:4]]
            f_str = "\n".join(f_lines) if f_lines else "دادهٔ صندوقی از فید زنده گزارش نشده است."
            d_str = "\n".join(d_lines) if d_lines else "تارگت بانکی از فید زنده گزارش نشده است."

            return (
                f"### 🏦 گزارشات نهادی بانک‌های وال‌استریت و صندوق‌های ETF کریپتو\n\n"
                f"💵 **ورودی خالص روزانه ETFهای اسپات:** **{bank.get('total_daily_net', 'نامشخص')}**\n"
                f"🎯 **شاخص همگرایی سازمانی:** **{bank.get('gauge_score', 'نامشخص')}/100** ({bank.get('gauge_text', 'نامشخص')})\n\n"
                f"🏛️ **صندوق‌های سرمایه‌گذاری:**\n{f_str}\n\n"
                f"💼 **اهداف قیمتی میزهای بانکی:**\n{d_str}\n\n"
                f"🧭 **بیانیهٔ نهایی:** {bank.get('executive_summary_fa', 'نامشخص')}"
            )
        except Exception as e:
            return f"خطا در پردازش اطلاعات وال‌استریت: {str(e)}"

    return None
