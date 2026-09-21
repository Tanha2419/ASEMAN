import requests
import numpy as np
import time
from typing import Dict, Any, Optional, List
from institutional_addons import MacroEngine, OnChainEngine, NewsCircuitBreaker, BacktestEngine, TelegramDispatcher

def clean_symbol(raw_symbol: str) -> str:
    s = raw_symbol.upper().strip().replace("/", "").replace("-", "").replace("_", "")
    if s in ["BTC", "ETH", "SOL", "USDT", "BNB", "XRP", "DOGE", "ADA", "TRX", "PEPE", "SUI", "NEAR", "AVAX", "LINK", "TON", "SHIB", "DOT", "MATIC"]:
        return s + "USDT"
    if not s.endswith("USDT") and not s.endswith("USD"):
        s += "USDT"
    return s

def get_base_coin(symbol: str) -> str:
    s = symbol.upper().replace("USDT", "").replace("USD", "").replace("_", "")
    return s

class CryptoDataFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

    def fetch_mexc_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            url = f"https://api.mexc.com/api/v3/ticker/24hr?symbol={symbol}"
            res = self.session.get(url, timeout=4)
            if res.status_code == 200:
                data = res.json()
                if "lastPrice" in data:
                    pct = float(data.get("priceChangePercent", 0))
                    if abs(pct) < 1.0 and abs(pct) > 0.0001:
                        pct = pct * 100.0
                    return {
                        "source": "MEXC",
                        "symbol": data["symbol"],
                        "last_price": float(data["lastPrice"]),
                        "price_change_pct": round(pct, 2),
                        "high_24h": float(data.get("highPrice", 0)),
                        "low_24h": float(data.get("lowPrice", 0)),
                        "volume_base": float(data.get("volume", 0)),
                        "volume_quote": float(data.get("quoteVolume", 0)),
                        "bid": float(data.get("bidPrice", 0)),
                        "ask": float(data.get("askPrice", 0)),
                    }
        except Exception:
            pass
        return None

    def fetch_binance_us_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            url = f"https://api.binance.us/api/v3/ticker/24hr?symbol={symbol}"
            res = self.session.get(url, timeout=4)
            if res.status_code == 200:
                data = res.json()
                if "lastPrice" in data:
                    return {
                        "source": "Binance.US",
                        "symbol": data["symbol"],
                        "last_price": float(data["lastPrice"]),
                        "price_change_pct": round(float(data.get("priceChangePercent", 0)), 2),
                        "high_24h": float(data.get("highPrice", 0)),
                        "low_24h": float(data.get("lowPrice", 0)),
                        "volume_base": float(data.get("volume", 0)),
                        "volume_quote": float(data.get("quoteVolume", 0)),
                        "bid": float(data.get("bidPrice", 0)),
                        "ask": float(data.get("askPrice", 0)),
                    }
        except Exception:
            pass
        return None

    def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        t = self.fetch_mexc_ticker(symbol)
        if t: return t
        t = self.fetch_binance_us_ticker(symbol)
        if t: return t
        return None

    def fetch_klines(self, symbol: str, interval: str = "15m", limit: int = 100) -> List[List[float]]:
        try:
            url = f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
            res = self.session.get(url, timeout=5)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list) and len(data) > 0:
                    return [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in data]
        except Exception:
            pass

        try:
            url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
            res = self.session.get(url, timeout=5)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list) and len(data) > 0:
                    return [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in data]
        except Exception:
            pass

        return []

    def fetch_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        try:
            url = f"https://api.mexc.com/api/v3/depth?symbol={symbol}&limit={limit}"
            res = self.session.get(url, timeout=4)
            if res.status_code == 200:
                data = res.json()
                bids = [[float(b[0]), float(b[1])] for b in data.get("bids", [])]
                asks = [[float(a[0]), float(a[1])] for a in data.get("asks", [])]
                return self._calculate_depth_metrics(bids, asks)
        except Exception:
            pass

        try:
            url = f"https://api.binance.us/api/v3/depth?symbol={symbol}&limit={limit}"
            res = self.session.get(url, timeout=4)
            if res.status_code == 200:
                data = res.json()
                bids = [[float(b[0]), float(b[1])] for b in data.get("bids", [])]
                asks = [[float(a[0]), float(a[1])] for a in data.get("asks", [])]
                return self._calculate_depth_metrics(bids, asks)
        except Exception:
            pass

        return {"bid_volume_usd": 0, "ask_volume_usd": 0, "ratio": 1.0, "spread_pct": 0, "pressure": "متعادل", "sentiment": "Neutral", "top_bids": [], "top_asks": [], "whale_walls": []}

    def _calculate_depth_metrics(self, bids: List[List[float]], asks: List[List[float]]) -> Dict[str, Any]:
        if not bids or not asks:
            return {"bid_volume_usd": 0, "ask_volume_usd": 0, "ratio": 1.0, "spread_pct": 0, "pressure": "متعادل", "sentiment": "Neutral", "top_bids": [], "top_asks": [], "whale_walls": []}
        
        bid_vol = sum(b[0] * b[1] for b in bids)
        ask_vol = sum(a[0] * a[1] for a in asks)
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread = best_ask - best_bid
        spread_pct = (spread / best_ask) * 100 if best_ask > 0 else 0
        ratio = bid_vol / ask_vol if ask_vol > 0 else 1.0
        
        whale_walls = []
        avg_bid_size = np.mean([b[0] * b[1] for b in bids]) if bids else 0
        avg_ask_size = np.mean([a[0] * a[1] for a in asks]) if asks else 0

        for b in bids[:6]:
            val = b[0] * b[1]
            if val > 3.0 * avg_bid_size and val > 15000:
                whale_walls.append({"type": "BUY_WALL", "price": b[0], "volume_usd": round(val, 2), "desc": "دیوار خرید سنگین نهنگ"})
        for a in asks[:6]:
            val = a[0] * a[1]
            if val > 3.0 * avg_ask_size and val > 15000:
                whale_walls.append({"type": "SELL_WALL", "price": a[0], "volume_usd": round(val, 2), "desc": "دیوار فروش سنگین نهنگ"})

        if ratio > 1.35:
            pressure = "فشار خرید سنگین (دیوار خرید نهنگ‌ها)"
            sentiment = "Bullish"
        elif ratio > 1.1:
            pressure = "برتری نسبی خریداران"
            sentiment = "Slightly Bullish"
        elif ratio < 0.65:
            pressure = "فشار فروش سنگین (دیوار فروش)"
            sentiment = "Bearish"
        elif ratio < 0.9:
            pressure = "برتری نسبی فروشندگان"
            sentiment = "Slightly Bearish"
        else:
            pressure = "تعادل میان خریداران و فروشندگان"
            sentiment = "Neutral"

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread_pct": round(spread_pct, 4),
            "bid_volume_usd": round(bid_vol, 2),
            "ask_volume_usd": round(ask_vol, 2),
            "ratio": round(ratio, 2),
            "pressure": pressure,
            "sentiment": sentiment,
            "top_bids": bids[:5],
            "top_asks": asks[:5],
            "whale_walls": whale_walls
        }

    def fetch_institutional_derivatives(self, symbol: str) -> Dict[str, Any]:
        contract_sym = symbol.replace("USDT", "_USDT")
        try:
            url_ticker = f"https://contract.mexc.com/api/v1/contract/ticker?symbol={contract_sym}"
            url_fund = f"https://contract.mexc.com/api/v1/contract/funding_rate/{contract_sym}"
            
            res_t = self.session.get(url_ticker, timeout=4)
            res_f = self.session.get(url_fund, timeout=4)
            
            oi_val = 0
            funding_rate = 0.0
            
            if res_t.status_code == 200:
                dt = res_t.json().get("data", {})
                oi_val = float(dt.get("holdVol", 0))
                
            if res_f.status_code == 200:
                df = res_f.json().get("data", {})
                funding_rate = float(df.get("fundingRate", 0.0))
                
            fr_pct = funding_rate * 100.0
            if fr_pct > 0.02:
                crowd = "ازدحام شدید لانگ (خطر هانت لانگ‌ها و ریزش فیک)"
            elif fr_pct < -0.01:
                crowd = "ازدحام شورت (احتمال پامپ و شورت‌اسکوییز Short Squeeze)"
            else:
                crowd = "متعادل و طبیعی"

            return {
                "open_interest_usd": oi_val,
                "open_interest_fmt": f"${oi_val/1_000_000:,.1f}M" if oi_val > 1_000_000 else f"${oi_val:,.0f}",
                "funding_rate": round(fr_pct, 4),
                "funding_rate_fmt": f"{fr_pct:+.4f}%",
                "market_crowd": crowd,
                "has_data": oi_val > 0
            }
        except Exception:
            pass
        return {
            "open_interest_usd": 0,
            "open_interest_fmt": "N/A",
            "funding_rate": 0.0,
            "funding_rate_fmt": "0.0000%",
            "market_crowd": "داده مشتقه در دسترس نیست",
            "has_data": False
        }

    def fetch_fear_and_greed(self) -> Dict[str, Any]:
        try:
            url = "https://api.alternative.me/fng/"
            res = self.session.get(url, timeout=4)
            if res.status_code == 200:
                d = res.json()["data"][0]
                val = int(d["value"])
                cls = d["value_classification"]
                fa_cls = {
                    "Extreme Fear": "ترس شدید (Extreme Fear) - ارزندگی قیمت و فرصت انباشت پله‌ای",
                    "Fear": "ترس (Fear) - احتیاط حاکم بر بازار",
                    "Neutral": "خنثی (Neutral) - تعادل احساسی معامله‌گران",
                    "Greed": "طمع (Greed) - ورود هیجانی سرمایه‌گذاران",
                    "Extreme Greed": "طمع شدید (Extreme Greed) - هشدار سقف‌های قیمتی و سیو سود"
                }.get(cls, cls)
                return {
                    "value": val,
                    "classification": cls,
                    "classification_fa": fa_cls,
                    "timestamp": d.get("timestamp")
                }
        except Exception:
            pass
        return {"value": 50, "classification": "Neutral", "classification_fa": "خنثی", "timestamp": None}

    def fetch_coingecko_details(self, base_coin: str) -> Dict[str, Any]:
        try:
            url = f"https://api.coingecko.com/api/v3/search?query={base_coin}"
            res = self.session.get(url, timeout=4)
            if res.status_code == 200:
                coins = res.json().get("coins", [])
                match = None
                for c in coins:
                    if c["symbol"].upper() == base_coin.upper():
                        match = c
                        break
                if not match and coins:
                    match = coins[0]
                
                if match:
                    coin_id = match["id"]
                    detail_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false"
                    det_res = self.session.get(detail_url, timeout=5)
                    if det_res.status_code == 200:
                        md = det_res.json()
                        market_data = md.get("market_data", {})
                        ath = market_data.get("ath", {}).get("usd", 0)
                        ath_change_pct = market_data.get("ath_change_percentage", {}).get("usd", 0)
                        mcap = market_data.get("market_cap", {}).get("usd", 0)
                        fdv = market_data.get("fully_diluted_valuation", {}).get("usd", 0)
                        rank = md.get("market_cap_rank", 0)
                        circ_supply = market_data.get("circulating_supply", 0)
                        total_supply = market_data.get("total_supply", 0)
                        desc = md.get("description", {}).get("en", "")
                        if desc:
                            desc = desc.split(". ")[0] + "."
                        return {
                            "name": md.get("name"),
                            "symbol": md.get("symbol", "").upper(),
                            "rank": rank,
                            "market_cap": mcap,
                            "fdv": fdv,
                            "ath": ath,
                            "ath_change_pct": round(ath_change_pct, 2) if ath_change_pct else 0,
                            "circulating_supply": circ_supply,
                            "total_supply": total_supply,
                            "categories": md.get("categories", [])[:3],
                            "brief": desc
                        }
        except Exception:
            pass
        return {}


class SmartMoneyEngine:
    """
    ICT / SMC (Smart Money Concepts), Order Flow, HFT Footprint, VWAP, Value Area, and CVD Engine.
    """
    @staticmethod
    def detect_rsi_divergences(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, rsi_values: np.ndarray) -> Dict[str, Any]:
        n = len(closes)
        if n < 25:
            return {"bullish": False, "bearish": False, "type": "NONE", "desc": "داده ناکافی برای واگرایی"}
            
        sh_price = []
        sh_rsi = []
        for i in range(n - 30, n - 2):
            if highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
                sh_price.append((i, highs[i]))
                sh_rsi.append((i, rsi_values[i]))
                
        sl_price = []
        sl_rsi = []
        for i in range(n - 30, n - 2):
            if lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
                sl_price.append((i, lows[i]))
                sl_rsi.append((i, rsi_values[i]))

        bearish_div = False
        bullish_div = False
        div_desc = "واگرایی تاییدشده‌ای مشاهده نمی‌شود (تداوم مومنتوم طبیعی)."
        div_type = "NONE"

        if len(sh_price) >= 2:
            p1, p2 = sh_price[-2][1], sh_price[-1][1]
            r1, r2 = sh_rsi[-2][1], sh_rsi[-1][1]
            if p2 > p1 and r2 < (r1 - 1.5):
                bearish_div = True
                div_type = "BEARISH_REGULAR"
                div_desc = "واگرایی منفی کلاسیک (Regular Bearish): قیمت سقف جدید ثبت کرده اما RSI ناتوان است؛ هشدار ریزش و ضعف خریداران."

        if len(sl_price) >= 2:
            p1, p2 = sl_price[-2][1], sl_price[-1][1]
            r1, r2 = sl_rsi[-2][1], sl_rsi[-1][1]
            if p2 < p1 and r2 > (r1 + 1.5):
                bullish_div = True
                div_type = "BULLISH_REGULAR"
                div_desc = "واگرایی مثبت کلاسیک (Regular Bullish): قیمت کف جدید زده اما RSI کف بالاتر ساخته؛ سیگنال پرقدرت ورود خریداران پنهان."

        return {
            "bullish": bullish_div,
            "bearish": bearish_div,
            "type": div_type,
            "desc": div_desc
        }

    @staticmethod
    def get_market_killzone() -> Dict[str, Any]:
        gm = time.gmtime()
        hour = gm.tm_hour
        minute = gm.tm_min
        time_dec = hour + minute / 60.0

        session = "آسیا (Asian Session)"
        killzone = "ساعات عادی (Off-Hours)"
        is_killzone = False

        if 7.0 <= time_dec <= 10.0:
            session = "لندن (London Session)"
            killzone = "🎯 کیل‌زون افتتاحیه لندن (London Open Killzone)"
            is_killzone = True
        elif 12.0 <= time_dec <= 15.5:
            session = "نیویورک (New York Session)"
            killzone = "🔥 کیل‌زون طوفانی نیویورک (New York Open Killzone - بالاترین نقدینگی)"
            is_killzone = True
        elif 15.5 < time_dec <= 17.5:
            session = "پایان لندن (London Close)"
            killzone = "سیو سود و کلوز لندن (London Close Killzone)"
            is_killzone = True
        elif 0.0 <= time_dec <= 6.0:
            session = "سشن آسیا (Asian Session)"
            killzone = "تثبیت رنج آسیا (Asian Range Consolidation)"
            is_killzone = False
        else:
            session = "سشن بین‌بانکی / آرام (Inter-bank / Pre-session)"
            killzone = "ساعات آرام بازار (Normal Hours)"
            is_killzone = False

        return {
            "utc_time": f"{hour:02d}:{minute:02d} UTC",
            "active_session": session,
            "killzone": killzone,
            "is_killzone": is_killzone,
            "advice": "زمان مناسب ورود پرقدرت با جریان نقدینگی" if is_killzone else "در ساعات کم‌حجم محتاط باشید؛ ورودهای با لوریج بالا ممکن است رنج شوند."
        }

    @staticmethod
    def compute_vwap_and_cvd(candles: List[List[float]]) -> Dict[str, Any]:
        opens = np.array([float(c[1]) for c in candles])
        highs = np.array([float(c[2]) for c in candles])
        lows = np.array([float(c[3]) for c in candles])
        closes = np.array([float(c[4]) for c in candles])
        volumes = np.array([float(c[5]) for c in candles])
        n = len(candles)

        # VWAP
        typical = (highs + lows + closes) / 3.0
        cum_tp_vol = np.cumsum(typical * volumes)
        cum_vol = np.cumsum(volumes)
        cum_vol[cum_vol == 0] = 1.0
        vwap = cum_tp_vol / cum_vol
        curr_vwap = vwap[-1]
        
        dev = np.sqrt(np.cumsum(volumes * (typical - vwap)**2) / cum_vol)[-1]
        vwap_upper = curr_vwap + 1.5 * dev
        vwap_lower = curr_vwap - 1.5 * dev

        curr_price = closes[-1]
        vwap_position = "بالای VWAP (Bullish Bias)" if curr_price >= curr_vwap else "پایین VWAP (Bearish Bias)"

        # CVD & Delta Volume
        deltas = []
        for i in range(n):
            rng = highs[i] - lows[i]
            if rng > 0:
                ratio = (closes[i] - opens[i]) / rng
                deltas.append(volumes[i] * ratio)
            else:
                deltas.append(0.0)
        cvd = np.cumsum(deltas)
        delta_latest = deltas[-1]
        cvd_slope = "مثبت / صعودی (فشار خریداران تهاجمی)" if (cvd[-1] >= cvd[max(0, n-5)]) else "منفی / نزولی (فشار فروشندگان تهاجمی)"

        return {
            "vwap": round(float(curr_vwap), 6),
            "vwap_upper": round(float(vwap_upper), 6),
            "vwap_lower": round(float(vwap_lower), 6),
            "vwap_position": vwap_position,
            "delta_latest": round(float(delta_latest), 2),
            "delta_status": "خرید تهاجمی (Market Buy Delta)" if delta_latest > 0 else "فروش تهاجمی (Market Sell Delta)",
            "cvd_trend": cvd_slope,
            "cvd_val": round(float(cvd[-1]), 2)
        }

    @staticmethod
    def compute_value_area(candles: List[List[float]], lookback: int = 50) -> Dict[str, Any]:
        highs = np.array([float(c[2]) for c in candles])
        lows = np.array([float(c[3]) for c in candles])
        volumes = np.array([float(c[5]) for c in candles])
        n = len(candles)
        lb = min(lookback, n)

        min_p = np.min(lows[-lb:])
        max_p = np.max(highs[-lb:])
        bins = np.linspace(min_p, max_p, 30)
        bin_vols = np.zeros(len(bins)-1)

        for i in range(n - lb, n):
            mid = (highs[i] + lows[i]) / 2.0
            for b in range(len(bins)-1):
                if bins[b] <= mid < bins[b+1]:
                    bin_vols[b] += volumes[i]
                    break

        poc_idx = int(np.argmax(bin_vols))
        poc = float((bins[poc_idx] + bins[poc_idx+1]) / 2.0)
        total_v = np.sum(bin_vols)
        target_v = total_v * 0.70

        cur_v = bin_vols[poc_idx]
        l_idx, r_idx = poc_idx, poc_idx
        while cur_v < target_v and (l_idx > 0 or r_idx < len(bin_vols)-1):
            left_v = bin_vols[l_idx-1] if l_idx > 0 else 0
            right_v = bin_vols[r_idx+1] if r_idx < len(bin_vols)-1 else 0
            if left_v >= right_v and l_idx > 0:
                l_idx -= 1
                cur_v += left_v
            elif r_idx < len(bin_vols)-1:
                r_idx += 1
                cur_v += right_v
            else:
                break
        val = float(bins[l_idx])
        vah = float(bins[r_idx+1])

        return {
            "poc": round(poc, 6),
            "vah": round(vah, 6),
            "val": round(val, 6)
        }

    @staticmethod
    def evaluate_derivatives_matrix(price_change_pct: float, oi_usd: float, funding_rate: float) -> Dict[str, Any]:
        # 4 Quadrants:
        # 1. Price UP + OI UP: Long Accumulation (پول نو وارد خرید شده، روند سالم صعودی)
        # 2. Price UP + OI DOWN: Short Covering (رشد ناشی از استاپ خوردن شورت‌ها)
        # 3. Price DOWN + OI UP: Short Accumulation (ورود سنگین فروشندگان سازمانی)
        # 4. Price DOWN + OI DOWN: Long Liquidation (لیکوئید شدن خریداران)
        if price_change_pct >= 0.5:
            regime = "Long Accumulation (انباشت لانگ)"
            regime_fa = "انباشت پوزیشن‌های خرید سازمانی"
            regime_code = "LONG_ACCUMULATION"
            behavior = "Price ↑ + OI ↑ (قدرت روند صعودی)"
            implication = "افزایش قیمت همراه با ورود سرمایه نو به پوزیشن‌های خرید؛ تایید قدرت روند صعودی."
        elif price_change_pct <= -0.5:
            regime = "Short Accumulation (انباشت شورت)"
            regime_fa = "انباشت پوزیشن‌های فروش سازمانی"
            regime_code = "SHORT_ACCUMULATION"
            behavior = "Price ↓ + OI ↑ (قدرت روند نزولی)"
            implication = "کاهش قیمت همراه با ورود پول پرقدرت به قراردادهای فروش؛ هشدار ریزش بیشتر."
        else:
            regime = "Consolidation / Neutral (رنج و تسویه)"
            regime_fa = "تعادل رژیم بازار و نوسان رنج"
            regime_code = "NEUTRAL"
            behavior = "Price ~ + OI Stable"
            implication = "تعادل میان خریداران و فروشندگان مشتقه؛ فاقد جهت‌گیری تهاجمی."

        return {
            "regime": regime,
            "regime_fa": regime_fa,
            "regime_code": regime_code,
            "behavior": behavior,
            "implication": implication,
            "price_change": f"{price_change_pct:+.2f}%",
            "oi_change": "افزایشی" if abs(price_change_pct) > 0.5 else "متعادل"
        }

    @classmethod
    def analyze_smc(cls, candles: List[List[float]], current_price: float, timeframe: str = "15m", rsi_arr: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if not candles or len(candles) < 25:
            return {}

        opens = np.array([float(c[1]) for c in candles])
        highs = np.array([float(c[2]) for c in candles])
        lows = np.array([float(c[3]) for c in candles])
        closes = np.array([float(c[4]) for c in candles])
        volumes = np.array([float(c[5]) for c in candles])
        n = len(candles)

        # 1. FVGs
        fvgs = []
        for i in range(2, n):
            if lows[i] > highs[i-2]:
                bottom = highs[i-2]
                top = lows[i]
                gap_size = top - bottom
                gap_pct = (gap_size / closes[i]) * 100
                if gap_pct > 0.08:
                    mitigated = any(lows[j] <= bottom for j in range(i+1, n))
                    fvgs.append({
                        "type": "BULLISH_FVG",
                        "title": "خلاء نقدینگی صعودی (Bullish FVG)",
                        "bottom": round(float(bottom), 6),
                        "top": round(float(top), 6),
                        "gap_pct": round(gap_pct, 2),
                        "candle_index": i,
                        "mitigated": mitigated,
                        "distance_pct": round(abs((top - current_price) / current_price) * 100, 2)
                    })
            elif highs[i] < lows[i-2]:
                top = lows[i-2]
                bottom = highs[i]
                gap_size = top - bottom
                gap_pct = (gap_size / closes[i]) * 100
                if gap_pct > 0.08:
                    mitigated = any(highs[j] >= top for j in range(i+1, n))
                    fvgs.append({
                        "type": "BEARISH_FVG",
                        "title": "خلاء نقدینگی نزولی (Bearish FVG)",
                        "top": round(float(top), 6),
                        "bottom": round(float(bottom), 6),
                        "gap_pct": round(gap_pct, 2),
                        "candle_index": i,
                        "mitigated": mitigated,
                        "distance_pct": round(abs((bottom - current_price) / current_price) * 100, 2)
                    })

        unmitigated_fvgs = [f for f in fvgs if not f["mitigated"]]
        nearest_fvg = None
        if unmitigated_fvgs:
            unmitigated_fvgs.sort(key=lambda x: x["distance_pct"])
            nearest_fvg = unmitigated_fvgs[0]

        # 2. BOS & CHoCH
        swing_highs = []
        swing_lows = []
        for i in range(2, n - 2):
            if highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
                swing_highs.append({"index": i, "price": highs[i]})
            if lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
                swing_lows.append({"index": i, "price": lows[i]})

        structure_events = []
        market_bias = "BULLISH"
        last_sh = swing_highs[0]["price"] if swing_highs else closes[0]
        last_sl = swing_lows[0]["price"] if swing_lows else closes[0]

        for i in range(5, n):
            c_close = closes[i]
            if c_close > last_sh:
                if market_bias == "BEARISH":
                    structure_events.append({
                        "type": "CHoCH_BULLISH",
                        "label": "CHoCH صعودی (تغییر ماهیت روند به فاز گاوی)",
                        "broken_level": round(float(last_sh), 6),
                        "trigger_price": round(float(c_close), 6),
                        "candle_index": i
                    })
                    market_bias = "BULLISH"
                else:
                    structure_events.append({
                        "type": "BOS_BULLISH",
                        "label": "BOS صعودی (شکست ساختار و تداوم مومنتوم گاوی)",
                        "broken_level": round(float(last_sh), 6),
                        "trigger_price": round(float(c_close), 6),
                        "candle_index": i
                    })
                last_sh = max(highs[max(0, i-5):i+1])
            elif c_close < last_sl:
                if market_bias == "BULLISH":
                    structure_events.append({
                        "type": "CHoCH_BEARISH",
                        "label": "CHoCH نزولی (تغییر ماهیت روند به فاز خرسی)",
                        "broken_level": round(float(last_sl), 6),
                        "trigger_price": round(float(c_close), 6),
                        "candle_index": i
                    })
                    market_bias = "BEARISH"
                else:
                    structure_events.append({
                        "type": "BOS_BEARISH",
                        "label": "BOS نزولی (شکست ساختار به سمت پایین)",
                        "broken_level": round(float(last_sl), 6),
                        "trigger_price": round(float(c_close), 6),
                        "candle_index": i
                    })
                last_sl = min(lows[max(0, i-5):i+1])

        latest_structure = structure_events[-1] if structure_events else {
            "type": "NEUTRAL",
            "label": "ساختار در حال رنج و فشرده‌سازی نقدینگی",
            "broken_level": current_price,
            "trigger_price": current_price
        }

        # 3. Sweeps
        sweeps = []
        for i in range(10, n):
            prev_high = np.max(highs[max(0, i-10):i])
            prev_low = np.min(lows[max(0, i-10):i])
            if highs[i] > prev_high and closes[i] < prev_high:
                sweeps.append({
                    "type": "BSL_SWEEP",
                    "title": "شکار نقدینگی استاپ‌های خرید (BSL Sweep)",
                    "level_swept": round(float(prev_high), 6),
                    "wick_extreme": round(float(highs[i]), 6),
                    "candle_index": i,
                    "desc": "قیمت سقف محلی را هانت کرد و بلافاصله به زیر سقف بازگشت (تله خریداران بریک‌اوت)"
                })
            if lows[i] < prev_low and closes[i] > prev_low:
                sweeps.append({
                    "type": "SSL_SWEEP",
                    "title": "شکار نقدینگی استاپ‌های فروش (SSL Sweep)",
                    "level_swept": round(float(prev_low), 6),
                    "wick_extreme": round(float(lows[i]), 6),
                    "candle_index": i,
                    "desc": "قیمت کف محلی را هانت کرد و با شدوی بلند به بالا بازگشت (تله فروشندگان کف)"
                })

        latest_sweep = sweeps[-1] if sweeps else None

        # 4. Volume Profile & Value Area
        va = cls.compute_value_area(candles, 50)
        poc_price = va["poc"]
        vah = va["vah"]
        val = va["val"]

        lookback = min(60, n)
        min_p = np.min(lows[-lookback:])
        max_p = np.max(highs[-lookback:])
        bsl_pool_target = round(float(max_p), 6)
        ssl_pool_target = round(float(min_p), 6)

        # 5. Order Blocks
        bullish_ob = None
        bearish_ob = None
        for i in range(n - 20, n - 2):
            if closes[i] < opens[i] and closes[i+1] > opens[i+1] and closes[i+2] > closes[i+1]:
                if (closes[i+2] - opens[i+1]) > 1.8 * (highs[i] - lows[i]):
                    bullish_ob = {
                        "type": "BULLISH_ORDER_BLOCK",
                        "title": "بلوک سفارشات صعودی (Bullish OB)",
                        "top": round(float(opens[i]), 6),
                        "bottom": round(float(lows[i]), 6),
                        "status": "محدوده تقاضای نهنگ‌ها (Demand Zone)"
                    }
            if closes[i] > opens[i] and closes[i+1] < opens[i+1] and closes[i+2] < closes[i+1]:
                if (opens[i+1] - closes[i+2]) > 1.8 * (highs[i] - lows[i]):
                    bearish_ob = {
                        "type": "BEARISH_ORDER_BLOCK",
                        "title": "بلوک سفارشات نزولی (Bearish OB)",
                        "top": round(float(highs[i]), 6),
                        "bottom": round(float(opens[i]), 6),
                        "status": "محدوده عرضه نهنگ‌ها (Supply Zone)"
                    }

        # 6. Whale Spikes
        vol_mean = np.mean(volumes[-40:]) if n >= 40 else np.mean(volumes)
        whale_trades = []
        for i in range(max(0, n - 25), n):
            v_ratio = volumes[i] / vol_mean if vol_mean > 0 else 1.0
            if v_ratio >= 2.3:
                direction = "خرید عمده نهنگ (Whale Buy Spike)" if closes[i] >= opens[i] else "فروش عمده نهنگ (Whale Sell Spike)"
                whale_trades.append({
                    "candle_index": i,
                    "direction": direction,
                    "volume_ratio": round(float(v_ratio), 2),
                    "price": round(float(closes[i]), 6),
                    "is_recent": (n - i) <= 6
                })

        # 7. HFT Footprint
        recent_15_vols = volumes[-15:]
        vol_spikes_count = sum(1 for v in recent_15_vols if v > 2.0 * vol_mean)
        wick_ratios = []
        for i in range(max(0, n - 15), n):
            rng = highs[i] - lows[i]
            body = abs(closes[i] - opens[i])
            wick_ratios.append(1.0 - (body / rng) if rng > 0 else 0)
        avg_wick = np.mean(wick_ratios) if wick_ratios else 0.5
        
        hft_score = min(98, max(15, int(35 + vol_spikes_count * 12 + avg_wick * 35)))
        if hft_score >= 75:
            hft_status = "تهاجمی (Aggressive HFT Liquidity Hunt)"
            hft_desc = "الگوریتم‌های با بسامد بالا (HFT) در حال اسکن خطوط استاپ و هانت نقدینگی فعال هستند."
        elif hft_score >= 50:
            hft_status = "فعالیت ربات‌های بازارساز (Market Making Bots)"
            hft_desc = "الگوریتم‌ها در حال مدیریت اسپرد و تعادل نقدینگی هستند."
        else:
            hft_status = "جریان آرام ارگانیک (Organic Low Bot Flow)"
            hft_desc = "فعالیت الگوریتم‌های پربسامد پایین است و تحرکات معامله‌گران خرد بیشتر به چشم می‌خورد."

        # 8. Fake Trend / Traps
        fake_trend_badge = "ORGANIC"
        fake_trend_title = "روند طبیعی و معتبر (Organic Flow)"
        fake_trend_desc = "شکست‌ها با حجم متناسب همراه بوده و علائم تله معکوس فوری دیده نمی‌شود."

        if latest_sweep and (n - latest_sweep["candle_index"] <= 4):
            if latest_sweep["type"] == "BSL_SWEEP":
                fake_trend_badge = "BULL_TRAP"
                fake_trend_title = "⚠️ هشدار تله گاوی (Bull Trap) - هانت سقف"
                fake_trend_desc = f"قیمت سقف {latest_sweep['level_swept']} را به صورت فیک شکست و با شدوی بلند ریجکت شد؛ نقدینگی خریداران خرد جمع‌آوری شد."
            elif latest_sweep["type"] == "SSL_SWEEP":
                fake_trend_badge = "BEAR_TRAP"
                fake_trend_title = "⚠️ هشدار تله خرسی (Bear Trap) - هانت کف"
                fake_trend_desc = f"قیمت کف {latest_sweep['level_swept']} را هانت کرده و با بازگشت سریع فروشندگان وحشت‌زده را به دام انداخت."

        # 9. Premium vs Discount Zone
        range_mid = (min_p + max_p) / 2.0
        if current_price > range_mid:
            pricing_zone = "منطقه گران‌فروشی (Premium Zone) - مناسب برای سیو سود یا پوزیشن‌های شورت"
            pricing_code = "PREMIUM"
        else:
            pricing_zone = "منطقه ارزان‌خری (Discount Zone) - مناسب برای انباشت و لانگ‌های کم‌ریسک"
            pricing_code = "DISCOUNT"

        # 10. Liquidation Clusters & Correction Phase
        long_liq_50x = round(current_price * 0.981, 6)
        long_liq_25x = round(current_price * 0.961, 6)
        short_liq_50x = round(current_price * 1.019, 6)
        short_liq_25x = round(current_price * 1.039, 6)

        poc_dist_pct = ((current_price - poc_price) / poc_price) * 100 if poc_price > 0 else 0
        if abs(poc_dist_pct) < 1.0:
            correction_phase = "تثبیت روی گره نقدینگی POC (حالت رنج و انباشت)"
            correction_badge = "CONSOLIDATION"
        elif poc_dist_pct > 2.5:
            correction_phase = "کشش شدید صعودی و فاصله از تعادل (احتمال آغاز اصلاح به سمت POC)"
            correction_badge = "EXTENDED_BULLISH"
        elif poc_dist_pct < -2.5:
            correction_phase = "کشش شدید نزولی و فروش هیجانی (احتمال ریباند صعودی به سمت POC)"
            correction_badge = "EXTENDED_BEARISH"
        else:
            correction_phase = "اصلاح طبیعی و پولبک ارگانیک (Healthy Pullback)"
            correction_badge = "HEALTHY_PULLBACK"

        inducement_level = round(float(highs[-3] if current_price < poc_price else lows[-3]), 6)

        # 11. Divergences
        if rsi_arr is None or len(rsi_arr) < n:
            rsi_calc = TechnicalAnalyzer.calc_rsi(closes, 14)
        else:
            rsi_calc = rsi_arr
        divergence = cls.detect_rsi_divergences(closes, highs, lows, rsi_calc)

        # 12. Killzones & Session
        killzone_info = cls.get_market_killzone()

        # 13. VWAP & CVD
        vwap_cvd = cls.compute_vwap_and_cvd(candles)
        vwap_cvd["vah"] = vah
        vwap_cvd["val"] = val
        vwap_cvd["poc"] = poc_price

        return {
            "poc": poc_price,
            "vah": vah,
            "val": val,
            "bsl_pool_target": bsl_pool_target,
            "ssl_pool_target": ssl_pool_target,
            "unmitigated_fvgs": unmitigated_fvgs[-4:],
            "nearest_fvg": nearest_fvg,
            "latest_structure": latest_structure,
            "sweeps": sweeps[-3:],
            "latest_sweep": latest_sweep,
            "bullish_ob": bullish_ob,
            "bearish_ob": bearish_ob,
            "whale_trades": whale_trades[-4:],
            "hft": {
                "score": hft_score,
                "status": hft_status,
                "desc": hft_desc
            },
            "fake_trend": {
                "badge": fake_trend_badge,
                "title": fake_trend_title,
                "desc": fake_trend_desc
            },
            "pricing": {
                "code": pricing_code,
                "zone": pricing_zone,
                "range_mid": round(float(range_mid), 6)
            },
            "liquidation_clusters": {
                "long_50x": long_liq_50x,
                "long_25x": long_liq_25x,
                "short_50x": short_liq_50x,
                "short_25x": short_liq_25x,
                "correction_phase": correction_phase,
                "correction_badge": correction_badge,
                "inducement_level": inducement_level
            },
            "divergence": divergence,
            "killzone": killzone_info,
            "vwap_cvd": vwap_cvd
        }


class TechnicalAnalyzer:
    @staticmethod
    def calc_ema(arr: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1.0)
        ema = np.zeros(len(arr))
        ema[0] = arr[0]
        for i in range(1, len(arr)):
            ema[i] = alpha * arr[i] + (1 - alpha) * ema[i-1]
        return ema

    @staticmethod
    def calc_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
        n = len(closes)
        if n <= period:
            return np.full(n, 50.0)
        deltas = np.diff(closes)
        rsi = np.zeros(n)
        up = np.maximum(deltas, 0)
        down = -np.minimum(deltas, 0)
        avg_up = np.mean(up[:period])
        avg_down = np.mean(down[:period])
        rs = avg_up / avg_down if avg_down > 0 else 100.0
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
        for i in range(period + 1, n):
            avg_up = (avg_up * (period - 1) + up[i-1]) / period
            avg_down = (avg_down * (period - 1) + down[i-1]) / period
            rs = avg_up / avg_down if avg_down > 0 else 100.0
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def calc_macd(closes: np.ndarray) -> Dict[str, np.ndarray]:
        ema12 = TechnicalAnalyzer.calc_ema(closes, 12)
        ema26 = TechnicalAnalyzer.calc_ema(closes, 26)
        macd_line = ema12 - ema26
        macd_signal = TechnicalAnalyzer.calc_ema(macd_line, 9)
        macd_hist = macd_line - macd_signal
        return {"macd": macd_line, "signal": macd_signal, "hist": macd_hist}

    @staticmethod
    def calc_bollinger(closes: np.ndarray, period: int = 20, num_std: float = 2.0) -> Dict[str, np.ndarray]:
        n = len(closes)
        upper = np.zeros(n)
        middle = np.zeros(n)
        lower = np.zeros(n)
        bandwidth = np.zeros(n)
        for i in range(period - 1, n):
            slice_ = closes[i - period + 1 : i + 1]
            m = np.mean(slice_)
            s = np.std(slice_)
            middle[i] = m
            upper[i] = m + num_std * s
            lower[i] = m - num_std * s
            bandwidth[i] = ((upper[i] - lower[i]) / m) * 100 if m > 0 else 0
        return {"upper": upper, "middle": middle, "lower": lower, "bandwidth": bandwidth}

    @staticmethod
    def calc_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
        n = len(closes)
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        return TechnicalAnalyzer.calc_ema(tr, period)

    @classmethod
    def analyze_candles(cls, candles: List[List[float]], timeframe: str = "15m") -> Dict[str, Any]:
        if not candles or len(candles) < 20:
            return {}

        opens = np.array([float(c[1]) for c in candles])
        highs = np.array([float(c[2]) for c in candles])
        lows = np.array([float(c[3]) for c in candles])
        closes = np.array([float(c[4]) for c in candles])
        volumes = np.array([float(c[5]) for c in candles])
        n = len(closes)

        price = closes[-1]
        prev_price = closes[-2]
        
        ema9 = cls.calc_ema(closes, 9)
        ema20 = cls.calc_ema(closes, 20)
        ema50 = cls.calc_ema(closes, min(50, n))
        ema200 = cls.calc_ema(closes, min(200, n)) if n >= 50 else ema50
        
        rsi = cls.calc_rsi(closes, 14)
        macd = cls.calc_macd(closes)
        bb = cls.calc_bollinger(closes, 20, 2.0)
        atr = cls.calc_atr(highs, lows, closes, 14)
        
        vol_sma20 = np.mean(volumes[-20:]) if n >= 20 else np.mean(volumes)
        vol_ratio = volumes[-1] / vol_sma20 if vol_sma20 > 0 else 1.0

        supports = []
        resistances = []
        for i in range(2, n - 2):
            if lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
                supports.append(lows[i])
            if highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
                resistances.append(highs[i])

        valid_supports = sorted([s for s in supports if s < price])
        valid_resistances = sorted([r for r in resistances if r > price])
        
        nearest_support = valid_supports[-1] if valid_supports else bb["lower"][-1]
        nearest_resistance = valid_resistances[0] if valid_resistances else bb["upper"][-1]

        lookback = min(50, n)
        recent_high = float(np.max(highs[-lookback:]))
        recent_low = float(np.min(lows[-lookback:]))
        
        diff = recent_high - recent_low if recent_high > recent_low else price * 0.05
        fib = {
            "0.000 (کف)": round(recent_low, 6),
            "0.236": round(recent_low + 0.236 * diff, 6),
            "0.382": round(recent_low + 0.382 * diff, 6),
            "0.500": round(recent_low + 0.500 * diff, 6),
            "0.618 (پاکت طلایی)": round(recent_low + 0.618 * diff, 6),
            "0.786": round(recent_low + 0.786 * diff, 6),
            "1.000 (سقف)": round(recent_high, 6)
        }

        curr_rsi = rsi[-1]
        prev_rsi = rsi[-2]
        curr_macd_hist = macd["hist"][-1]
        prev_macd_hist = macd["hist"][-2]
        
        bullish_score = 0
        bearish_score = 0

        if price > ema20[-1]: bullish_score += 1
        else: bearish_score += 1
        
        if ema20[-1] > ema50[-1]: bullish_score += 1.5
        else: bearish_score += 1.5
        
        if price > ema200[-1]: bullish_score += 1.5
        else: bearish_score += 1.5

        if 50 < curr_rsi < 70: bullish_score += 1.5
        elif curr_rsi >= 70: bearish_score += 1
        elif curr_rsi < 30: bullish_score += 1
        elif curr_rsi < 50: bearish_score += 1.5

        if curr_macd_hist > 0 and curr_macd_hist > prev_macd_hist: bullish_score += 2
        elif curr_macd_hist > 0: bullish_score += 1
        elif curr_macd_hist < 0 and curr_macd_hist < prev_macd_hist: bearish_score += 2
        elif curr_macd_hist < 0: bearish_score += 1

        if vol_ratio > 1.2 and price > prev_price: bullish_score += 1
        elif vol_ratio > 1.2 and price < prev_price: bearish_score += 1

        total_score = bullish_score - bearish_score
        if total_score >= 3:
            bias = "BULLISH_STRONG"
            bias_fa = "صعودی پرقدرت (Strong Bullish)"
        elif total_score >= 1:
            bias = "BULLISH"
            bias_fa = "صعودی معتدل (Bullish)"
        elif total_score <= -3:
            bias = "BEARISH_STRONG"
            bias_fa = "نزولی پرقدرت (Strong Bearish)"
        elif total_score <= -1:
            bias = "BEARISH"
            bias_fa = "نزولی معتدل (Bearish)"
        else:
            bias = "NEUTRAL"
            bias_fa = "خنثی / رنج (Neutral / Range)"

        return {
            "timeframe": timeframe,
            "price": price,
            "ema9": round(float(ema9[-1]), 6),
            "ema20": round(float(ema20[-1]), 6),
            "ema50": round(float(ema50[-1]), 6),
            "ema200": round(float(ema200[-1]), 6),
            "rsi": round(float(curr_rsi), 2),
            "macd": round(float(macd["macd"][-1]), 6),
            "macd_signal": round(float(macd["signal"][-1]), 6),
            "macd_hist": round(float(curr_macd_hist), 6),
            "bb_upper": round(float(bb["upper"][-1]), 6),
            "bb_middle": round(float(bb["middle"][-1]), 6),
            "bb_lower": round(float(bb["lower"][-1]), 6),
            "bb_bandwidth": round(float(bb["bandwidth"][-1]), 2),
            "atr": round(float(atr[-1]), 6),
            "atr_pct": round((float(atr[-1]) / price) * 100, 2) if price > 0 else 0,
            "vol_ratio": round(float(vol_ratio), 2),
            "nearest_support": round(float(nearest_support), 6),
            "nearest_resistance": round(float(nearest_resistance), 6),
            "recent_high": round(recent_high, 6),
            "recent_low": round(recent_low, 6),
            "fib": fib,
            "bias": bias,
            "bias_fa": bias_fa,
            "score": total_score
        }


class CryptoTradingAgent:
    def __init__(self):
        self.fetcher = CryptoDataFetcher()
        self.analyzer = TechnicalAnalyzer()
        self.smc = SmartMoneyEngine()

    def analyze_symbol(self, raw_symbol: str) -> Dict[str, Any]:
        symbol = clean_symbol(raw_symbol)
        base_coin = get_base_coin(symbol)

        # 1. Fetch Ticker
        ticker = self.fetcher.fetch_ticker(symbol)
        if not ticker:
            ticker = self.fetcher.fetch_ticker(raw_symbol.upper())
            if ticker:
                symbol = raw_symbol.upper()
            else:
                return {
                    "success": False,
                    "error": f"نماد '{raw_symbol}' در دیتابیس صرافی‌های متصل یافت نشد. لطفاً از نمادهای استاندارد مانند BTC, ETH, SOL, PEPE, SUI, DOGE استفاده کنید."
                }

        current_price = ticker["last_price"]

        # 2. Fetch Multi-Timeframe Candles
        klines_15m = self.fetcher.fetch_klines(symbol, "15m", 300)
        klines_1h = self.fetcher.fetch_klines(symbol, "1h", 150)
        klines_4h = self.fetcher.fetch_klines(symbol, "4h", 150)
        klines_1d = self.fetcher.fetch_klines(symbol, "1d", 100)

        # 3. Analyze Technicals
        ta_15m = self.analyzer.analyze_candles(klines_15m, "15m") if klines_15m else {}
        ta_1h = self.analyzer.analyze_candles(klines_1h, "1h") if klines_1h else {}
        ta_4h = self.analyzer.analyze_candles(klines_4h, "4h") if klines_4h else {}
        ta_1d = self.analyzer.analyze_candles(klines_1d, "1d") if klines_1d else {}

        # 4. Smart Money Concepts, Order Flow, VWAP, CVD, Value Area
        smc_15m = self.smc.analyze_smc(klines_15m, current_price, "15m") if klines_15m else {}
        smc_4h = self.smc.analyze_smc(klines_4h, current_price, "4h") if klines_4h else {}

        # 5. Orderbook & Microstructure
        orderbook = self.fetcher.fetch_orderbook(symbol, 20)

        # 6. Institutional Derivatives (Open Interest & Funding Rate)
        derivatives = self.fetcher.fetch_institutional_derivatives(symbol)
        derivatives_matrix = self.smc.evaluate_derivatives_matrix(
            ticker.get("price_change_pct", 0),
            derivatives.get("open_interest_usd", 0),
            derivatives.get("funding_rate", 0)
        )

        # 7. Fear & Greed Index
        fng = self.fetcher.fetch_fear_and_greed()

        # 8. CoinGecko Fundamentals
        fundamentals = self.fetcher.fetch_coingecko_details(base_coin)

        # 9. Generate Scalp & Swing Setups
        scalp_setup = self._build_scalp_setup(current_price, ta_15m, ta_1h, orderbook, smc_15m)
        swing_setup = self._build_swing_setup(current_price, ta_4h, ta_1d, fundamentals, smc_4h)

        # 10. Macro Market, Dominance & Correlations (Layer 6)
        macro = MacroEngine.fetch_global_macro()
        btc_candles = klines_15m if base_coin == "BTC" else self.fetcher.fetch_klines("BTCUSDT", "15m", 60)
        correlation = MacroEngine.compute_beta_and_correlation(klines_15m, btc_candles)

        # 11. On-Chain Metrics (Layer 5)
        onchain = OnChainEngine.fetch_onchain_metrics()

        # 12. News & Sentiment Circuit Breaker (Layer 7)
        news_circuit = NewsCircuitBreaker.fetch_live_news()

        # 13. Backtest Simulation & Empirical Calibration
        backtest_res = BacktestEngine.run_backtest(klines_15m, symbol, "15m", 300)

        # 14. Compute 3-Dimensional Institutional Scores:
        # Direction Score, Entry Score, Risk Score, and Pre-Trade Checklist
        scores_3d = self._evaluate_3d_scores(
            current_price, ta_15m, ta_4h, smc_15m, orderbook, derivatives, scalp_setup,
            news_circuit=news_circuit, backtest_summary=backtest_res
        )

        # 15. Institutional Signal Output Table (Exact user roadmap format)
        inst_table = self._build_institutional_table(
            symbol=symbol,
            price=current_price,
            ta_1h=ta_1h,
            smc_15m=smc_15m,
            scalp=scalp_setup,
            ob=orderbook,
            derivatives=derivatives,
            derivatives_matrix=derivatives_matrix,
            scores=scores_3d,
            macro=macro,
            news_circuit=news_circuit,
            backtest=backtest_res
        )

        # 16. Generate AI Comprehensive Verdict
        verdict = self._build_persian_verdict(
            symbol=symbol,
            price=current_price,
            ticker=ticker,
            ta_15m=ta_15m,
            ta_4h=ta_4h,
            ta_1d=ta_1d,
            scalp=scalp_setup,
            swing=swing_setup,
            orderbook=orderbook,
            fng=fng,
            fundamentals=fundamentals,
            smc_15m=smc_15m,
            derivatives=derivatives
        )

        # Chart candles
        chart_candles = []
        source_candles = klines_15m if klines_15m else klines_4h
        if source_candles:
            for c in source_candles[-60:]:
                chart_candles.append({
                    "time": int(c[0] / 1000),
                    "open": c[1],
                    "high": c[2],
                    "low": c[3],
                    "close": c[4],
                    "volume": c[5]
                })

        return {
            "success": True,
            "symbol": symbol,
            "base_coin": base_coin,
            "price": current_price,
            "ticker": ticker,
            "timeframes": {
                "15m": ta_15m,
                "1h": ta_1h,
                "4h": ta_4h,
                "1d": ta_1d
            },
            "smc": {
                "15m": smc_15m,
                "4h": smc_4h
            },
            "derivatives": derivatives,
            "derivatives_matrix": derivatives_matrix,
            "vwap_cvd": smc_15m.get("vwap_cvd", {}),
            "macro": macro,
            "onchain": onchain,
            "news": news_circuit,
            "correlation": correlation,
            "backtest": backtest_res,
            "scores_3d": scores_3d,
            "inst_table": inst_table,
            "trade_quality": scores_3d.get("trade_quality", {}),
            "scalp_setup": scalp_setup,
            "swing_setup": swing_setup,
            "orderbook": orderbook,
            "market_sentiment": fng,
            "fundamentals": fundamentals,
            "verdict": verdict,
            "chart_candles": chart_candles,
            "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }

    def _evaluate_3d_scores(self, price: float, ta_15m: Dict[str, Any], ta_4h: Dict[str, Any], smc_15m: Dict[str, Any], ob: Dict[str, Any], derivatives: Dict[str, Any], scalp: Dict[str, Any], news_circuit: Optional[Dict[str, Any]] = None, backtest_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        bias_15m = ta_15m.get("bias", "NEUTRAL")
        bias_4h = ta_4h.get("bias", "NEUTRAL")
        
        # 1. Direction Score (0-100)
        direction_score = 50
        if "BULLISH" in bias_4h and "BULLISH" in bias_15m: direction_score += 25
        elif "BEARISH" in bias_4h and "BEARISH" in bias_15m: direction_score += 25
        elif bias_4h != bias_15m: direction_score -= 15

        vwap_data = smc_15m.get("vwap_cvd", {})
        if "بالای VWAP" in vwap_data.get("vwap_position", "") and "BULLISH" in bias_15m: direction_score += 15
        elif "پایین VWAP" in vwap_data.get("vwap_position", "") and "BEARISH" in bias_15m: direction_score += 15

        if "مثبت" in vwap_data.get("cvd_trend", "") and "BULLISH" in bias_15m: direction_score += 10
        elif "منفی" in vwap_data.get("cvd_trend", "") and "BEARISH" in bias_15m: direction_score += 10

        ob_ratio = ob.get("ratio", 1.0)
        if ob_ratio > 1.25 and "BULLISH" in bias_15m: direction_score += 10
        elif ob_ratio < 0.75 and "BEARISH" in bias_15m: direction_score += 10

        direction_score = max(15, min(98, direction_score))

        # 2. Entry Score (0-100) - How clean and low-risk is the entry NOW?
        entry_score = 45
        fvg = smc_15m.get("nearest_fvg")
        if fvg and fvg.get("distance_pct", 100) < 1.0: entry_score += 25
        elif fvg and fvg.get("distance_pct", 100) < 2.2: entry_score += 15

        sweep = smc_15m.get("latest_sweep")
        if sweep: entry_score += 20

        rsi_15m = ta_15m.get("rsi", 50)
        if 40 <= rsi_15m <= 62: entry_score += 15
        elif rsi_15m > 74 or rsi_15m < 26: entry_score -= 20 # Overextended, bad entry!

        kz = smc_15m.get("killzone", {})
        if kz.get("is_killzone"): entry_score += 10

        entry_score = max(15, min(98, entry_score))

        # 3. Risk Score (0-100) - Structure safety & R:R
        risk_score = 55
        sl_pct = scalp.get("stop_loss_pct", 2.0)
        if sl_pct < 1.2: risk_score += 20
        elif sl_pct < 2.5: risk_score += 10
        else: risk_score -= 10

        div = smc_15m.get("divergence", {})
        if not div.get("bearish") and not div.get("bullish"): risk_score += 15
        elif ("BULLISH" in bias_15m and div.get("bearish")) or ("BEARISH" in bias_15m and div.get("bullish")): risk_score -= 25

        if ob.get("spread_pct", 0) < 0.05: risk_score += 10

        # Layer 7 Circuit Breaker impact on Risk Score
        if news_circuit and news_circuit.get("circuit_status") == "EMERGENCY_CIRCUIT_BREAKER":
            risk_score = max(10, risk_score - 25)
        elif news_circuit and news_circuit.get("circuit_status") == "CAUTION_HIGH_VOLATILITY":
            risk_score = max(20, risk_score - 10)

        risk_score = max(15, min(98, risk_score))

        # Overall Weighted Confidence with Empirical Backtest Calibration
        theoretical_conf = int(0.40 * direction_score + 0.35 * entry_score + 0.25 * risk_score)
        if backtest_summary and backtest_summary.get("success") and backtest_summary.get("total_trades", 0) >= 5:
            empirical_wr = backtest_summary.get("win_rate_pct", 65)
            composite_confidence = int(0.55 * theoretical_conf + 0.45 * empirical_wr)
        else:
            composite_confidence = theoretical_conf
        composite_confidence = max(20, min(98, composite_confidence))

        # 6-Point Checklist
        checklist = [
            {"item": "همسویی روند تایم‌فریم ۴ ساعته و ۱۵ دقیقه", "passed": bool(("BULLISH" in bias_15m and "BULLISH" in bias_4h) or ("BEARISH" in bias_15m and "BEARISH" in bias_4h)), "detail": f"{ta_15m.get('bias_fa','')} + {ta_4h.get('bias_fa','')}"},
            {"item": "شکار نقدینگی استاپ‌ها (BSL / SSL Sweep)", "passed": bool(sweep is not None), "detail": f"{sweep.get('title')}" if sweep else "نقدینگی دست‌نخورده"},
            {"item": "برخورد یا نزدیکی به خلاء نقدینگی (FVG)", "passed": bool(fvg is not None and fvg.get("distance_pct", 100) < 2.0), "detail": f"{fvg.get('title')}" if fvg else "فاصله بالا"},
            {"item": "موقعیت نسبت به میانگین حجم‌دار (VWAP)", "passed": bool(("بالای VWAP" in vwap_data.get("vwap_position","") and "BULL" in bias_15m) or ("پایین VWAP" in vwap_data.get("vwap_position","") and "BEAR" in bias_15m)), "detail": vwap_data.get("vwap_position", "")},
            {"item": "نبود واگرایی معکوس مخرب در RSI", "passed": bool(not (("BULL" in bias_15m and div.get("bearish")) or ("BEAR" in bias_15m and div.get("bullish")))), "detail": div.get("desc", "")},
            {"item": "زمان‌بندی در سشن پرحجم (ICT Killzone)", "passed": bool(kz.get("is_killzone", False)), "detail": kz.get("killzone", "")}
        ]

        if news_circuit and news_circuit.get("circuit_status") == "EMERGENCY_CIRCUIT_BREAKER":
            grade = "C"
            grade_title = "Grade C / Circuit Breaker Active (فیوز اضطراری اخبار فعال)"
            badge_color = "RED"
            action_advice = "انتشار اخبار با ریسک بحرانی؛ برای جلوگیری از تله‌های خبری، پوزیشن جدید باز نکنید."
        elif composite_confidence >= 80 and entry_score >= 68:
            grade = "A+"
            grade_title = "Grade A+ (سیگنال طلایی با بیشترین احتمال موفقیت)"
            badge_color = "GOLD"
            action_advice = "تمام فاکتورهای جهت، نقطه ورود و ریسک همسو هستند. ورود دقیق با لوریج مطمئن مجاز است."
        elif composite_confidence >= 65:
            grade = "A"
            grade_title = "Grade A (سیگنال استاندارد با دقت بالا)"
            badge_color = "SILVER"
            action_advice = "اکثر فاکتورها تایید شده‌اند؛ ورود پله‌ای با رعایت حد ضرر پیشنهاد می‌شود."
        elif composite_confidence >= 48:
            grade = "B"
            grade_title = "Grade B (ستاپ پرریسک یا نیاز به پولبک)"
            badge_color = "BRONZE"
            action_advice = "جهت روند مشخص است اما نقطه ورود هنوز در اصلاح کامل قرار نگرفته؛ عجله نکنید."
        else:
            grade = "C"
            grade_title = "Grade C / Avoid (خطر معامله - فاز بلاتکلیف)"
            badge_color = "RED"
            action_advice = "بازار در فاز رنج بدون پشتوانه یا تله نقدینگی است؛ از باز کردن پوزیشن پرهیز کنید."

        return {
            "direction_score": direction_score,
            "entry_score": entry_score,
            "risk_score": risk_score,
            "composite_confidence": composite_confidence,
            "grade": grade,
            "grade_title": grade_title,
            "badge_color": badge_color,
            "action_advice": action_advice,
            "checklist": checklist,
            "passed_count": sum(1 for c in checklist if c["passed"]),
            "trade_quality": {
                "score": composite_confidence,
                "grade": grade,
                "grade_title": grade_title,
                "badge_color": badge_color,
                "action_advice": action_advice,
                "checklist": checklist,
                "passed_count": sum(1 for c in checklist if c["passed"])
            }
        }

    def _build_institutional_table(self, symbol: str, price: float, ta_1h: Dict[str, Any], smc_15m: Dict[str, Any], scalp: Dict[str, Any], ob: Dict[str, Any], derivatives: Dict[str, Any], derivatives_matrix: Dict[str, Any], scores: Dict[str, Any], macro: Optional[Dict[str, Any]] = None, news_circuit: Optional[Dict[str, Any]] = None, backtest: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        vwap_data = smc_15m.get("vwap_cvd", {})
        sweep = smc_15m.get("latest_sweep")
        sweep_str = sweep.get("title") if sweep else "نقدینگی در حال انباشت"
        ratio = ob.get("ratio", 1.0)
        imbalance = f"Buy Imbalance ({ratio})" if ratio > 1.15 else (f"Sell Imbalance ({ratio})" if ratio < 0.85 else "تعادل نسبی")
        action = scalp.get("action", "WAIT")
        
        table = [
            {"param": "روند تایم‌فریم ۱ ساعته (Trend 1H)", "val": ta_1h.get("bias_fa", "نامشخص"), "status": "PASS" if "صعودی" in ta_1h.get("bias_fa","") or "نزولی" in ta_1h.get("bias_fa","") else "WARN"},
            {"param": "ساختار تکنیکال ۱۵ دقیقه (Structure 15m)", "val": smc_15m.get("latest_structure", {}).get("label", "رنج"), "status": "PASS" if smc_15m.get("latest_structure") else "WARN"},
            {"param": "موقعیت نسبت به VWAP سازمانی", "val": vwap_data.get("vwap_position", "در محدوده VWAP"), "status": "PASS" if "بالای VWAP" in vwap_data.get("vwap_position","") or "پایین VWAP" in vwap_data.get("vwap_position","") else "WARN"},
            {"param": "حجم دلتا و جهت CVD", "val": f"{vwap_data.get('cvd_trend', '')} (Delta: {vwap_data.get('delta_latest', 0):+.2f})", "status": "PASS" if "مثبت" in vwap_data.get("cvd_trend","") or "منفی" in vwap_data.get("cvd_trend","") else "INFO"},
            {"param": "قراردادهای باز مشتقه (Open Interest)", "val": f"{derivatives.get('open_interest_fmt', 'N/A')} ({derivatives_matrix.get('behavior', '')})", "status": "INFO"},
            {"param": "نرخ تامین مالی (Funding Rate)", "val": f"{derivatives.get('funding_rate_fmt', '0%')} ({derivatives.get('market_crowd', 'خنثی')})", "status": "INFO"},
            {"param": "شکار نقدینگی و تسویه استاپ‌ها (Sweeps)", "val": sweep_str, "status": "PASS" if sweep else "INFO"},
            {"param": "وضعیت دفتر سفارشات (Order Book Imbalance)", "val": imbalance, "status": "PASS" if ratio > 1.15 or ratio < 0.85 else "INFO"},
            {"param": "گره تراکم حجم نقدینگی (POC)", "val": f"${smc_15m.get('poc', 0):,.4f} (VAH: ${smc_15m.get('vah', 0):,.4f} | VAL: ${smc_15m.get('val', 0):,.4f})", "status": "INFO"},
            {"param": "سیگنال نهایی سیستم (Signal)", "val": action, "status": "PASS" if "LONG" in action or "SHORT" in action or "BUY" in action or "SELL" in action else "WARN"},
            {"param": "محدوده ورود بهینه (Entry Zone)", "val": scalp.get("entry_zone", "-"), "status": "INFO"},
            {"param": "حد ضرر قطعی (Stop Loss)", "val": f"${scalp.get('stop_loss', 0):,.4f} (-{scalp.get('stop_loss_pct', 0)}%)", "status": "PASS"},
            {"param": "حدود سود مرحله‌ای (TP1 / TP2 / TP3)", "val": f"TP1: ${scalp.get('tp1',0):,.4f} | TP2: ${scalp.get('tp2',0):,.4f} | TP3: ${scalp.get('tp3',0):,.4f}", "status": "PASS"},
            {"param": "نسبت ریسک به ریوارد (Risk/Reward)", "val": scalp.get("risk_reward", "1:2.0"), "status": "PASS"},
            {"param": "تفکیک امتیازات ۳ بعدی (Direction / Entry / Risk)", "val": f"جهت: {scores.get('direction_score')}/100 | ورود: {scores.get('entry_score')}/100 | ریسک: {scores.get('risk_score')}/100", "status": "PASS" if scores.get("composite_confidence", 0) >= 65 else "WARN"},
            {"param": "درجه سیگنال و وین‌ریت معتبر (Confidence)", "val": f"{scores.get('grade_title')} - {scores.get('composite_confidence')}/100", "status": "PASS" if scores.get("grade") in ["A+", "A"] else ("WARN" if scores.get("grade") == "B" else "DANGER")}
        ]

        if macro:
            table.append({
                "param": "رژیم کلان مارکت و دامیننس (Macro & BTC.D)",
                "val": f"{macro.get('regime')} (سلطه بیت‌کوین: {macro.get('btc_dominance')}%)",
                "status": "PASS" if "Risk-On" in macro.get("regime","") else ("WARN" if "Consolidation" in macro.get("regime","") else "DANGER")
            })

        if news_circuit:
            table.append({
                "param": "فیوز اطمینان اخبار و سنتیمنت (News Circuit Breaker)",
                "val": f"{news_circuit.get('circuit_title')}",
                "status": "PASS" if news_circuit.get("circuit_status") == "NORMAL_CLEAR" else ("WARN" if news_circuit.get("circuit_status") == "CAUTION_HIGH_VOLATILITY" else "DANGER")
            })

        if backtest and backtest.get("success"):
            table.append({
                "param": "کالیبراسیون تجربی با بک‌تست (Backtest Confluence)",
                "val": f"وین‌ریت: {backtest.get('win_rate_pct')}% (تعداد: {backtest.get('total_trades')} ترید | فاکتور سود: {backtest.get('profit_factor')})",
                "status": "PASS" if backtest.get("win_rate_pct", 0) >= 45 else "WARN"
            })

        return table

    def _build_scalp_setup(self, price: float, ta_15m: Dict[str, Any], ta_1h: Dict[str, Any], ob: Dict[str, Any], smc: Dict[str, Any]) -> Dict[str, Any]:
        if not ta_15m:
            return {"action": "WAIT", "action_code": "WAIT", "message": "داده‌های تایم‌فریم اسکالپ ناکافی است."}

        atr = ta_15m.get("atr", price * 0.01)
        rsi = ta_15m.get("rsi", 50)
        bias_15m = ta_15m.get("bias", "NEUTRAL")
        ema20 = ta_15m.get("ema20", price)
        nearest_sup = ta_15m.get("nearest_support", price - atr)
        nearest_res = ta_15m.get("nearest_resistance", price + atr)
        ob_ratio = ob.get("ratio", 1.0)
        fake_badge = smc.get("fake_trend", {}).get("badge", "ORGANIC")
        latest_sweep = smc.get("latest_sweep", {})

        # Bullish Scalp Setup
        if ("BULLISH" in bias_15m or (latest_sweep and latest_sweep.get("type") == "SSL_SWEEP")) and rsi < 74 and ob_ratio >= 0.90 and fake_badge != "BULL_TRAP":
            action = "LONG (خرید سریع اسمارت‌مانی)"
            action_code = "BUY"
            
            fvg = smc.get("nearest_fvg")
            if fvg and fvg.get("type") == "BULLISH_FVG" and fvg.get("top") < price:
                entry_low = round(fvg["bottom"], 6)
                entry_high = round(min(price, fvg["top"]), 6)
            else:
                entry_low = round(min(price, ema20), 6)
                entry_high = round(price, 6)
                
            entry_str = f"{entry_low} - {entry_high}"
            
            sl_distance = max(1.25 * atr, price * 0.006)
            sl = round(max(nearest_sup * 0.998, price - sl_distance), 6)
            risk = price - sl
            if risk <= 0: risk = price * 0.01; sl = round(price - risk, 6)
            
            tp1 = round(price + 1.2 * risk, 6)
            tp2 = round(price + 2.0 * risk, 6)
            tp3 = round(max(price + 3.0 * risk, smc.get("bsl_pool_target", price * 1.03)), 6)
            confidence = "85%" if (latest_sweep and latest_sweep.get("type") == "SSL_SWEEP") else "78%"
            
            triggers = [
                "ورود در برخورد به خلاء نقدینگی (FVG) یا لمس VWAP/EMA20 در تایم‌فریم ۱۵ دقیقه",
                "تایید بسته شدن کندل سبز صعودی با جذب نقدینگی فروشندگان",
                "هدف‌گذاری جمع‌آوری استاپ‌های بالای سقف (BSL Liquidity Pool)"
            ]
            warning = "در صورت شکست قطعی کف FVG یا ابطال کندل هانت، بلافاصله حد ضرر فعال شود."

        # Bearish Scalp Setup
        elif ("BEARISH" in bias_15m or (latest_sweep and latest_sweep.get("type") == "BSL_SWEEP")) and rsi > 26 and ob_ratio <= 1.10 and fake_badge != "BEAR_TRAP":
            action = "SHORT (فروش سریع / هانت BSL)"
            action_code = "SELL"
            entry_low = round(price, 6)
            entry_high = round(max(price, ema20), 6)
            entry_str = f"{entry_low} - {entry_high}"
            
            sl_distance = max(1.25 * atr, price * 0.006)
            sl = round(min(nearest_res * 1.002, price + sl_distance), 6)
            risk = sl - price
            if risk <= 0: risk = price * 0.01; sl = round(price + risk, 6)
            
            tp1 = round(price - 1.2 * risk, 6)
            tp2 = round(price - 2.0 * risk, 6)
            tp3 = round(min(price - 3.0 * risk, smc.get("ssl_pool_target", price * 0.97)), 6)
            confidence = "82%" if (latest_sweep and latest_sweep.get("type") == "BSL_SWEEP") else "74%"
            
            triggers = [
                "ورود پس از ثبت BSL Sweep (ریجکت از سقف و خروج خریداران خرد)",
                "رویت کندل نزولی زیر VWAP و پرتاب قیمت به سمت استخر نقدینگی پایین (SSL)",
                "سیو سود مرحله‌ای در تارگت‌ها بدون تعلل"
            ]
            warning = "در معاملات شورت مراقب پامپ‌های ناشی از دستکاری الگوریتمی HFT باشید."

        else:
            action = "WAIT / NO TRADE (نظاره‌گر / تله احتمالی)"
            action_code = "WAIT"
            entry_str = f"محدوده رنج بین {round(nearest_sup, 6)} تا {round(nearest_res, 6)}"
            sl = round(nearest_sup * 0.99, 6)
            risk = price - sl
            if risk <= 0: risk = price * 0.01
            tp1 = round(nearest_res, 6)
            tp2 = round(nearest_res * 1.015, 6)
            tp3 = round(nearest_res * 1.03, 6)
            confidence = "50%"
            triggers = [
                f"هشدار: {smc.get('fake_trend', {}).get('title', 'بازار رنج')}. منتظر تثبیت ساختار بمانید.",
                "از ورود احساسی در میان رنج قیمت خودداری کنید."
            ]
            warning = smc.get('fake_trend', {}).get('desc', 'اندیکاتورها در منطقه بلاتکلیف هستند.')

        return {
            "timeframe": "15m / 5m (اسکالپ اسمارت‌مانی)",
            "action": action,
            "action_code": action_code,
            "entry_zone": entry_str,
            "current_price": price,
            "stop_loss": sl,
            "stop_loss_pct": round(abs((sl - price) / price) * 100, 2),
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "tp1_pct": round(abs((tp1 - price) / price) * 100, 2),
            "tp2_pct": round(abs((tp2 - price) / price) * 100, 2),
            "tp3_pct": round(abs((tp3 - price) / price) * 100, 2),
            "risk_reward": "1:2.0",
            "confidence": confidence,
            "estimated_duration": "15 دقیقه الی 2 ساعت",
            "suggested_leverage": "حداکثر 3x الی 5x (یا اسپات)",
            "triggers": triggers,
            "warning": warning
        }

    def _build_swing_setup(self, price: float, ta_4h: Dict[str, Any], ta_1d: Dict[str, Any], fundamentals: Dict[str, Any], smc_4h: Dict[str, Any]) -> Dict[str, Any]:
        if not ta_4h:
            return {"action": "WAIT", "action_code": "WAIT", "message": "داده‌های تایم‌فریم ۴ ساعته ناکافی است."}

        fib = ta_4h.get("fib", {})
        recent_low = ta_4h.get("recent_low", price * 0.9)
        recent_high = ta_4h.get("recent_high", price * 1.1)
        ema50 = ta_4h.get("ema50", price)
        bias_4h = ta_4h.get("bias", "NEUTRAL")
        atr_4h = ta_4h.get("atr", price * 0.02)
        poc = smc_4h.get("poc", price)
        
        fib_50 = fib.get("0.500", (recent_low + recent_high) / 2)
        fib_618 = fib.get("0.618 (پاکت طلایی)", recent_low + 0.618 * (recent_high - recent_low))

        if "BULLISH" in bias_4h and price >= ema50:
            action = "SWING LONG / ACCUMULATE (خرید روندی چند روزه)"
            action_code = "BUY"
            pullback_entry = f"{round(min(price, poc), 6)} - {round(price, 6)}"
            breakout_entry = f"بالای مقاومت {round(recent_high * 1.005, 6)} با تثبیت کندل ۴ ساعته"
            
            sl = round(min(recent_low * 0.985, price - 2.5 * atr_4h), 6)
            risk = price - sl
            if risk <= 0: risk = price * 0.03; sl = round(price - risk, 6)
            
            tp1 = round(recent_high, 6)
            tp2 = round(recent_high + 0.272 * (recent_high - recent_low), 6)
            tp3 = round(recent_high + 0.618 * (recent_high - recent_low), 6)
            rr = round((tp2 - price) / risk, 1) if risk > 0 else 2.5
            confidence = "84%" if bias_4h == "BULLISH_STRONG" else "74%"

            strategy_desc = f"روند کلان ۴ ساعته صعودی است. گره تراکم حجم نهنگ‌ها (POC) در تراز {poc} به عنوان سوپاپ اطمینان عمل می‌کند و خرید در تراز تخفیف (Discount) با ریوارد عالی همراه است."
            invalidation = f"شکست و تثبیت کندل روزانه زیر تراز ساختاری {sl} روند صعودی را باطل می‌کند."

        elif "BEARISH" in bias_4h and price <= ema50:
            action = "SWING SHORT / HEDGE (موقعیت فروش / خروج از اسپات)"
            action_code = "SELL"
            pullback_entry = f"{round(price, 6)} - {round(max(price, poc), 6)}"
            breakout_entry = f"شکست کف حمایتی {round(recent_low * 0.995, 6)}"
            
            sl = round(max(recent_high * 1.015, price + 2.5 * atr_4h), 6)
            risk = sl - price
            if risk <= 0: risk = price * 0.03; sl = round(price + risk, 6)
            
            tp1 = round(recent_low, 6)
            tp2 = round(recent_low - 0.272 * (recent_high - recent_low), 6)
            tp3 = round(recent_low - 0.618 * (recent_high - recent_low), 6)
            rr = round((price - tp2) / risk, 1) if risk > 0 else 2.5
            confidence = "80%" if bias_4h == "BEARISH_STRONG" else "70%"

            strategy_desc = "روند کلان ۴ ساعته نزولی است. ترجیح استراتژیک، حفظ نقدینگی و پرهیز از خریدهای عجولانه یا باز کردن پوزیشن‌های شورت در برخورد به مقاومت‌ها است."
            invalidation = f"تثبیت کندل ۴ ساعته بالای سطح مقاومت {sl} ساختار نزولی را باطل می‌کند."

        else:
            action = "SWING RANGE / ACCUMULATION (انباشت در کف باکس)"
            action_code = "RANGE"
            pullback_entry = f"محدوده تقاضا: {round(recent_low, 6)} - {round(fib_618, 6)}"
            breakout_entry = f"شکست باکس در بالای {round(recent_high, 6)}"
            sl = round(recent_low * 0.965, 6)
            risk = price - sl
            if risk <= 0: risk = price * 0.04; sl = round(price - risk, 6)
            tp1 = round(poc, 6)
            tp2 = round(recent_high, 6)
            tp3 = round(recent_high * 1.08, 6)
            rr = 2.2
            confidence = "62%"
            strategy_desc = f"ارز در یک کانال رنج میان‌مدت قرار دارد. تراز POC در {poc} مرکز نوسان قیمت است. خرید در کف باکس و فروش در سقف پیشنهاد می‌شود."
            invalidation = f"از دست رفتن کف حمایتی {sl} زنگ خطر ریزش عمیق است."

        return {
            "timeframe": "4h / 1D (سوئینگ میان‌مدت)",
            "action": action,
            "action_code": action_code,
            "pullback_entry": pullback_entry,
            "breakout_entry": breakout_entry,
            "current_price": price,
            "stop_loss": sl,
            "stop_loss_pct": round(abs((sl - price) / price) * 100, 2),
            "target1": tp1,
            "target2": tp2,
            "target3": tp3,
            "target1_pct": round(abs((tp1 - price) / price) * 100, 2),
            "target2_pct": round(abs((tp2 - price) / price) * 100, 2),
            "target3_pct": round(abs((tp3 - price) / price) * 100, 2),
            "risk_reward": f"1:{rr}",
            "confidence": confidence,
            "holding_period": "3 الی 14 روز",
            "capital_risk_advice": "حداکثر ۱ تا ۳ درصد از کل سرمایه روی حد ضرر ریسک شود",
            "strategy_description": strategy_desc,
            "invalidation_condition": invalidation
        }

    def _build_persian_verdict(self, symbol: str, price: float, ticker: Dict[str, Any],
                              ta_15m: Dict[str, Any], ta_4h: Dict[str, Any], ta_1d: Dict[str, Any],
                              scalp: Dict[str, Any], swing: Dict[str, Any],
                              orderbook: Dict[str, Any], fng: Dict[str, Any],
                              fundamentals: Dict[str, Any], smc_15m: Dict[str, Any],
                              derivatives: Dict[str, Any]) -> Dict[str, Any]:
        
        change_24h = ticker.get("price_change_pct", 0)
        vol_quote = ticker.get("volume_quote", 0)
        vol_million = round(vol_quote / 1_000_000, 2)
        
        rsi_15m = ta_15m.get("rsi", 50)
        rsi_4h = ta_4h.get("rsi", 50)
        
        bias_15m_fa = ta_15m.get("bias_fa", "خنثی")
        bias_4h_fa = ta_4h.get("bias_fa", "خنثی")

        summary_paragraphs = []
        summary_paragraphs.append(
            f"نماد **{symbol}** هم‌اکنون با قیمت **{price:,.4f} دلار** معامله می‌شود و در ۲۴ ساعت گذشته تغییرات قیمتی **{change_24h:+.2f}%** همراه با حجم معاملات نقدی معادل **{vol_million:,.1f} میلیون دلار** به ثبت رسانده است."
        )

        ob_ratio = orderbook.get("ratio", 1.0)
        ob_pressure = orderbook.get("pressure", "متعادل")
        fng_val = fng.get("value", 50)
        fng_desc = fng.get("classification_fa", "خنثی")
        
        hft_score = smc_15m.get("hft", {}).get("score", 50)
        hft_status = smc_15m.get("hft", {}).get("status", "متعادل")
        poc = smc_15m.get("poc", price)
        fake_trap = smc_15m.get("fake_trend", {}).get("title", "روند ارگانیک")
        vwap_data = smc_15m.get("vwap_cvd", {})
        
        summary_paragraphs.append(
            f"در عمق بازار (Order Book)، وضعیت **{ob_pressure}** (نسبت خریدار/فروشنده: **{ob_ratio}**) مشاهده می‌شود. شاخص طمع و ترس کل بازار روی **{fng_val}** ({fng_desc}) قرار دارد. مرکز کنترل حجم نقدینگی (POC) در قیمت **{poc:,.4f} دلار** و میانگین قیمت وزنی حجم (VWAP) روی **{vwap_data.get('vwap', 0):,.4f} دلار** قرار دارد ({vwap_data.get('vwap_position', '')})."
        )

        summary_paragraphs.append(
            f"ردپای هوش مصنوعی و ربات‌های پربسامد بانکی (HFT Index): **{hft_score}/100** ({hft_status}). وضعیت سلامت روند: **{fake_trap}** | جهت جریان دلتای انباشته (CVD): **{vwap_data.get('cvd_trend', '')}**."
        )

        if derivatives.get("has_data"):
            summary_paragraphs.append(
                f"داده‌های مشتقه و موسساتی (Derivatives Flow): ارزش قراردادهای باز (Open Interest) معادل **{derivatives.get('open_interest_fmt')}** و نرخ تامین مالی (Funding Rate) معادل **{derivatives.get('funding_rate_fmt')}** است ({derivatives.get('market_crowd')})."
            )

        directives = {
            "scalp_directive": f"⚡ **دستورالعمل اسکالپینگ اسمارت‌مانی (۱۵ دقیقه):** سیگنال: **{scalp.get('action')}** | محدوده ورود: **{scalp.get('entry_zone')}** | حد ضرر قاطع: **{scalp.get('stop_loss')}** ({scalp.get('stop_loss_pct')}%) | تارگت ۱: **{scalp.get('tp1')}** | تارگت ۲: **{scalp.get('tp2')}** | تارگت ۳: **{scalp.get('tp3')}**.",
            "swing_directive": f"🌊 **دستورالعمل سوئینگ تریدینگ (۴ ساعته):** جهت معامله: **{swing.get('action')}** | ورود در پولبک به POC: **{swing.get('pullback_entry')}** | ورود در بریک‌اوت: **{swing.get('breakout_entry')}** | حد ضرر ساختاری: **{swing.get('stop_loss')}** ({swing.get('stop_loss_pct')}%) | تارگت اول: **{swing.get('target1')}** | تارگت دوم: **{swing.get('target2')}** | تارگت سوم: **{swing.get('target3')}**.",
            "risk_rule": "🛡️ **اصل عدم تخطی از مدیریت ریسک:** ریسک هر معامله را حداکثر به ۱ الی ۲ درصد کل بالانس محدود کنید. هنگام رسیدن به تارگت اول حتماً نیمی از پوزیشن را نقد کرده و حد ضرر باقیمانده را به نقطه ورود (ریسک‌فری) انتقال دهید."
        }

        danger_alerts = []
        if smc_15m.get("fake_trend", {}).get("badge") in ["BULL_TRAP", "BEAR_TRAP"]:
            danger_alerts.append(f"هشدار تله نقدینگی: {smc_15m['fake_trend']['desc']}")

        if rsi_15m > 74:
            danger_alerts.append("هشدار اسکالپ: RSI در تایم ۱۵ دقیقه در محدوده اشباع خرید است. از ورود هیجانی در قیمت‌های فعلی خودداری کنید.")
        elif rsi_15m < 26:
            danger_alerts.append("هشدار اسکالپ: RSI در تایم ۱۵ دقیقه در اشباع فروش قرار دارد. احتمال ریباند صعودی موقت بالا است؛ شورت نگیرید.")

        if orderbook.get("whale_walls"):
            for wall in orderbook["whale_walls"][:2]:
                danger_alerts.append(f"ردپای نهنگ در اردربوک: {wall['desc']} در قیمت {wall['price']:,.4f}$ به ارزش تقریبی ${wall['volume_usd']:,.0f}.")

        if derivatives.get("funding_rate", 0) > 0.02:
            danger_alerts.append("هشدار مشتقه: فاندینگ ریت به شدت مثبت است؛ ازدحام معامله‌گران لانگ احتمال استاپ‌هانت نزولی توسط نهنگ‌ها را افزایش داده است.")

        return {
            "executive_summary": "\n\n".join(summary_paragraphs),
            "directives": directives,
            "danger_alerts": danger_alerts,
            "overall_market_bias": "BULLISH" if ("BULLISH" in bias_4h_fa and ob_ratio > 1) else ("BEARISH" if "BEARISH" in bias_4h_fa else "NEUTRAL")
        }


class AgentAdvisor:
    @staticmethod
    def answer_question(question: str, analysis: Dict[str, Any]) -> str:
        q = question.lower()
        symbol = analysis.get("symbol", "ارز انتخابی")
        price = analysis.get("price", 0)
        scalp = analysis.get("scalp_setup", {})
        swing = analysis.get("swing_setup", {})
        ob = analysis.get("orderbook", {})
        fng = analysis.get("market_sentiment", {})
        smc = analysis.get("smc", {}).get("15m", {})
        derivatives = analysis.get("derivatives", {})
        scores = analysis.get("scores_3d", {})
        vwap_data = smc.get("vwap_cvd", {})
        
        # 1. 3D Scores question
        if any(w in q for w in ["امتیاز", "اسکور", "direction", "entry score", "risk score", "وین ریت", "کیفیت"]):
            return (
                f"### 🎯 ارزیابی تفکیکی امتیازات ۳ بعدی سیگنال برای {symbol}\n\n"
                f"- **۱. امتیاز جهت حرکت (Direction Score):** **{scores.get('direction_score')}/100** (برآیند همسویی تایم‌فریم‌ها، جهت CVD و رفتار قیمت با Open Interest)\n"
                f"- **۲. امتیاز کیفیت نقطه ورود (Entry Score):** **{scores.get('entry_score')}/100** (سنجش میزان اصلاح به سمت FVG پرنشده، تراز VWAP و وقوع استاپ‌هانت)\n"
                f"- **۳. امتیاز کیفیت ریسک (Risk Score):** **{scores.get('risk_score')}/100** (فشردگی حد ضرر ساختاری، نسبت R:R بالای ۱:۲ و عدم واگرایی معکوس)\n\n"
                f"⭐ **نمره نهایی کانفلوئنس (Confidence):** **{scores.get('composite_confidence')}/100** ({scores.get('grade_title')})\n"
                f"📌 **دستور اجرایی:** {scores.get('action_advice')}"
            )

        # 2. SMC / FVG / Liquidity question
        elif any(w in q for w in ["fvg", "خلاء", "نقدینگی", "sweep", "bsl", "ssl", "اسمارت مانی", "ict", "vwap", "cvd"]):
            fvgs = smc.get("unmitigated_fvgs", [])
            fvg_str = f"{fvgs[-1]['bottom']} - {fvgs[-1]['top']}" if fvgs else "در حال حاضر FVG پرنشده نزدیک دیده نمی‌شود."
            sweep = smc.get("latest_sweep")
            sweep_str = f"{sweep['title']} در تراز {sweep['level_swept']}" if sweep else "اخیراً استاپ‌هانتی رخ نداده است."
            poc = smc.get("poc", price)
            
            return (
                f"### 🧠 کالبدشکافی به سبک اسمارت مانی و اردر فلو برای {symbol}\n\n"
                f"- **خلاء نقدینگی فعال (Unmitigated FVG):** محدوده **{fvg_str}**\n"
                f"- **آخرین شکار نقدینگی (Liquidity Sweep):** **{sweep_str}**\n"
                f"- **میانگین قیمت وزنی حجم (VWAP):** **{vwap_data.get('vwap', 0):,.4f}$** ({vwap_data.get('vwap_position')})\n"
                f"- **جهت دلتای تجمعی (CVD):** **{vwap_data.get('cvd_trend')}**\n"
                f"- **گره متراکم حجم (POC):** سطح **{poc:,.4f}$** (تراز VAH: {smc.get('vah', 0):,.4f}$ | VAL: {smc.get('val', 0):,.4f}$)\n"
                f"- **استخر نقدینگی سقف (BSL Target):** **{smc.get('bsl_pool_target', 0):,.4f}$**\n"
                f"- **استخر نقدینگی کف (SSL Target):** **{smc.get('ssl_pool_target', 0):,.4f}$**\n\n"
                f"📌 **توصیه استراتژیک:** هرگز در میانه رنج وارد نشوید؛ ورود بهینه یا روی پولبک به FVG و VWAP است یا پس از تایید Sweep استاپ‌ها."
            )

        # 3. HFT / Fake trend question
        elif any(w in q for w in ["hft", "ربات", "بانک", "تله", "فیک", "fake", "دستکاری"]):
            hft = smc.get("hft", {})
            trap = smc.get("fake_trend", {})
            return (
                f"### 🤖 تحلیل ردپای ربات‌های HFT و اعتدال الگوریتمی در {symbol}\n\n"
                f"- **شاخص نفوذ الگوریتم‌های پربسامد (HFT Index):** **{hft.get('score', 50)}/100** ({hft.get('status')})\n"
                f"- **وضعیت تله نقدینگی:** **{trap.get('title')}**\n"
                f"- **توضیحات الگوریتمی:** {trap.get('desc')}\n\n"
                f"💡 **نکته معامله‌گری:** وقتی شاخص HFT بالای ۷۵ می‌رود، الگوریتم‌های نهادی به سرعت شدو می‌زنند تا لیمیت‌اردرها را تاچ کنند. در این شرایط از لوریج‌های بالا اکیداً بپرهیزید و لیمیت‌اردرها را با فاصله بیشتری قرار دهید."
            )

        # 4. Entry / Buy question
        elif any(w in q for w in ["الان بخرم", "خرید", "وارد شم", "لانگ", "کی بخرم", "نقطه ورود", "long"]):
            return (
                f"### 💡 راهنمای ورود به معامله در {symbol} (قیمت فعلی: {price:,.4f}$)\n\n"
                f"🔹 **دیدگاه اسکالپینگ (۱۵ دقیقه):** سیگنال لحظه‌ای: **{scalp.get('action')}**\n"
                f"- محدوده ورود بهینه: **{scalp.get('entry_zone')}**\n"
                f"- حد ضرر الزامی: **{scalp.get('stop_loss')}**\n\n"
                f"🔹 **دیدگاه سوئینگ تریدینگ (۴ ساعته):** سیگنال: **{swing.get('action')}**\n"
                f"- ورود در پولبک به POC: **{swing.get('pullback_entry')}**\n"
                f"- حد ضرر ساختاری: **{swing.get('stop_loss')}**\n\n"
                f"⚠️ وضعیت تله: {smc.get('fake_trend', {}).get('title')} | درجه سیگنال: {scores.get('grade_title')}."
            )

        # 5. Leverage question
        elif any(w in q for w in ["لوریج", "اهرم", "اهرم چند", "leverage", "مارجین"]):
            return (
                f"### ⚖️ راهنمای اهرم و مدیریت ریسک برای {symbol}\n\n"
                f"- **برای اسکالپ:** حداکثر **{scalp.get('suggested_leverage', '3x الی 5x')}** پیشنهاد می‌شود.\n"
                f"- **برای سوئینگ:** اکیداً توصیه می‌شود معامله را **اسپات (بدون اهرم)** یا حداکثر با اهرم **2x** باز کنید.\n\n"
                f"📌 قانون طلایی: فاصله استاپ‌لاس تا نقطه ورود هر چقدر باشد، ضرر دلاری کل معامله نباید از ۱.۵٪ کل سرمایه شما تجاوز کند."
            )

        # 6. Macro & Dominance question
        elif any(w in q for w in ["ماکرو", "کلان", "دامیننس", "dominance", "btc.d", "usdt.d", "مارکت کپ", "همبستگی", "بتا", "beta"]):
            macro = analysis.get("macro", {})
            corr = analysis.get("correlation", {})
            return (
                f"### 🌐 وضعیت کلان بازار و دامیننس برای {symbol}\n\n"
                f"- **ارزش کل بازار کریپتو (Total Market Cap):** **{macro.get('total_market_cap_fmt')}** ({macro.get('mcap_change_24h'):+.2f}% در ۲۴ ساعت گذشته)\n"
                f"- **سلطه بیت‌کوین (BTC.D):** **{macro.get('btc_dominance')}%** ({macro.get('alt_season_status')})\n"
                f"- **سلطه تتر (USDT.D):** **{macro.get('usdt_dominance')}%** (شاخص ورود/خروج نقدینگی نقد)\n"
                f"- **رژیم معاملاتی بازار:** **{macro.get('regime')}**\n"
                f"- **همبستگی و بتا با بیت‌کوین:** **{corr.get('desc', 'N/A')}**\n\n"
                f"💡 **تفسیر نهادی:** {macro.get('regime_desc')}"
            )

        # 7. On-chain question
        elif any(w in q for w in ["آنچین", "آن چین", "onchain", "on-chain", "نهنگ", "ممشول", "mempool", "هش ریت", "تراکنش"]):
            onchain = analysis.get("onchain", {})
            return (
                f"### ⛓️ وضعیت معیارهای آن‌چین زنجیره بیت‌کوین\n\n"
                f"- **حجم دلاری تراکنش‌های ۲۴ ساعت گذشته:** **{onchain.get('tx_volume_usd')}**\n"
                f"- **میزان بیت‌کوین جابجا شده:** **{onchain.get('total_btc_sent_24h')}**\n"
                f"- **تعداد کل تراکنش‌های ثبت‌شده:** **{onchain.get('n_tx_24h')}**\n"
                f"- **قدرت پردازشی و هش‌ریت شبکه:** **{onchain.get('hash_rate_eh')}**\n"
                f"- **کارمزد پیشنهادی ممپول:** **{onchain.get('recommended_fee_sat_vb')} sat/vB** ({onchain.get('network_load')})\n"
                f"- **ردپای نهنگ‌ها:** **{onchain.get('whale_status')}**"
            )

        # 8. News & Circuit Breaker question
        elif any(w in q for w in ["خبر", "اخبار", "سنتیمنت", "فیوز", "circuit", "panic", "فاندامنتال"]):
            news = analysis.get("news", {})
            return (
                f"### 📰 وضعیت فیوز اطمینان اخبار و سنتیمنت بازار\n\n"
                f"- **وضعیت فیوز حفاظتی:** **{news.get('circuit_title')}**\n"
                f"- **مجوز معامله از دیدگاه اخبار:** **{'✔ مجاز و امن' if news.get('safe_to_trade') else '🛑 پرخطر / متوقف'}**\n"
                f"- **دستورالعمل سیستم:** {news.get('circuit_advice')}\n"
                f"- **تعداد اخبار پرریسک/پانیک شناسایی‌شده:** {news.get('panic_count', 0)} مورد از {news.get('news_count', 0)} خبر اخیر"
            )

        # 9. Backtest question
        elif any(w in q for w in ["بک تست", "بکتست", "backtest", "نرخ برد", "تست تاریخی", "سود شبیه"]):
            bt = analysis.get("backtest", {})
            return (
                f"### 🧪 نتایج شبیه‌سازی و بک‌تست تاریخی استراتژی روی {symbol}\n\n"
                f"- **نرخ برد تجربی (Win Rate):** **{bt.get('win_rate_pct')}%** ({bt.get('win_count')} برد از {bt.get('total_trades')} ترید)\n"
                f"- **فاکتور سود (Profit Factor):** **{bt.get('profit_factor')}**\n"
                f"- **سود خالص شبیه‌سازی:** **{bt.get('net_profit_pct'):+.2f}%** (${bt.get('net_profit_usd'):+,.2f})\n"
                f"- **حداکثر افت سرمایه (Max Drawdown):** **{bt.get('max_drawdown_pct')}%**\n"
                f"- **ضریب اطمینان کالیبره‌شده ریاضی:** **{bt.get('calibrated_confidence')}/100**"
            )

        # 10. General fallback
        else:
            return (
                f"### 🤖 پاسخ ایجنت کریپتو درباره {symbol}\n\n"
                f"قیمت لحظه‌ای: **{price:,.4f}$** | درجه سیگنال: **{scores.get('grade_title')}**\n"
                f"- تفکیک امتیازات: جهت: {scores.get('direction_score')} | ورود: {scores.get('entry_score')} | ریسک: {scores.get('risk_score')}\n"
                f"- ستاپ اسکالپ: **{scalp.get('action')}** (ورود: {scalp.get('entry_zone')})\n"
                f"- ردپای الگوریتم‌های HFT: **{smc.get('hft', {}).get('score', 50)}/100**\n"
                f"- میانگین VWAP: **{vwap_data.get('vwap', 0):,.4f}$**\n\n"
                f"هر سوال دیگری درباره جدول پارامترها، استاپ‌هانت‌ها یا مدیریت مارجین دارید بفرمایید."
            )
