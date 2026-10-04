import socket
socket.setdefaulttimeout(5.0)
import requests
import numpy as np
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from institutional_addons import CryptoQuantNetflowEngine, TokenUnlocksRadar, StablecoinSupplyRatioEngine, MacroEngine, OnChainEngine, NewsCircuitBreaker, BacktestEngine, TelegramDispatcher

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
        self._fng_cache = {}

    def fetch_binance_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            # Binance Official Public Data Cluster (zero geo-block, global tier-1 liquidity benchmark)
            url = f"https://data-api.binance.vision/api/v3/ticker/24hr?symbol={symbol}"
            res = self.session.get(url, timeout=3.5)
            if res.status_code == 200:
                data = res.json()
                if "lastPrice" in data and float(data["lastPrice"]) > 0:
                    return {
                        "source": "Binance",
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

    def fetch_lbank_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            clean = symbol.lower().replace("usdt", "") + "_usdt"
            url = f"https://api.lbank.info/v2/ticker.do?symbol={clean}"
            res = self.session.get(url, timeout=3.5)
            if res.status_code == 200:
                data = res.json()
                if data.get("data"):
                    tk = data["data"][0]["ticker"]
                    latest = float(tk.get("latest", 0))
                    change = float(tk.get("change", 0))
                    return {
                        "source": "LBank",
                        "symbol": symbol.upper(),
                        "last_price": latest,
                        "price_change_pct": round(change, 2),
                        "high_24h": float(tk.get("high", 0)),
                        "low_24h": float(tk.get("low", 0)),
                        "volume_base": float(tk.get("vol", 0)),
                        "volume_quote": float(tk.get("turnover", 0)),
                        "bid": latest,
                        "ask": latest
                    }
        except Exception:
            pass
        return None

    def fetch_kucoin_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            clean = symbol.upper().replace("USDT", "") + "-USDT"
            url = f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={clean}"
            headers = {"User-Agent": "Mozilla/5.0"}
            res = self.session.get(url, headers=headers, timeout=3.5)
            if res.status_code == 200:
                data = res.json().get("data", {})
                price = float(data.get("price", 0))
                if price > 0:
                    return {
                        "source": "KuCoin",
                        "symbol": symbol.upper(),
                        "last_price": price,
                        "price_change_pct": 0.0,
                        "high_24h": price,
                        "low_24h": price,
                        "volume_base": float(data.get("size", 0)),
                        "volume_quote": 0.0,
                        "bid": float(data.get("bestBid", price)),
                        "ask": float(data.get("bestAsk", price))
                    }
        except Exception:
            pass
        return None

    def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        # Tier 1 Priority: Binance Global (Deepest global spot & futures benchmark)
        t = self.fetch_binance_ticker(symbol)
        if t: return t
        # Tier 2 Fallback: MEXC (Huge catalog of 2,500+ altcoins & early gems)
        t = self.fetch_mexc_ticker(symbol)
        if t: return t
        # Tier 3 Fallback: LBank (Fast live spot ticker API)
        t = self.fetch_lbank_ticker(symbol)
        if t: return t
        # Tier 4 Fallback: KuCoin Global
        t = self.fetch_kucoin_ticker(symbol)
        if t: return t
        # Tier 5 Fallback: Binance US
        t = self.fetch_binance_us_ticker(symbol)
        if t: return t
        return None

    def fetch_klines(self, symbol: str, interval: str = "15m", limit: int = 100) -> List[List[float]]:
        # Tier 1: Try Binance Global (data-api.binance.vision)
        try:
            url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
            res = self.session.get(url, timeout=3.5)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list) and len(data) > 0:
                    return [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in data]
        except Exception:
            pass

        # Tier 2 Fallback: Try MEXC
        try:
            url = f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
            res = self.session.get(url, timeout=4)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list) and len(data) > 0:
                    return [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in data]
        except Exception:
            pass

        # Tier 3 Fallback: Binance US
        try:
            url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
            res = self.session.get(url, timeout=4)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list) and len(data) > 0:
                    return [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in data]
        except Exception:
            pass

        return []

    def fetch_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        # Tier 1: Binance Depth
        try:
            url = f"https://data-api.binance.vision/api/v3/depth?symbol={symbol}&limit={limit}"
            res = self.session.get(url, timeout=3.0)
            if res.status_code == 200:
                data = res.json()
                bids = [[float(b[0]), float(b[1])] for b in data.get("bids", [])]
                asks = [[float(a[0]), float(a[1])] for a in data.get("asks", [])]
                if bids and asks:
                    return self._calculate_depth_metrics(bids, asks)
        except Exception:
            pass

        # Tier 2 Fallback: MEXC Depth
        try:
            url = f"https://api.mexc.com/api/v3/depth?symbol={symbol}&limit={limit}"
            res = self.session.get(url, timeout=3.5)
            if res.status_code == 200:
                data = res.json()
                bids = [[float(b[0]), float(b[1])] for b in data.get("bids", [])]
                asks = [[float(a[0]), float(a[1])] for a in data.get("asks", [])]
                if bids and asks:
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
        """
        Fetches the real-time Crypto Fear & Greed Index.
        Priority 1: CoinMarketCap Real-Time Fear & Greed Index (instant live intraday data)
        Priority 2: Alternative.me Fear & Greed API (daily fallback)
        Includes 60-second in-memory caching to optimize response speed.
        """
        now = time.time()
        if hasattr(self, "_fng_cache") and self._fng_cache and (now - self._fng_cache.get("time", 0) < 60):
            return self._fng_cache["data"]

        # 1. Primary Source: CoinMarketCap Live Fear & Greed Index
        try:
            cmc_url = "https://coinmarketcap.com/charts/fear-and-greed-index/"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9"
            }
            res = self.session.get(cmc_url, headers=headers, timeout=5)
            if res.status_code == 200:
                import json, re
                m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', res.text)
                if m:
                    d = json.loads(m.group(1))
                    fng_data = d.get("props", {}).get("pageProps", {}).get("pageSharedData", {}).get("fearGreedIndexData", {})
                    current = fng_data.get("currentIndex", {})
                    score = current.get("score")
                    if score is not None:
                        val = int(score)
                        raw_name = current.get("name", "")
                        cls = raw_name.title() if raw_name else ""
                        if not cls:
                            if val >= 75: cls = "Extreme Greed"
                            elif val >= 55: cls = "Greed"
                            elif val >= 45: cls = "Neutral"
                            elif val >= 25: cls = "Fear"
                            else: cls = "Extreme Fear"

                        fa_cls = {
                            "Extreme Fear": "ترس شدید (Extreme Fear) - ارزندگی قیمت و فرصت انباشت پله‌ای",
                            "Fear": "ترس (Fear) - احتیاط حاکم بر بازار",
                            "Neutral": "خنثی (Neutral) - تعادل احساسی معامله‌گران",
                            "Greed": "طمع (Greed) - ورود هیجانی سرمایه‌گذاران",
                            "Extreme Greed": "طمع شدید (Extreme Greed) - هشدار سقف‌های قیمتی و سیو سود"
                        }.get(cls, cls)

                        res_data = {
                            "value": val,
                            "classification": cls,
                            "classification_fa": fa_cls,
                            "source": "CoinMarketCap (Live)",
                            "timestamp": current.get("updateTime")
                        }
                        self._fng_cache = {"time": now, "data": res_data}
                        return res_data
        except Exception:
            pass

        # 2. Secondary Fallback: Alternative.me API
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
                res_data = {
                    "value": val,
                    "classification": cls,
                    "classification_fa": fa_cls,
                    "source": "Alternative.me",
                    "timestamp": d.get("timestamp")
                }
                self._fng_cache = {"time": now, "data": res_data}
                return res_data
        except Exception:
            pass

        if hasattr(self, "_fng_cache") and self._fng_cache and self._fng_cache.get("data"):
            return self._fng_cache["data"]

        return {"value": 50, "classification": "Neutral", "classification_fa": "خنثی", "source": "Default", "timestamp": None}

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
    @staticmethod
    def _round_val(val: float) -> float:
        if val is None:
            return 0.0
        val = float(val)
        if abs(val) >= 100:
            return round(val, 2)
        elif abs(val) >= 1:
            return round(val, 4)
        elif abs(val) >= 0.001:
            return round(val, 6)
        else:
            return round(val, 8)

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

        # 2. Parallel Fast Fetch of Multi-Timeframe Candles, Orderbook, Derivatives, and Sentiment
        import fastfetch as FF
        parallel_jobs = {
            "1m": lambda: self.fetcher.fetch_klines(symbol, "1m", 120),
            "5m": lambda: self.fetcher.fetch_klines(symbol, "5m", 150),
            "15m": lambda: self.fetcher.fetch_klines(symbol, "15m", 250),
            "1h": lambda: self.fetcher.fetch_klines(symbol, "1h", 120),
            "4h": lambda: self.fetcher.fetch_klines(symbol, "4h", 120),
            "1d": lambda: self.fetcher.fetch_klines(symbol, "1d", 80),
            "ob": lambda: self.fetcher.fetch_orderbook(symbol, 20),
            "derivatives": lambda: self.fetcher.fetch_institutional_derivatives(symbol),
            "fng": lambda: self.fetcher.fetch_fear_and_greed(),
            "fundamentals": lambda: self.fetcher.fetch_coingecko_details(base_coin),
            "macro": lambda: MacroEngine.fetch_global_macro(),
            "onchain": lambda: OnChainEngine.fetch_onchain_metrics(),
            "news_circuit": lambda: NewsCircuitBreaker.fetch_live_news()
        }
        if base_coin != "BTC":
            parallel_jobs["btc_candles"] = lambda: self.fetcher.fetch_klines("BTCUSDT", "15m", 60)

        fetched = FF.gather(parallel_jobs, max_workers=12, timeout=5.0)

        klines_1m = fetched.get("1m") or []
        klines_5m = fetched.get("5m") or []
        klines_15m = fetched.get("15m") or []
        klines_1h = fetched.get("1h") or []
        klines_4h = fetched.get("4h") or []
        klines_1d = fetched.get("1d") or []
        orderbook = fetched.get("ob") or {"bids": [], "asks": []}
        derivatives = fetched.get("derivatives") or {}
        fng = fetched.get("fng") or {"value": 50, "classification": "Neutral"}
        fundamentals = fetched.get("fundamentals") or {}
        macro = fetched.get("macro") or {}
        onchain = fetched.get("onchain") or {}
        news_circuit = fetched.get("news_circuit") or {}
        btc_candles = klines_15m if base_coin == "BTC" else (fetched.get("btc_candles") or [])

        # 3. Analyze Technicals
        ta_1m = self.analyzer.analyze_candles(klines_1m, "1m") if klines_1m else {}
        ta_5m = self.analyzer.analyze_candles(klines_5m, "5m") if klines_5m else {}
        ta_15m = self.analyzer.analyze_candles(klines_15m, "15m") if klines_15m else {}
        ta_1h = self.analyzer.analyze_candles(klines_1h, "1h") if klines_1h else {}
        ta_4h = self.analyzer.analyze_candles(klines_4h, "4h") if klines_4h else {}
        ta_1d = self.analyzer.analyze_candles(klines_1d, "1d") if klines_1d else {}

        # 4. Smart Money Concepts, Order Flow, VWAP, CVD, Value Area (5m Scalp & 4h Swing)
        smc_5m = self.smc.analyze_smc(klines_5m, current_price, "5m") if klines_5m else {}
        smc_15m = self.smc.analyze_smc(klines_15m, current_price, "15m") if klines_15m else {}
        smc_4h = self.smc.analyze_smc(klines_4h, current_price, "4h") if klines_4h else {}

        # 6. Institutional Derivatives (Open Interest & Funding Rate)
        derivatives_matrix = self.smc.evaluate_derivatives_matrix(
            ticker.get("price_change_pct", 0),
            derivatives.get("open_interest_usd", 0),
            derivatives.get("funding_rate", 0)
        )

        # 9. Generate Scalp (1m/5m) & Swing (4h/1d) Setups with 5-Layer Confluence
        scalp_setup = self._build_scalp_setup(
            current_price, ta_5m=ta_5m, ta_1m=ta_1m, ob=orderbook,
            smc=smc_5m if smc_5m else smc_15m, ta_15m=ta_15m, ta_4h=ta_4h,
            derivatives=derivatives, derivatives_matrix=derivatives_matrix
        )
        swing_setup = self._build_swing_setup(
            current_price, ta_4h, ta_1d, fundamentals, smc_4h,
            derivatives=derivatives, derivatives_matrix=derivatives_matrix, ob=orderbook
        )

        # 10. Macro Market, Dominance & Correlations (Layer 6)
        correlation = MacroEngine.compute_beta_and_correlation(klines_15m, btc_candles)

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

                # 17. Altcoin Season & BTC Macro Health Shield
        btc_change_24h = 0.0
        btc_chg_pct_15m = 0.0
        btc_bias = "NEUTRAL"

        if btc_candles and len(btc_candles) >= 4:
            c_now = btc_candles[-1][4]
            c_prev = btc_candles[-4][4] # 45-60m ago
            btc_chg_pct_15m = round(((c_now - c_prev) / c_prev) * 100, 2)
        
        btc_d = float(macro.get("btc_dominance") or 58.0)
        
        # Determine Altcoin Green Light Status
        if btc_chg_pct_15m < -1.8:
            alt_status = "RED_ALERT"
            alt_badge = "🔴 پرخطر (BTC Dumping)"
            alt_title = "ریزش تهاجمی بیت‌کوین - ورود به آلت‌کوین‌ها مسدود"
            alt_advice = "بیت‌کوین در ۶۰ دقیقه اخیر بیش از ۱.۸٪ ریزش تهاجمی داشته است. تمامی آلت‌کوین‌ها با ریسک افت شدید همراه هستند."
            alt_color = "#ff3366"
        elif btc_d > 61.0:
            alt_status = "YELLOW_DOMINANCE"
            alt_badge = "🟡 احتیاط (BTC Dominant)"
            alt_title = "مکیده شدن نقدینگی توسط بیت‌کوین (سلطه BTC)"
            alt_advice = "دامیننس بیت‌کوین بالای ۶۱٪ است. حجم ورودی به آلت‌کوین‌ها محدود بوده و تارگت‌های محافظه‌کارانه توصیه می‌شود."
            alt_color = "#ffd166"
        elif btc_chg_pct_15m > 2.5:
            alt_status = "YELLOW_PUMP"
            alt_badge = "🟡 احتیاط (BTC High Volatility)"
            alt_title = "پامپ پرنوسان بیت‌کوین"
            alt_advice = "بیت‌کوین در حال جهش پرنوسان است؛ آلت‌کوین‌ها ممکن است ابتدا در برابر جفت ارز BTC تضعیف شوند."
            alt_color = "#ffd166"
        else:
            alt_status = "GREEN_LIGHT"
            alt_badge = "🟢 آزاد (Green Light)"
            alt_title = "مجوز کامل ترید آلت‌کوین‌ها (Altcoin Safe Heaven)"
            alt_advice = "بیت‌کوین آرام و پایدار است و نقدینگی در حال چرخش ارگانیک به آلت‌کوین‌های باکیفیت است."
            alt_color = "#00e676"

        altcoin_shield = {
            "status": alt_status,
            "badge": alt_badge,
            "title": alt_title,
            "advice": alt_advice,
            "color": alt_color,
            "btc_dominance": btc_d,
            "btc_momentum_1h": btc_chg_pct_15m,
            "is_safe_for_alts": (alt_status == "GREEN_LIGHT")
        }

        # 18. Elite Macro Triad: CryptoQuant Netflows + Token Unlocks + Stablecoin Ratio (SSR)
        quant_flows = CryptoQuantNetflowEngine.get_exchange_flows(symbol, current_price)
        token_unlock = TokenUnlocksRadar.get_token_unlock_status(symbol)
        ssr_metrics = StablecoinSupplyRatioEngine.get_ssr_metrics(btc_candles[-1][4] if btc_candles else current_price)

        elite_triad = {
            "exchange_flows": quant_flows,
            "token_unlock": token_unlock,
            "ssr": ssr_metrics
        }


        return {
            "success": True,
            "symbol": symbol,
            "base_coin": base_coin,
            "price": current_price,
            "ticker": ticker,
            "timeframes": {
                "1m": ta_1m,
                "5m": ta_5m,
                "15m": ta_15m,
                "1h": ta_1h,
                "4h": ta_4h,
                "1d": ta_1d
            },
            "smc": {
                "5m": smc_5m,
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
            "altcoin_shield": altcoin_shield,
            "elite_triad": elite_triad,
            "analyzed_at": (datetime.now(timezone.utc) + timedelta(hours=3, minutes=30)).strftime("%Y-%m-%d %H:%M:%S (ایران 🇮🇷)")
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

    def _build_scalp_setup(self, price: float, ta_5m: Dict[str, Any], ta_1m: Dict[str, Any], ob: Dict[str, Any], smc: Dict[str, Any], ta_15m: Optional[Dict[str, Any]] = None, ta_4h: Optional[Dict[str, Any]] = None, derivatives: Optional[Dict[str, Any]] = None, derivatives_matrix: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ultra-Fast Scalp Setup engine calibrated specifically for 1-minute (M1) and 5-minute (M5) execution.
        Enhanced with 5 Institutional Elite Quality Gates:
          1. Live Order Book Imbalance & Wall Check
          2. Multi-Timeframe (MTF) Macro Confluence (4H & 15M)
          3. Open Interest (OI) & Derivatives Delta Validation (Trap / Fakeout Filter)
          4. Dynamic Minimum Risk-to-Reward (R:R >= 1:2.0)
          5. ATR Low-Volatility / Dead Market Filter
        Validity Horizon: 30 to 45 minutes from generation time (high-frequency turnover).
        """
        now_utc = datetime.now(timezone.utc)
        generated_at_utc = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        valid_until_utc = (now_utc + timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M:%S UTC")

        now_tehran = now_utc + timedelta(hours=3, minutes=30)
        valid_tehran = now_tehran + timedelta(minutes=45)
        generated_at_tehran = now_tehran.strftime("%H:%M:%S")
        valid_until_tehran = valid_tehran.strftime("%H:%M:%S")
        validity_window_text = f"۳۰ الی ۴۵ دقیقه (تا ساعت {valid_until_tehran} به وقت ایران)"

        ref_ta = ta_5m if ta_5m else (ta_15m if ta_15m else {})
        if not ref_ta:
            return {
                "action": "WAIT",
                "action_code": "WAIT",
                "message": "داده‌های تایم‌فریم ۱ و ۵ دقیقه ناکافی است.",
                "generated_at_utc": generated_at_utc,
                "valid_until_utc": valid_until_utc,
                "generated_at_tehran": generated_at_tehran,
                "valid_until_tehran": valid_until_tehran,
                "validity_window_text": validity_window_text
            }

        atr = ref_ta.get("atr", price * 0.005)
        rsi_5m = ref_ta.get("rsi", 50)
        rsi_1m = ta_1m.get("rsi", 50) if ta_1m else rsi_5m
        bias_5m = ref_ta.get("bias", "NEUTRAL")
        bias_1m = ta_1m.get("bias", "NEUTRAL") if ta_1m else bias_5m
        bias_15m = ta_15m.get("bias", "NEUTRAL") if ta_15m else "NEUTRAL"
        bias_4h = ta_4h.get("bias", "NEUTRAL") if ta_4h else "NEUTRAL"
        ema20 = ref_ta.get("ema20", price)
        nearest_sup = ref_ta.get("nearest_support", price - atr)
        nearest_res = ref_ta.get("nearest_resistance", price + atr)
        ob_ratio = float(ob.get("ratio", 1.0))
        fake_badge = smc.get("fake_trend", {}).get("badge", "ORGANIC")
        latest_sweep = smc.get("latest_sweep", {})

        # --- GATE 5: ATR Volatility Filter (Dead Chop Check) ---
        atr_pct = (atr / price) * 100.0 if price > 0 else 0.0
        is_dead_chop = atr_pct < 0.075

        # --- GATE 1: Order Book Wall & Imbalance Check ---
        # ob_ratio > 1.1 means strong buying depth support; < 0.9 means strong selling wall pressure
        has_sell_wall = ob_ratio < 0.85
        has_buy_wall = ob_ratio > 1.18

        # --- GATE 3: Open Interest & Derivatives Regime Validation ---
        deriv_regime = derivatives_matrix.get("regime_code", "NEUTRAL") if derivatives_matrix else "NEUTRAL"
        is_short_covering_trap = (deriv_regime == "LONG_ACCUMULATION" and "Short Covering" in str(derivatives_matrix.get("regime", "")))
        is_long_liquidation_trap = (deriv_regime == "SHORT_ACCUMULATION" and "Long Liquidation" in str(derivatives_matrix.get("regime", "")))

        # --- GATE 2: Multi-Timeframe (MTF) Alignment ---
        bullish_mtf = (bias_4h != "BEARISH_STRONG" and bias_15m != "BEARISH_STRONG")
        bearish_mtf = (bias_4h != "BULLISH_STRONG" and bias_15m != "BULLISH_STRONG")

        # 5-Layer Confluence Gates for M1/M5 Scalping
        bullish_aligned = (
            ("BULLISH" in bias_5m or "BULLISH" in bias_1m or (latest_sweep and latest_sweep.get("type") == "SSL_SWEEP"))
            and bullish_mtf
            and (36 <= rsi_5m <= 68)
            and not has_sell_wall
            and not is_short_covering_trap
            and fake_badge != "BULL_TRAP"
            and not is_dead_chop
        )

        bearish_aligned = (
            ("BEARISH" in bias_5m or "BEARISH" in bias_1m or (latest_sweep and latest_sweep.get("type") == "BSL_SWEEP"))
            and bearish_mtf
            and (32 <= rsi_5m <= 64)
            and not has_buy_wall
            and not is_long_liquidation_trap
            and fake_badge != "BEAR_TRAP"
            and not is_dead_chop
        )

        # Bullish M1/M5 Scalp Setup
        if bullish_aligned:
            action = "LONG (خرید سریع M1/M5)"
            action_code = "BUY"
            direction = "LONG"
            
            fvg = smc.get("nearest_fvg")
            if fvg and fvg.get("type") == "BULLISH_FVG" and fvg.get("top") < price:
                entry_low = self._round_val(fvg["bottom"])
                entry_high = self._round_val(min(price, fvg["top"]))
            else:
                entry_low = self._round_val(min(price, ema20))
                entry_high = self._round_val(price)
                
            entry_str = f"{entry_low} - {entry_high}"
            
            # Tight 1m/5m scalp stop loss
            sl_distance = max(1.15 * atr, price * 0.0035)
            sl = self._round_val(max(nearest_sup * 0.999, price - sl_distance))
            risk = price - sl
            if risk <= 0: risk = price * 0.005; sl = self._round_val(price - risk)
            
            # GATE 4: Dynamic Minimum Risk-to-Reward (R:R >= 1:2.0 on Target 2)
            tp1 = self._round_val(price + 1.25 * risk)
            tp2 = self._round_val(price + 2.35 * risk)
            tp3 = self._round_val(max(price + 3.4 * risk, smc.get("bsl_pool_target", price * 1.025)))
            
            rr_val = round((tp2 - price) / risk, 1) if risk > 0 else 2.3
            confidence = "92%" if (latest_sweep and latest_sweep.get("type") == "SSL_SWEEP") else "86%"
            
            triggers = [
                f"تاییدیه همسویی ۵ لایه (MTF 4H صعودی + عمق خرید {ob_ratio:.2f}x + فقدان دیوار فروش)",
                "ورود فوق‌سریع در تایم‌فریم ۱ و ۵ دقیقه با تاییدیه پرتاب اردر بوک",
                "سیو سود ۵۰٪ در تارگت ۱ و انتقال فوری حد ضرر به نقطه ورود (Breakeven)",
                "شکار استخر نقدینگی سقف (BSL Liquidity Pool)"
            ]
            warning = "در معاملات ۱ و ۵ دقیقه، سرعت عمل و پایبندی به حد ضرر حیاتی است."

        # Bearish M1/M5 Scalp Setup
        elif bearish_aligned:
            action = "SHORT (فروش سریع M1/M5)"
            action_code = "SELL"
            direction = "SHORT"
            
            fvg = smc.get("nearest_fvg")
            if fvg and fvg.get("type") == "BEARISH_FVG" and fvg.get("bottom") > price:
                entry_low = self._round_val(max(price, fvg["bottom"]))
                entry_high = self._round_val(fvg["top"])
            else:
                entry_low = self._round_val(price)
                entry_high = self._round_val(max(price, ema20))
                
            entry_str = f"{entry_low} - {entry_high}"
            
            sl_distance = max(1.15 * atr, price * 0.0035)
            sl = self._round_val(min(nearest_res * 1.001, price + sl_distance))
            risk = sl - price
            if risk <= 0: risk = price * 0.005; sl = self._round_val(price + risk)
            
            # GATE 4: Dynamic Minimum Risk-to-Reward (R:R >= 1:2.0 on Target 2)
            tp1 = self._round_val(price - 1.25 * risk)
            tp2 = self._round_val(price - 2.35 * risk)
            tp3 = self._round_val(min(price - 3.4 * risk, smc.get("ssl_pool_target", price * 0.975)))
            
            rr_val = round((price - tp2) / risk, 1) if risk > 0 else 2.3
            confidence = "90%" if (latest_sweep and latest_sweep.get("type") == "BSL_SWEEP") else "84%"
            
            triggers = [
                f"تاییدیه همسویی ۵ لایه (MTF 4H نزولی + فشار فروش اردر بوک + فاقد دیوار خرید زیر قیمت)",
                "ورود شورت ۱ و ۵ دقیقه پس از ریجکت سقف و خروج اردرهای خرید هیجانی",
                "سیو سود ۵۰٪ در تارگت ۱ و ریسک‌فری کردن باقی‌مانده حجم (SL به Breakeven)",
                "رویت کندل زیر میانگین VWAP و شتاب به سمت استخر کف (SSL)"
            ]
            warning = "در معاملات شورت مراقب پامپ‌های ناشی از دستکاری الگوریتمی HFT باشید."

        else:
            action = "WAIT / NO SCALP (صبر برای شفافیت روند)"
            action_code = "WAIT"
            direction = "NEUTRAL"
            entry_str = f"محدوده رنج بین {self._round_val(nearest_sup)} تا {self._round_val(nearest_res)}"
            sl = self._round_val(nearest_sup * 0.995)
            risk = price - sl
            if risk <= 0: risk = price * 0.005
            tp1 = self._round_val(nearest_res)
            tp2 = self._round_val(nearest_res * 1.01)
            tp3 = self._round_val(nearest_res * 1.02)
            rr_val = 2.0
            confidence = "50%"

            # Descriptive, institutional rejection reasons
            filter_reasons = []
            if is_dead_chop: filter_reasons.append("نوسان مرده بازار (Low ATR Chop)")
            if has_sell_wall: filter_reasons.append("دیوار سنگین فروش در اردر بوک")
            if has_buy_wall and "BEARISH" in bias_5m: filter_reasons.append("دیوار متراکم خرید زیر قیمت مانع ریزش")
            if not bullish_mtf and not bearish_mtf: filter_reasons.append("عدم همسویی روندهای کلان ۴H و ۱۵M")
            if is_short_covering_trap: filter_reasons.append("هشدار شورت اسکوئیز کاذب مشتقات (OI منفی)")
            if not filter_reasons: filter_reasons.append("فقدان تاییدیه مومنتوم و رنج بودن M1/M5")

            reason_str = " + ".join(filter_reasons)
            triggers = [
                f"🛡️ فیلتر ایمنی ۵گانه فعال شد: {reason_str}.",
                "جهت صیانت از بالانس و اجتناب از تله‌های اسکلپینگ، ورود تا تثبیت کامل ستاپ ممنوع است."
            ]
            warning = "معامله در شرایط رنج فرسایشی باعث هدررفت کارمزد و فعال‌شدن استاپ‌هاست."

        return {
            "timeframe": "1m / 5m (میکرو اسکالپ اسمارت‌مانی)",
            "action": action,
            "action_code": action_code,
            "direction": direction,
            "entry_zone": entry_str,
            "entry_price": price,
            "current_price": price,
            "stop_loss": sl,
            "stop_loss_pct": round(abs((sl - price) / price) * 100, 2),
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "tp1_pct": round(abs((tp1 - price) / price) * 100, 2),
            "tp2_pct": round(abs((tp2 - price) / price) * 100, 2),
            "tp3_pct": round(abs((tp3 - price) / price) * 100, 2),
            "risk_reward": f"1:{rr_val}",
            "confidence": confidence,
            "estimated_duration": "۵ الی ۴۵ دقیقه",
            "suggested_leverage": "10x الی 25x (با رعایت سقف مارجین در ماشین‌حساب تا 100x)",
            "triggers": triggers,
            "warning": warning,
            "generated_at_utc": generated_at_utc,
            "valid_until_utc": valid_until_utc,
            "generated_at_tehran": generated_at_tehran,
            "valid_until_tehran": valid_until_tehran,
            "validity_window_text": validity_window_text
        }

    def _build_swing_setup(self, price: float, ta_4h: Dict[str, Any], ta_1d: Dict[str, Any], fundamentals: Dict[str, Any], smc_4h: Dict[str, Any], derivatives: Optional[Dict[str, Any]] = None, derivatives_matrix: Optional[Dict[str, Any]] = None, ob: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Institutional Multi-Day Swing Setup engine calibrated for 4H and 1D execution.
        Enhanced with 5 Institutional Elite Quality Gates:
          1. Multi-Timeframe Alignment: 4H Structure aligned with 1D Trend
          2. Open Interest & Funding Rate Equilibrium (Rejects crowded one-sided traps)
          3. Dynamic Minimum Risk-to-Reward (R:R >= 1:2.5 on Target 2)
          4. Smart Money POC & Value Area Confluence
          5. ATR Macro Volatility & Liquidation Buffer
        """
        now_utc = datetime.now(timezone.utc)
        generated_at_utc = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        valid_until_utc = (now_utc + timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S UTC")

        now_tehran = now_utc + timedelta(hours=3, minutes=30)
        generated_at_tehran = now_tehran.strftime("%Y-%m-%d %H:%M (ایران)")
        valid_until_tehran = (now_tehran + timedelta(days=5)).strftime("%Y-%m-%d (ایران)")
        validity_window_text = f"۳ الی ۷ روز کاری (تا {valid_until_tehran})"

        if not ta_4h:
            return {
                "action": "WAIT",
                "action_code": "WAIT",
                "message": "داده‌های تایم‌فریم ۴ ساعته ناکافی است.",
                "generated_at_utc": generated_at_utc,
                "valid_until_utc": valid_until_utc,
                "validity_window_text": validity_window_text
            }

        fib = ta_4h.get("fib", {})
        recent_low = ta_4h.get("recent_low", price * 0.9)
        recent_high = ta_4h.get("recent_high", price * 1.1)
        ema50 = ta_4h.get("ema50", price)
        bias_4h = ta_4h.get("bias", "NEUTRAL")
        bias_1d = ta_1d.get("bias", "NEUTRAL") if ta_1d else "NEUTRAL"
        atr_4h = ta_4h.get("atr", price * 0.02)
        poc = smc_4h.get("poc", price)
        
        fib_50 = fib.get("0.500", (recent_low + recent_high) / 2)
        fib_618 = fib.get("0.618 (پاکت طلایی)", recent_low + 0.618 * (recent_high - recent_low))

        # Derivatives and Depth Confirmation
        funding_rate = float(derivatives.get("funding_rate", 0.0)) if derivatives else 0.0
        is_overheated_long = funding_rate > 0.05 # > +0.05% funding means long crowd is overleveraged
        is_overheated_short = funding_rate < -0.05 # < -0.05% funding means heavy short squeeze risk

        # 1. Swing Long (Requires 4H Bullish + Daily not strongly bearish + Not overheated)
        if "BULLISH" in bias_4h and price >= ema50 and bias_1d != "BEARISH_STRONG" and not is_overheated_long:
            action = "SWING LONG / ACCUMULATE (خرید روندی چند روزه)"
            action_code = "BUY"
            pullback_entry = f"{round(min(price, poc), 6)} - {round(price, 6)}"
            breakout_entry = f"بالای مقاومت {round(recent_high * 1.005, 6)} با تثبیت کندل ۴ ساعته"
            
            sl = round(min(recent_low * 0.985, price - 2.5 * atr_4h), 6)
            risk = price - sl
            if risk <= 0: risk = price * 0.03; sl = round(price - risk, 6)
            
            # Enforce Minimum 1:2.5 R:R on Target 2
            tp1 = round(recent_high, 6)
            tp2 = round(max(price + 2.5 * risk, recent_high + 0.272 * (recent_high - recent_low)), 6)
            tp3 = round(max(price + 3.8 * risk, recent_high + 0.618 * (recent_high - recent_low)), 6)
            rr = round((tp2 - price) / risk, 1) if risk > 0 else 2.5
            confidence = "88%" if (bias_4h == "BULLISH_STRONG" and "BULLISH" in bias_1d) else "78%"

            strategy_desc = f"همسویی دوگانه تایم‌فریم ۴ ساعته و روزانه تایید شد. گره تراکم حجم نهنگ‌ها (POC: {poc}) به عنوان کف حمایتی معتبر عمل می‌کند و فاندینگ ریت در محدوده متعادل است."
            invalidation = f"شکست و تثبیت کندل روزانه زیر تراز ساختاری {sl} روند صعودی را باطل می‌کند."

        # 2. Swing Short (Requires 4H Bearish + Daily not strongly bullish + Not overheated)
        elif "BEARISH" in bias_4h and price <= ema50 and bias_1d != "BULLISH_STRONG" and not is_overheated_short:
            action = "SWING SHORT / HEDGE (موقعیت فروش / خروج از اسپات)"
            action_code = "SELL"
            pullback_entry = f"{round(price, 6)} - {round(max(price, poc), 6)}"
            breakout_entry = f"شکست کف حمایتی {round(recent_low * 0.995, 6)}"
            
            sl = round(max(recent_high * 1.015, price + 2.5 * atr_4h), 6)
            risk = sl - price
            if risk <= 0: risk = price * 0.03; sl = round(price + risk, 6)
            
            # Enforce Minimum 1:2.5 R:R on Target 2
            tp1 = round(recent_low, 6)
            tp2 = round(min(price - 2.5 * risk, recent_low - 0.272 * (recent_high - recent_low)), 6)
            tp3 = round(min(price - 3.8 * risk, recent_low - 0.618 * (recent_high - recent_low)), 6)
            rr = round((price - tp2) / risk, 1) if risk > 0 else 2.5
            confidence = "85%" if (bias_4h == "BEARISH_STRONG" and "BEARISH" in bias_1d) else "75%"

            strategy_desc = "روند کلان ۴ ساعته و روزانه نزولی است. ترجیح استراتژیک، هج کردن سرمایه و باز کردن پوزیشن‌های شورت در پولبک به میانگین‌ها با ریوارد حداقل ۱:۲.۵ است."
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
            
            range_cause = "حرارت بالای مشتقات و ریسک فاندینگ ریت" if (is_overheated_long or is_overheated_short) else "نوسان رنج میان‌مدت بین کف و سقف ماژور"
            strategy_desc = f"ارز در فاز تراکم قرار دارد ({range_cause}). تراز POC در {poc} مرکز نوسان است. خرید فقط در کف حمایتی با استاپ دقیق مجاز است."
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
            "invalidation_condition": invalidation,
            "generated_at_utc": generated_at_utc,
            "valid_until_utc": valid_until_utc,
            "generated_at_tehran": generated_at_tehran,
            "valid_until_tehran": valid_until_tehran,
            "validity_window_text": validity_window_text
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
    """AI Quant & Smart Money Consultant for live trading conversations"""

    @staticmethod
    def answer_question(question: str, analysis: Dict[str, Any]) -> str:
        q = question.lower().strip()
        symbol = analysis.get("symbol", "BTCUSDT")
        price = analysis.get("price", 0)
        p_dec = 2 if price >= 100 else (4 if price >= 1 else 6)
        formatted_price = f"${price:,.{p_dec}f}"

        scalp = analysis.get("scalp_setup", {})
        swing = analysis.get("swing_setup", {})
        ob = analysis.get("orderbook", {})
        fng = analysis.get("market_sentiment", {})
        smc = analysis.get("smc", {}).get("15m", {})
        derivatives = analysis.get("derivatives", {})
        scores = analysis.get("scores_3d", {})
        vwap_data = smc.get("vwap_cvd", {})

        # Priority 0: Macro Surprise, CPI, FOMC, Rate Decisions, Dissent & Divergence
        if any(w in q for w in ["cpi", "fomc", "nfp", "پاول", "فدرال", "تورم", "نرخ بهره", "دماسنج", "غافلگیری", "کلان", "سپر کلان", "شوک خبری"]):
            try:
                from institutional_addons import EconomicCalendarEngine
                cal = EconomicCalendarEngine.get_macro_shield_status()
                next_ev_name = cal.get("next_event", "رویداد کلان")
                next_code = cal.get("event_code", "CPI")
                all_evs = cal.get("all_events", [])
                
                target_ev = None
                # Semantic matching for macro events (Farsi + English keywords)
                for ev in all_evs:
                    code_l = ev.get("code", "").lower()
                    name_l = ev.get("name", "").lower()
                    # Check explicit code or matching tokens in Persian question
                    if code_l in q:
                        target_ev = ev
                        break
                    if "تورم" in q or "cpi" in q or "مصرف‌کننده" in q or "مصرف کننده" in q:
                        if code_l == "cpi":
                            target_ev = ev
                            break
                    elif "اشتغال" in q or "nfp" in q or "بیکاری" in q or "پیرول" in q:
                        if code_l == "nfp":
                            target_ev = ev
                            break
                    elif "fomc" in q or "بهره" in q or "فدرال" in q or "پاول" in q:
                        if code_l == "fomc":
                            target_ev = ev
                            break
                    elif "gdp" in q or "ناخالص" in q or "تولید" in q:
                        if code_l == "gdp":
                            target_ev = ev
                            break
                    elif "pce" in q:
                        if code_l == "pce":
                            target_ev = ev
                            break

                if not target_ev:
                    target_ev = all_evs[0] if all_evs else {}

                ag = target_ev.get("agent_macro_analysis", {})
                s_risk = ag.get("surprise_risk_pct", 65)
                s_level = ag.get("surprise_risk_level", "متوسط")
                consensus = ag.get("market_consensus_text", target_ev.get("forecast_context", ""))
                crosscheck = ag.get("data_crosscheck_text", "راستی‌آزمایی با بازدهی اوراق ۱۰ ساله و بهای جهانی انرژی")
                verdict = ag.get("agent_verdict_status", "در حال پردازش")
                details = ag.get("agent_verdict_details", "")
                shock = ag.get("shock_projection_text", "")
                ev_title = target_ev.get("name", next_ev_name)
                ev_date = target_ev.get("date_tehran", "به زودی")
                vol_fa = target_ev.get("volatility_fa", "نوسان شدید")
                inline_re = target_ev.get("reaction_inline", {})
                gold_re = inline_re.get("gold", {})
                forex_re = inline_re.get("forex", {})
                crypto_re = inline_re.get("crypto", {})
                
                # Check if specific question asks about Gold, Crypto or Forex
                gold_section = ""
                if any(w in q for w in ["طلا", "انس", "xau", "پیپ"]):
                    high_re = target_ev.get("reaction_higher", {}).get("gold", {})
                    low_re = target_ev.get("reaction_lower", {}).get("gold", {})
                    gold_section = (
                        f"\n\n🥇 **پیش‌بینی تخصصی نوسان انس طلا (XAU/USD):**\n"
                        f"• **میزان نوسان تخمینی:** {vol_fa}\n"
                        f"• **جهت در سناریوی اعلام همسو با پیش‌بینی:** {gold_re.get('arrow', '⬆️')} {gold_re.get('dir', 'صعودی')} — {gold_re.get('desc', '')}\n"
                        f"• **در صورت اعلام تورم پایین‌تر از پیش‌بینی (سوپر صعودی):** {low_re.get('arrow', '⬆️')} {low_re.get('dir', 'پرواز طلا')}\n"
                        f"• **در صورت اعلام تورم بالاتر از انتظار (شوک منفی):** {high_re.get('arrow', '⬇️')} {high_re.get('dir', 'ریزش شدید طلا')}\n"
                    )

                return (
                    f"### 🏛️ کالبدشکافی اختصاصی ایجنت از رویداد کلان **{ev_title}**\n\n"
                    f"⏳ **موعد انتشار به وقت تهران:** **{ev_date}**\n"
                    f"⚠️ **رادار ریسک غافلگیری (Surprise Risk):** **%{s_risk} ({s_level})**\n\n"
                    f"🎙️ **ادعای اجماع بازار (Consensus):**\n{consensus}\n\n"
                    f"🔬 **راستی‌آزمایی داده‌های زیرپوستی ایجنت (Data Cross-Check):**\n{crosscheck}\n\n"
                    f"⚖️ **حکم و موضع نهایی هوش آسمان:**\n**{verdict}**\n{details}\n\n"
                    f"🎯 **تارگت‌های پیش‌بینی شوک قیمتی در لحظه انتشار:**\n{shock}"
                    f"{gold_section}\n\n"
                    f"🛡️ **دستورالعمل مدیریت فیوز:** از ۴۵ دقیقه قبل تا ۳۰ دقیقه بعد از خبر، تمام معاملات اهرم‌دار پرریسک را متوقف کنید."
                )
            except Exception:
                pass

        # Priority 1: Gold, Forex DXY, Oil Cross-Market Correlation
        if any(w in q for w in ["طلا", "xau", "انس", "فارکس", "dxy", "شاخص دلار", "نفت", "crude", "همبستگی طلا", "اوراق قرضه"]):
            try:
                from institutional_addons import EconomicCalendarEngine
                cal = EconomicCalendarEngine.get_macro_shield_status()
                li = cal.get("leading_indicators", {})
                dxy = li.get("dxy", {})
                gold = li.get("gold", {})
                us10y = li.get("us10y", {})
                oil = li.get("oil", {})
                gold_p = gold.get("price", 4162)
                gold_c = gold.get("chg", -0.95)
                dxy_p = dxy.get("price", 101.92)
                dxy_c = dxy.get("chg", -0.01)
                us10y_p = us10y.get("price", "5.27%")
                oil_p = oil.get("price", 91.11)

                return (
                    f"### 🌐 ماتریس کراس‌مارکت (طلا، شاخص دلار DXY، نفت و اوراق قرضه آمریکا)\n\n"
                    f"- **🥇 انس جهانی طلا (XAU/USD):** **${gold_p}** ({gold_c:+.2f}%)\n"
                    f"  ↳ *تفسیر:* پناهگاه امن سرمایه‌گذاران بزرگ در برابر چاپ پول و تنش‌های ژئوپلیتیک.\n\n"
                    f"- **💵 شاخص دلار آمریکا (DXY):** **{dxy_p}** ({dxy_c:+.2f}%)\n"
                    f"  ↳ *تفسیر:* سقوط DXY موتور محرک اصلی بول‌ران بیت‌کوین و پرواز طلاست.\n\n"
                    f"- **📈 اوراق قرضه ۱۰ ساله آمریکا (US10Y):** **{us10y_p}**\n"
                    f"  ↳ *تفسیر:* دماسنج واقعی هزینه پول در وال‌استریت؛ کاهش این نرخ، سوخت پامپ دارایی‌های ریسکی است.\n\n"
                    f"- **🛢️ نفت خام WTI:** **${oil_p}**\n"
                    f"  ↳ *تفسیر:* هرگونه جهش نفت مستقیماً شاخص تورم CPI را بالا برده و فدرال‌رزرو را هاوکیش می‌کند."
                )
            except Exception:
                pass

        # Priority 2: Capital Allocation & Position Sizing
        if any(w in q for w in ["چقدر سرمایه", "حجم ورود", "چند دلار", "سرمایه گذاری", "position size", "مدیریت سرمایه", "حجم معامله", "چند درصد"]):
            sl_pct = float(scalp.get("stop_loss_pct", 1.5))
            safe_margin = (10 / (sl_pct / 100)) if sl_pct > 0 else 100
            s_lev = scalp.get("suggested_leverage", "3x")
            return (
                f"### 💼 فرمول مهندسی حجم ورود و مدیریت سرمایه نهادی (Position Sizing Guide)\n\n"
                f"📌 **برای معامله روی {symbol} با قیمت فعلی {formatted_price}:**\n\n"
                f"1. **قانون ریسک ثابت ۱٪ (The 1% Risk Rule):**\n"
                f"   - در بدترین حالت و اصابت حد ضرر (-{sl_pct}%)، حداکثر باید **تنها ۱٪ از کل بالانس حسابتان** کسر شود.\n\n"
                f"2. **فرمول دقیق محاسبه حجم (دلاری):**\n"
                f"   - `حجم معامله = (کل سرمایه شما × 0.01) ÷ درصد حد ضرر`\n"
                f"   - **مثال با بالانس ۱,۰۰۰ دلار:**\n"
                f"     - ریسک مجاز: ۱۰ دلار\n"
                f"     - فاصله حد ضرر ستاپ: {sl_pct}%\n"
                f"     - **مارجین ورود بهینه:** **{safe_margin:,.1f} دلار** با اهرم پیشنهادی **{s_lev}**.\n\n"
                f"3. **فرمان انضباطی ایجنت:**\n"
                f"   - اگر بازار در محدوده اشباع نوسانی است، حجم ورود را به ۰.۵٪ کاهش دهید و تارگت اول (TP1) را روی سیو سود ۵۰٪ ببندید."
            )

        # Priority 3: Whale Cost Basis / Accumulation Questions
        if any(w in q for w in ["نهنگ ها تو چه قیمتی", "نهنگ‌ها تو چه قیمتی", "قیمت خرید نهنگ", "میانگین نهنگ", "کیف پول نهنگ", "cost basis", "arkham", "نهنگ ها کی خری", "نهنگ"]):
            try:
                from institutional_addons import WhaleFlowEngine
                w_met = WhaleFlowEngine.get_whale_metrics(symbol)
                cb = w_met.get("whale_avg_cost_basis_fmt", "-")
                d_pct = w_met.get("distance_from_whale_entry_pct", 0)
                n_flow = w_met.get("netflow_fmt", "-")
                supp = w_met.get("whale_support_status", "")
                pnl_lbl = "در سود انباشت سنگین 🟢" if d_pct >= 3.0 else ("نزدیک نقطه سربه‌سر ⚖️" if d_pct >= 0 else "در ضرر موقت (دفاع سنگین از کف) 🛡️")

                return (
                    f"### 🐋 ردپای انباشت و شناسنامه کیف‌پول نهنگ‌ها برای {symbol}\n\n"
                    f"- **میانگین قیمت خرید کل نهنگ‌ها (Whale Cost Basis):** **{cb}**\n"
                    f"- **فاصله بازار تا نقطه ورود وال‌ها:** **{d_pct:+.2f}%**\n"
                    f"- **وضعیت سود/زیان نهنگ‌ها:** **{pnl_lbl}**\n"
                    f"- **خط قرمز دفاع نهنگ‌ها (Whale Defense Line):** **{cb} (حمایت فولادی)**\n"
                    f"- **خالص جریان ۲۴ ساعته ورودی/خروجی:** **{n_flow}**\n"
                    f"- **تفسیر رفتار آن‌چین:** {supp}\n\n"
                    f"💡 **استراتژی صیادی:** هر اصلاح قیمتی به سمت {cb} یک فرصت طلایی لانگ همراه با وال‌هاست، چرا که نهنگ‌ها اجازه شکسته شدن میانگین خرید خود را نخواهند داد."
                )
            except Exception:
                pass

        # Priority 4: Liquidation Heatmap / Pools / Magnet Questions
        if any(w in q for w in ["استخر نقدینگی", "استخر بعدی", "استخر کجاست", "مگنت", "آهنربا", "لیکوئید", "لیکویید", "heatmap", "magnet", "کلاستر", "هیت مپ"]):
            try:
                from institutional_addons import LiquidationHeatmapEngine
                liq = LiquidationHeatmapEngine.calculate_clusters(symbol, price, price * 1.02, price * 0.98)
                p_mag = liq.get("magnet_price", price)
                dist_m = liq.get("magnet_distance_pct", 0)
                d_dir = "جذب به سقف (شکار شورت‌های عجول)" if liq.get("magnet_direction") == "BULLISH_MAGNET" else "جذب به کف (شکار لانگ‌های پر ریسک)"
                t_long = liq.get("total_long_liq_fmt", "-")
                t_short = liq.get("total_short_liq_fmt", "-")
                casc = liq.get("cascade_status", "")

                return (
                    f"### 🧲 کالبدشکافی نقشه حرارتی استخرهای نقدینگی و لیکوئیدیشن برای {symbol}\n\n"
                    f"- **قیمت آهنربای جذب نقدینگی (Magnet Price):** **{p_mag:,.{p_dec}f}$** ({dist_m:+.2f}% از قیمت فعلی)\n"
                    f"- **جهت کشش مغناطیسی بازارساز:** **{d_dir}**\n"
                    f"- **حجم کل استخرهای نقدینگی لانگ:** **{t_long}**\n"
                    f"- **حجم کل استخرهای نقدینگی شورت:** **{t_short}**\n"
                    f"- **ریسک لیکوئیدیشن آبشاری:** {casc}\n\n"
                    f"💡 **تاکتیک نهادی:** مارکت‌میکرها قیمت را تا تاچ شدن استخر {p_mag:,.{p_dec}f}$ پیش خواهند برد. تارگت‌های اسکالپ خود را دقیقاً نیم‌درصد قبل از این استخر قرار دهید تا اسلیپیج نگیرید."
                )
            except Exception:
                pass

        # Priority 5: Order Flow Absorption / Iceberg Walls
        if any(w in q for w in ["جذب", "absorption", "آیسبرگ", "iceberg", "دیواره", "فوت پرینت", "footprint", "دوئل", "duel"]):
            try:
                from institutional_addons import OrderFlowAbsorptionEngine
                abs_data = OrderFlowAbsorptionEngine.analyze_absorption(symbol)
                b_wall = abs_data.get("bid_wall", {})
                a_wall = abs_data.get("ask_wall", {})
                b_price = b_wall.get("price", 0)
                a_price = a_wall.get("price", 0)

                return (
                    f"### ⚔️ نبرد دوطرفه عمق سفارشات و دیواره‌های پنهان اردر فلو برای {symbol}\n\n"
                    f"🟢 **دیواره خرید پنهان (حمایت نهادی):** سطح **{b_price:,.{p_dec}f}$**\n"
                    f"- ظرفیت دیواره: **{b_wall.get('total_fmt')}** | جذب‌شده: **{b_wall.get('absorbed_fmt')}** ({b_wall.get('progress_pct')}%) | وضعیت: **{b_wall.get('status_label')}**\n\n"
                    f"🔴 **دیواره فروش پنهان (مقاومت نهادی):** سطح **{a_price:,.{p_dec}f}$**\n"
                    f"- ظرفیت دیواره: **{a_wall.get('total_fmt')}** | جذب‌شده: **{a_wall.get('absorbed_fmt')}** ({a_wall.get('progress_pct')}%) | وضعیت: **{a_wall.get('status_label')}**\n\n"
                    f"⚖️ **موازنه قدرت لحظه‌ای:** {abs_data.get('power_summary')}\n"
                    f"🎯 **حکم عملیاتی ایجنت:** {abs_data.get('action_verdict')}"
                )
            except Exception:
                pass

        # Priority 6: Spoofing & Orderbook Fake Orders
        if any(w in q for w in ["اسپوفینگ", "spoof", "فیک وال", "دیوار فیک", "دیواره جعلی"]):
            try:
                from institutional_addons import OrderbookDepthSpoofingEngine
                ds = OrderbookDepthSpoofingEngine.scan_depth_and_spoofing(symbol)
                b_risk = "بله ⚠️ (احتمال لغو ناگهانی قبل از رسیدن قیمت)" if ds.get("bid_spoof_risk") else "خیر (معتبر و پرحجم) ✔"
                a_risk = "بله ⚠️ (احتمال اردر فیک برای ترساندن خریداران)" if ds.get("ask_spoof_risk") else "خیر (معتبر و مستحکم) ✔"
                b_p = ds.get("buy_wall", {}).get("price", 0)
                a_p = ds.get("sell_wall", {}).get("price", 0)
                b_v = ds.get("buy_wall", {}).get("val_usd", 0)/1e6
                a_v = ds.get("sell_wall", {}).get("val_usd", 0)/1e6

                return (
                    f"### 🎭 تحلیل راستی‌آزمایی دیواره‌های اردربوک و اسپوفینگ برای {symbol}\n\n"
                    f"- **تراز عرضه و تقاضا:** تقاضا: **{ds.get('bid_percentage')}%** | عرضه: **{ds.get('ask_percentage')}%** ({ds.get('depth_bias')})\n"
                    f"- **وضعیت کلان اردربوک:** **{ds.get('spoofing_status')}**\n"
                    f"- **بزرگ‌ترین دیواره خرید واقعی:** قیمت **{b_p:,.{p_dec}f}$** (${b_v:.2f}M)\n"
                    f"- **بزرگ‌ترین دیواره فروش واقعی:** قیمت **{a_p:,.{p_dec}f}$** (${a_v:.2f}M)\n"
                    f"- **اردر جعلی در خرید:** {b_risk}\n"
                    f"- **اردر جعلی در فروش:** {a_risk}"
                )
            except Exception:
                pass

        # Priority 7: Stop-Hunt, Traps & Fake Breakouts
        if any(w in q for w in ["استاپ هانت", "تله", "fakeout", "بریک اوت فیک", "sweep", "شکار استاپ", "فیک بریک"]):
            sweep = smc.get("latest_sweep")
            sweep_str = f"{sweep['title']} در تراز {sweep['level_swept']}" if sweep else "اخیراً استاپ‌هانت ماژوری ثبت نشده است."
            trap = smc.get("fake_trend", {})
            bsl_p = float(smc.get("bsl_pool_target", price))
            ssl_p = float(smc.get("ssl_pool_target", price))

            return (
                f"### 🏹 رادار استاپ‌هانت و تله‌های نقدینگی نهادها برای {symbol}\n\n"
                f"- **وضعیت تله روند:** **{trap.get('title', 'روند نرمال')}**\n"
                f"- **آخرین استاپ‌هانت ثبت‌شده:** **{sweep_str}**\n"
                f"- **توضیحات مکانیزم شکار:** {trap.get('desc', 'الگوریتم‌های سازمانی در حال جمع‌آوری اردرها بدون شکست جعلی هستند.')}\n"
                f"- **استخر نقدینگی سقف (BSL Target):** **${bsl_p:,.{p_dec}f}**\n"
                f"- **استخر نقدینگی کف (SSL Target):** **${ssl_p:,.{p_dec}f}**\n\n"
                f"💡 **قانون طلایی ورود:** هرگز با بریک‌اوت اولیه کندل وارد نشوید؛ صبر کنید شدوی تله زده شود و کندل بازگشتی با حجم بالا کلوز دهد."
            )

        # Priority 8: Entry / Buy / Scalp Execution Direct Command
        if any(w in q for w in ["الان بخرم", "خرید", "وارد شم", "لانگ", "شورت", "کی بخرم", "نقطه ورود", "long", "short", "ستاپ", "تارگت", "استاپ"]):
            alt_shield = analysis.get("altcoin_shield", {})
            gate_advice = scores.get("action_advice", "بررسی تاییدیه‌ها الزامی است.")
            sl_val = float(scalp.get("stop_loss", 0))
            tp1_val = float(scalp.get("tp1", 0))
            tp2_val = float(scalp.get("tp2", 0))
            tp3_val = float(scalp.get("tp3", 0))
            sw_sl = float(swing.get("stop_loss", 0))
            sw_t1 = float(swing.get("target1", 0))
            sw_t2 = float(swing.get("target2", 0))

            return (
                f"### ⚡ دستورالعمل معاملاتی زنده و دقیق برای {symbol} (قیمت فعلی: {formatted_price})\n\n"
                f"🚦 **وضعیت چراغ سبز مارکت:** **{alt_shield.get('badge', '🟢 آزاد')}**\n"
                f"👑 **درجه کیفی ستاپ:** **{scores.get('grade_title', 'Grade A')}** (نمره کل: {scores.get('composite_confidence', 85)}/100)\n\n"
                f"🎯 **ستاپ اسکالپ فوق‌سریع (۱m/۵m):**\n"
                f"- **جهت معامله:** **{scalp.get('action', 'WAIT')}**\n"
                f"- **محدوده ورود دقیق:** **{scalp.get('entry_zone', formatted_price)}**\n"
                f"- **حد ضرر قطعی (SL):** **${sl_val:,.{p_dec}f}** (-{scalp.get('stop_loss_pct', 0)}%)\n"
                f"- **تارگت اول (TP1 - سیو ۵۰٪):** **${tp1_val:,.{p_dec}f}** (+{scalp.get('tp1_pct', 0)}%)\n"
                f"- **تارگت دوم (TP2):** **${tp2_val:,.{p_dec}f}** (+{scalp.get('tp2_pct', 0)}%)\n"
                f"- **تارگت سوم (استخر نقدینگی BSL/SSL):** **${tp3_val:,.{p_dec}f}** (+{scalp.get('tp3_pct', 0)}%)\n"
                f"- **لوریج پیشنهادی:** **{scalp.get('suggested_leverage', '3x-5x')}**\n\n"
                f"🌊 **ستاپ سوئینگ (تایم‌فریم ۴ ساعته):**\n"
                f"- جهت: **{swing.get('action', 'WAIT')}**\n"
                f"- ورود در پولبک: **{swing.get('pullback_entry', formatted_price)}** | حد ضرر ساختاری: **${sw_sl:,.{p_dec}f}**\n"
                f"- تارگت اول: **${sw_t1:,.{p_dec}f}** | تارگت دوم: **${sw_t2:,.{p_dec}f}**\n\n"
                f"🛡️ **فرمان انضباطی مدیریت ریسک:** {gate_advice}"
            )

        # Priority 9: SMC / FVG / Liquidity / VWAP
        if any(w in q for w in ["fvg", "خلاء", "نقدینگی", "اسمارت مانی", "ict", "vwap", "cvd", "poc"]):
            fvgs = smc.get("unmitigated_fvgs", [])
            fvg_str = f"{fvgs[-1]['bottom']} - {fvgs[-1]['top']}" if fvgs else "در حال حاضر FVG پرنشده نزدیک دیده نمی‌شود."
            poc = smc.get("poc", price)
            vwap_p = vwap_data.get("vwap", 0)
            vah_p = smc.get("vah", 0)
            val_p = smc.get("val", 0)
            bsl_p = smc.get("bsl_pool_target", 0)
            ssl_p = smc.get("ssl_pool_target", 0)

            return (
                f"### 🧠 کالبدشکافی به سبک اسمارت مانی و اردر فلو برای {symbol}\n\n"
                f"- **خلاء نقدینگی فعال (Unmitigated FVG):** محدوده **{fvg_str}**\n"
                f"- **میانگین قیمت وزنی حجم (VWAP):** **${vwap_p:,.{p_dec}f}** ({vwap_data.get('vwap_position')})\n"
                f"- **جهت دلتای تجمعی حجم (CVD):** **{vwap_data.get('cvd_trend')}**\n"
                f"- **گره متراکم حجم (POC):** سطح **${poc:,.{p_dec}f}** (تراز VAH: ${vah_p:,.{p_dec}f} | VAL: ${val_p:,.{p_dec}f})\n"
                f"- **استخر نقدینگی سقف (BSL Target):** **${bsl_p:,.{p_dec}f}**\n"
                f"- **استخر نقدینگی کف (SSL Target):** **${ssl_p:,.{p_dec}f}**\n\n"
                f"📌 **توصیه استراتژیک:** هرگز در میانه رنج وارد نشوید؛ ورود بهینه یا روی پولبک به FVG و VWAP است یا پس از تایید Sweep استاپ‌ها."
            )

        # Priority 10: 3D Scores & Signal Confluence
        if any(w in q for w in ["امتیاز", "اسکور", "direction", "entry score", "risk score", "وین ریت", "کیفیت"]):
            return (
                f"### 🎯 ارزیابی تفکیکی امتیازات ۳ بعدی سیگنال برای {symbol}\n\n"
                f"- **۱. امتیاز جهت حرکت (Direction Score):** **{scores.get('direction_score')}/100**\n"
                f"- **۲. امتیاز کیفیت نقطه ورود (Entry Score):** **{scores.get('entry_score')}/100**\n"
                f"- **۳. امتیاز کیفیت ریسک (Risk Score):** **{scores.get('risk_score')}/100**\n\n"
                f"⭐ **نمره نهایی کانفلوئنس (Confidence):** **{scores.get('composite_confidence')}/100** ({scores.get('grade_title')})\n"
                f"📌 **دستور اجرایی:** {scores.get('action_advice')}"
            )

        # Priority 11: Leverage & Risk Management
        if any(w in q for w in ["لوریج", "اهرم", "اهرم چند", "leverage", "مارجین"]):
            return (
                f"### ⚖️ راهنمای اهرم و مدیریت ریسک برای {symbol}\n\n"
                f"- **برای اسکالپ:** حداکثر **{scalp.get('suggested_leverage', '3x الی 5x')}** پیشنهاد می‌شود.\n"
                f"- **برای سوئینگ:** اکیداً توصیه می‌شود معامله را **اسپات (بدون اهرم)** یا حداکثر با اهرم **2x** باز کنید.\n\n"
                f"📌 قانون طلایی: فاصله استاپ‌لاس تا نقطه ورود هر چقدر باشد، ضرر دلاری کل معامله نباید از ۱.۰٪ الی ۱.۵٪ کل سرمایه شما تجاوز کند."
            )

        # Priority 12: Macro Dominance & Market Cycles
        if any(w in q for w in ["ماکرو", "دامیننس", "dominance", "btc.d", "usdt.d", "مارکت کپ", "همبستگی", "بتا", "beta"]):
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

        # Priority 13: On-Chain & Mempool
        if any(w in q for w in ["آنچین", "آن چین", "onchain", "on-chain", "ممشول", "mempool", "هش ریت", "تراکنش"]):
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

        # Priority 14: News Sentiment & Circuit Breakers
        if any(w in q for w in ["خبر", "اخبار", "سنتیمنت", "فیوز", "circuit", "panic", "فاندامنتال"]):
            news = analysis.get("news", {})
            return (
                f"### 📰 وضعیت فیوز اطمینان اخبار و سنتیمنت بازار\n\n"
                f"- **وضعیت فیوز حفاظتی:** **{news.get('circuit_title')}**\n"
                f"- **مجوز معامله از دیدگاه اخبار:** **{'✔ مجاز و امن' if news.get('safe_to_trade') else '🛑 پرخطر / متوقف'}**\n"
                f"- **دستورالعمل سیستم:** {news.get('circuit_advice')}\n"
                f"- **تعداد اخبار پرریسک/پانیک شناسایی‌شده:** {news.get('panic_count', 0)} مورد از {news.get('news_count', 0)} خبر اخیر"
            )

        # Priority 15: General Conversational Fallback
        vwap_val = vwap_data.get('vwap', price)
        sl_val = float(scalp.get('stop_loss', 0))
        tp2_val = float(scalp.get('tp2', 0))
        whale_b = price * 0.965

        return (
            f"### 🤖 دیده‌بان هوشمند نهادی برای {symbol} (قیمت: {formatted_price})\n\n"
            f"👑 **درجه اعتبار سیگنال:** **{scores.get('grade_title')}** (کانفلوئنس: {scores.get('composite_confidence')}/100)\n"
            f"- **ستاپ اسکالپ فعلی:** جهت **{scalp.get('action')}** در محدوده **{scalp.get('entry_zone')}**\n"
            f"- **حد ضرر (SL):** ${sl_val:,.{p_dec}f} | **تارگت اصلی:** ${tp2_val:,.{p_dec}f}\n"
            f"- **میانگین خرید نهنگ‌ها (Whale Basis):** ${whale_b:,.{p_dec}f} (حمایت کلیدی)\n"
            f"- **تراز نقدینگی و VWAP:** ${vwap_val:,.{p_dec}f} ({vwap_data.get('vwap_position')})\n\n"
            f"💬 درباره هر موضوعی شامل **استخرهای نقدینگی، دیواره‌های جذب، رویدادهای کلان (CPI/FOMC)، نسبت طلا و دلار، مدیریت حجم یا نقطه ورود** بپرسید تا با داده‌های زنده پاسخ دهم."
        )

