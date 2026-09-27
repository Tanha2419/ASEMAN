# -*- coding: utf-8 -*-
"""واکشی موازی با کش — برای شتاب دادن به GoldenSix و بقیه موتورها.

مسئله ای که حل می کند
─────────────────────
GoldenSixCoreEngine شش درخواست شبکه را پشت سر هم می زد:

    OKX funding-rate      timeout 3
    OKX open-interest     timeout 3
    CoinGecko global      timeout 4
    DefiLlama stablecoins timeout 4   ← ۵۵۶ کیلوبایت
    mempool.space         timeout 4
    blockchain.info       timeout 4
                          ─────────
    بدترین حالت            ۲۲ ثانیه

اندازه گیری ۲۰۲۶-۰۹-۲۷:
    روی کامپیوتر معمولی   ۱.۰ ثانیه   ۶ از ۶ فیلتر سالم
    روی Render رایگان    ۲۸.۵ ثانیه   ۴ از ۶ فیلتر خطا

چون مرورگر بعد از ۱۵ ثانیه رها می کرد، سایت همیشه به حالت آفلاین
با داده ساختگی می افتاد.

راه حل
──────
۱. همه درخواست ها همزمان اجرا می شوند  → زمان کل = کندترین، نه جمع
۲. کش با عمر مشخص                      → داده ای که کند تغییر می کند
                                          هر بار دوباره گرفته نمی شود
۳. گزارش صادقانه                       → معلوم است کدام منبع نیامد،
                                          تا جای عدد واقعی، عدد ساختگی
                                          ننشیند
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

_UA = {"User-Agent": "Mozilla/5.0 (compatible; CryptoAgent/1.0)"}

_lock = threading.Lock()
_cache: Dict[str, Dict[str, Any]] = {}      # url -> {at, data, ok}

# چند وقت هر منبع تازه گرفته شود (ثانیه).
# منطق: چیزی که کند تغییر می کند لازم نیست هر دقیقه دوباره بیاید.
TTL: Dict[str, float] = {
    "okx.com": 60.0,                 # نرخ فاندینگ هر ۸ ساعت عوض می شود
    "coingecko.com": 300.0,          # دامیننس کند حرکت می کند
    "stablecoins.llama.fi": 1800.0,  # عرضه استیبل کوین روزانه تغییر می کند
    "mempool.space": 120.0,
    "blockchain.info": 300.0,
}
DEFAULT_TTL = 120.0


def _ttl_for(url: str) -> float:
    for host, t in TTL.items():
        if host in url:
            return t
    return DEFAULT_TTL


def get_json(url: str, timeout: float = 6.0,
             ttl: Optional[float] = None) -> Dict[str, Any]:
    """یک درخواست JSON با کش.

    خروجی همیشه دیکشنری است:
        {"ok": True,  "data": ..., "cached": bool, "age": float}
        {"ok": False, "error": "...", "stale": ... }

    اگر درخواست شکست بخورد ولی نسخه کهنه ای در کش باشد، همان
    برگردانده می شود با ok=False و stale — تا تصمیم گیرنده بداند
    داده تازه نیست.
    """
    ttl = _ttl_for(url) if ttl is None else ttl
    now = time.time()

    with _lock:
        hit = _cache.get(url)
        if hit and hit.get("ok") and (now - hit["at"] < ttl):
            return dict(ok=True, data=hit["data"], cached=True,
                        age=round(now - hit["at"], 1))

    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode("utf-8", "replace"))
        with _lock:
            _cache[url] = dict(at=now, data=payload, ok=True)
        return dict(ok=True, data=payload, cached=False, age=0.0)
    except Exception as e:
        with _lock:
            hit = _cache.get(url)
        if hit and hit.get("ok"):
            # داده کهنه بهتر از داده ساختگی است — ولی باید برچسب بخورد
            return dict(ok=False, error=str(e)[:120], stale=True,
                        data=hit["data"], age=round(now - hit["at"], 1))
        return dict(ok=False, error=str(e)[:120], stale=False, data=None)


def gather(jobs: Dict[str, Callable[[], Any]],
           max_workers: int = 8) -> Dict[str, Any]:
    """چند کار را همزمان اجرا می کند و نتیجه ها را برمی گرداند.

    هر کار که خطا بدهد، مقدارش None می شود و بقیه ادامه می دهند.
    """
    out: Dict[str, Any] = {}
    if not jobs:
        return out
    with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as ex:
        futures = {ex.submit(fn): name for name, fn in jobs.items()}
        for fut in futures:
            name = futures[fut]
            try:
                out[name] = fut.result()
            except Exception as e:
                out[name] = dict(ok=False, error=str(e)[:120], data=None)
    return out


def fetch_many(urls: Dict[str, str], timeout: float = 6.0) -> Dict[str, Dict]:
    """چند آدرس JSON را همزمان می گیرد.

        fetch_many({"okx": "https://...", "cg": "https://..."})
        → {"okx": {"ok": True, "data": {...}}, ...}
    """
    return gather({name: (lambda u=url: get_json(u, timeout))
                   for name, url in urls.items()})


def cache_stats() -> Dict[str, Any]:
    now = time.time()
    with _lock:
        return dict(entries=len(_cache),
                    items=[dict(url=u[:70], age=round(now - v["at"], 1),
                                ok=v.get("ok"))
                           for u, v in _cache.items()])
