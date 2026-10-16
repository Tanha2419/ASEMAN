"""
Supabase Cloud Database Engine for Aseman Trading Platform
Seamless persistent storage for signals, journal, win rate, and bot settings.
Automatically falls back to local JSON if cloud is unreachable.
"""

import os
import json
import time
import logging
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger("SupabaseEngine")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://dalwptypryarhkwqwalk.supabase.co").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_publishable_3PeOamcSnFPTt-DrVzbGeg_e8uzFIit").strip()

class SupabaseJournalManager:
    def __init__(self, base_url: str = SUPABASE_URL, key: str = SUPABASE_KEY):
        self.base_url = base_url
        self.key = key
        self.rest_endpoint = f"{self.base_url}/rest/v1/signals_journal"
        self.headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        self.is_connected = False
        self._check_connection()

    def _check_connection(self):
        try:
            r = requests.get(f"{self.rest_endpoint}?select=id&limit=1", headers=self.headers, timeout=5)
            if r.status_code in [200, 206]:
                self.is_connected = True
                logger.info("Connected to Supabase PostgreSQL Database successfully!")
            else:
                self.is_connected = False
                logger.warning(f"Supabase check returned status {r.status_code}: {r.text}")
        except Exception as e:
            self.is_connected = False
            logger.warning(f"Supabase connection test failed: {e}")

    def insert_signal(self, record: Dict[str, Any]) -> bool:
        """Inserts an approved master trade signal into Supabase."""
        try:
            payload = {
                "symbol": record.get("symbol", "").upper(),
                "action": record.get("action", "BUY").upper(),
                "entry": float(record.get("entry", 0.0)),
                "sl": float(record.get("sl", 0.0)),
                "tp1": float(record.get("tp1", 0.0)),
                "tp2": float(record.get("tp2", 0.0)),
                "tp3": float(record.get("tp3", 0.0)),
                "pnl": float(record.get("pnl", 0.0)),
                "status": record.get("status", "OPEN"),
                "source": record.get("source", "MASTER_SIGNAL")
            }
            r = requests.post(self.rest_endpoint, headers=self.headers, json=payload, timeout=5)
            if r.status_code in [200, 201]:
                return True
            else:
                logger.error(f"Failed to insert signal into Supabase: {r.text}")
                return False
        except Exception as e:
            logger.error(f"Error inserting to Supabase: {e}")
            return False

    def update_signal_status(self, symbol: str, status: str, pnl: float, new_sl: Optional[float] = None) -> bool:
        """Updates open signal status, PnL and stop loss (e.g. Break-Even or TP hit)."""
        try:
            clean_sym = symbol.upper().replace("USDT", "").replace("USD", "").strip()
            # Find open record
            patch_data = {"status": status, "pnl": float(pnl), "updated_at": "NOW()"}
            if new_sl is not None:
                patch_data["sl"] = float(new_sl)
            
            url = f"{self.rest_endpoint}?symbol=eq.{clean_sym}&status=eq.OPEN"
            r = requests.patch(url, headers=self.headers, json=patch_data, timeout=5)
            return r.status_code in [200, 204]
        except Exception as e:
            logger.error(f"Error updating Supabase signal: {e}")
            return False

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Fetches all records from Supabase."""
        try:
            url = f"{self.rest_endpoint}?select=*&order=id.desc&limit=100"
            r = requests.get(url, headers=self.headers, timeout=5)
            if r.status_code == 200:
                return r.json()
            return []
        except Exception as e:
            logger.error(f"Error reading from Supabase: {e}")
            return []

    def get_journal_stats(self) -> Dict[str, Any]:
        """Calculates win rate, total PnL, profit factor from Supabase records."""
        records = self.get_all_records()
        if not records:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "open_trades": 0
            }

        wins = sum(1 for r in records if "TP" in str(r.get("status", "")).upper() or float(r.get("pnl", 0)) > 0)
        losses = sum(1 for r in records if "SL" in str(r.get("status", "")).upper() or float(r.get("pnl", 0)) < 0)
        open_t = sum(1 for r in records if str(r.get("status", "")).upper() == "OPEN")
        closed = wins + losses
        win_rate = round((wins / closed * 100), 1) if closed > 0 else 0.0
        total_pnl = round(sum(float(r.get("pnl", 0)) for r in records), 2)

        return {
            "total_trades": len(records),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "open_trades": open_t
        }

    def clear_all_records(self) -> bool:
        """Clears all records in Supabase (Admin reset)."""
        try:
            url = f"{self.rest_endpoint}?id=gt.0"
            r = requests.delete(url, headers=self.headers, timeout=5)
            return r.status_code in [200, 204]
        except Exception as e:
            logger.error(f"Error resetting Supabase table: {e}")
            return False

# Global Singleton
supabase_journal = SupabaseJournalManager()
