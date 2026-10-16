import os
import json
import logging
import requests
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

logger = logging.getLogger("SupabaseJournal")

class SupabaseJournalManager:
    """
    Institutional Journal Engine for Supabase Cloud Database.
    Ensures persistent cloud storage across container reboots, unified signal tracking,
    and accurate win-rate calculations based exclusively on closed trades.
    """
    def __init__(self):
        self.supabase_url = os.environ.get("SUPABASE_URL", "https://qnhgfevhqewdzzjlyhda.supabase.co").rstrip('/')
        self.supabase_key = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFuaGdmZXZocWV3ZHp6amx5aGRhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQzNzIzNTMsImV4cCI6MjA1OTk0ODM1M30.4s8v_VIdqj10gC7oD_Xn72-8qR5PZ9fF8gV4jX0j-wE")
        self.headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        self.rest_endpoint = f"{self.supabase_url}/rest/v1/trade_journal"

    def insert_signal(self, record: Dict[str, Any]) -> bool:
        """Inserts or updates a trade setup into Supabase cloud."""
        try:
            signal_id = record.get("id") or record.get("signal_id")
            payload = {
                "symbol": record.get("symbol", "").upper(),
                "action": record.get("action", "BUY").upper(),
                "entry": float(record.get("entry", 0.0)),
                "sl": float(record.get("sl", 0.0)),
                "tp1": float(record.get("tp1", 0.0)),
                "tp2": float(record.get("tp2", 0.0)),
                "tp3": float(record.get("tp3", 0.0)),
                "pnl": float(record.get("pnl", 0.0)),
                "status": record.get("status", "TRACKING"),
                "source": record.get("source", "MASTER_SIGNAL")
            }
            if signal_id:
                payload["signal_id"] = str(signal_id)
            
            r = requests.post(self.rest_endpoint, headers=self.headers, json=payload, timeout=5)
            if r.status_code in [200, 201]:
                return True
            else:
                logger.error(f"Failed to insert signal into Supabase: {r.text}")
                return False
        except Exception as e:
            logger.error(f"Error inserting to Supabase: {e}")
            return False

    def update_signal_status(self, symbol: str, status: str, pnl: float, new_sl: Optional[float] = None, signal_id: Optional[str] = None) -> bool:
        """
        Updates open/tracking signal status, PnL and stop loss.
        Supports matching by unique signal_id or symbol with status in OPEN or TRACKING.
        Uses valid ISO timestamp instead of raw string NOW().
        """
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            patch_data = {"status": status, "pnl": float(pnl), "updated_at": now_iso}
            if new_sl is not None:
                patch_data["sl"] = float(new_sl)
            
            if signal_id:
                url = f"{self.rest_endpoint}?signal_id=eq.{signal_id}"
            else:
                clean_sym = symbol.upper().replace("USDT", "").replace("USD", "").strip()
                url = f"{self.rest_endpoint}?symbol=eq.{clean_sym}&status=in.(OPEN,TRACKING)&order=id.desc&limit=1"
            
            r = requests.patch(url, headers=self.headers, json=patch_data, timeout=5)
            return r.status_code in [200, 204]
        except Exception as e:
            logger.error(f"Error updating Supabase signal: {e}")
            return False

    def get_all_records(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Fetches up to limit records from Supabase."""
        try:
            url = f"{self.rest_endpoint}?select=*&order=id.desc&limit={limit}"
            r = requests.get(url, headers=self.headers, timeout=5)
            if r.status_code == 200:
                return r.json()
            return []
        except Exception as e:
            logger.error(f"Error reading from Supabase: {e}")
            return []

    def get_journal_stats(self) -> Dict[str, Any]:
        """
        Calculates win rate, total PnL and stats exclusively from CLOSED trades.
        Open/Tracking positions are separated and never counted as win or loss.
        """
        records = self.get_all_records(limit=1000)
        if not records:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "open_trades": 0
            }

        wins = 0
        losses = 0
        open_trades = 0
        total_pnl = 0.0

        for r in records:
            status = str(r.get("status", "")).upper()
            pnl = float(r.get("pnl", 0.0))

            if status in ["OPEN", "TRACKING"]:
                open_trades += 1
                continue

            # For closed trades only:
            total_pnl += pnl
            if "TP" in status or pnl > 0:
                wins += 1
            elif "SL" in status or pnl < 0:
                losses += 1
            elif "BREAKEVEN" in status:
                if pnl >= 0:
                    wins += 1
                else:
                    losses += 1

        closed_trades = wins + losses
        win_rate = round((wins / closed_trades * 100), 1) if closed_trades > 0 else 0.0

        return {
            "total_trades": len(records),
            "closed_trades": closed_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": round(total_pnl, 2),
            "open_trades": open_trades
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

supabase_journal = SupabaseJournalManager()
