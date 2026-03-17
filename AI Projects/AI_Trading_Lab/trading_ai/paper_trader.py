"""
paper_trader.py - Paper Trading Simulator
Runs virtual trades in real time using a trained model and live market data.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data_manager import fetch_ohlcv
from indicators import apply_indicators, INDICATOR_REGISTRY

SESSIONS_DIR = Path("paper_sessions")
SESSIONS_DIR.mkdir(exist_ok=True)


@dataclass
class Position:
    symbol: str
    direction: str         # "LONG" or "SHORT"
    entry_price: float
    shares: float
    entry_time: str
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    def update_pnl(self, current_price: float):
        if self.direction == "LONG":
            self.unrealized_pnl = round(self.shares * (current_price - self.entry_price), 2)
        else:
            self.unrealized_pnl = round(self.shares * (self.entry_price - current_price), 2)
        cost = self.shares * self.entry_price
        self.unrealized_pnl_pct = round(self.unrealized_pnl / cost * 100, 2) if cost else 0


@dataclass
class PaperTrade:
    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    entry_time: str
    exit_time: str
    signal_source: str = ""  # model name


@dataclass
class PaperSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Paper Session"
    initial_capital: float = 10_000.0
    cash: float = 10_000.0
    positions: dict = field(default_factory=dict)   # symbol -> Position
    closed_trades: list = field(default_factory=list)
    equity_history: list = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    commission_pct: float = 0.1
    models: dict = field(default_factory=dict)  # symbol -> model info

    @property
    def total_equity(self) -> float:
        pos_value = sum(
            p.shares * p.entry_price + p.unrealized_pnl
            for p in self.positions.values()
        )
        return round(self.cash + pos_value, 2)

    @property
    def total_return_pct(self) -> float:
        return round((self.total_equity - self.initial_capital) / self.initial_capital * 100, 2)

    @property
    def total_pnl(self) -> float:
        return round(self.total_equity - self.initial_capital, 2)

    def open_position(
        self,
        symbol: str,
        direction: str,
        current_price: float,
        size_pct: float = 95.0,
        signal_source: str = "",
    ) -> Optional[Position]:
        """Open a new position using a percentage of available cash."""
        if symbol in self.positions:
            return None  # already in a position

        adj_price = current_price * (1 + self.commission_pct / 100)
        max_spend = self.cash * (size_pct / 100)
        shares = max_spend / adj_price

        if shares <= 0 or max_spend < 1:
            return None

        pos = Position(
            symbol=symbol,
            direction=direction,
            entry_price=round(adj_price, 4),
            shares=round(shares, 6),
            entry_time=datetime.now().isoformat(),
        )
        self.positions[symbol] = pos
        self.cash -= shares * adj_price
        self.cash = round(self.cash, 2)
        return pos

    def close_position(self, symbol: str, current_price: float, signal_source: str = "") -> Optional[PaperTrade]:
        """Close an existing position at the given price."""
        if symbol not in self.positions:
            return None

        pos = self.positions.pop(symbol)
        adj_price = current_price * (1 - self.commission_pct / 100)

        if pos.direction == "LONG":
            pnl = pos.shares * (adj_price - pos.entry_price)
            self.cash += pos.shares * adj_price
        else:
            pnl = pos.shares * (pos.entry_price - adj_price)
            self.cash += pos.shares * pos.entry_price + pnl

        self.cash = round(self.cash, 2)
        pnl_pct = pnl / (pos.shares * pos.entry_price) * 100

        trade = PaperTrade(
            trade_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=round(adj_price, 4),
            shares=pos.shares,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            entry_time=pos.entry_time,
            exit_time=datetime.now().isoformat(),
            signal_source=signal_source,
        )
        self.closed_trades.append(trade)
        return trade

    def update_positions(self, prices: dict[str, float]):
        """Update unrealized P&L for all open positions."""
        for sym, pos in self.positions.items():
            if sym in prices:
                pos.update_pnl(prices[sym])

        self.equity_history.append({
            "time": datetime.now().isoformat(),
            "equity": self.total_equity,
            "cash": self.cash,
            "positions": len(self.positions),
        })

    def get_model_signal(
        self,
        symbol: str,
        model,
        indicator_configs: list[dict],
        period: str = "1y",
        interval: str = "1d",
    ) -> dict:
        """
        Fetch latest data, compute indicators, run model prediction.
        Returns {"signal": 1/0/-1, "confidence": float, "price": float}
        """
        try:
            df = fetch_ohlcv(symbol, period=period, interval=interval, cache_hours=1)

            if df.empty:
                return {"signal": 0, "confidence": 0.0, "price": 0.0,
                        "error": f"Fetch returned no data for {symbol}"}

            latest_price = round(float(df["Close"].iloc[-1]), 4)

            data = apply_indicators(df, indicator_configs)

            # Check which feature columns actually exist in the data
            missing_cols = [c for c in model.feature_cols if c not in data.columns]
            if missing_cols:
                return {"signal": 0, "confidence": 0.0, "price": latest_price,
                        "error": f"Missing indicator columns: {missing_cols[:5]}"}

            available_before = len(data)
            data.dropna(subset=model.feature_cols, inplace=True)

            if data.empty:
                return {"signal": 0, "confidence": 0.0, "price": latest_price,
                        "error": f"All {available_before} rows dropped after indicator NaN removal. "
                                 f"Try loading more historical data."}

            X = data[model.feature_cols].iloc[-1:].values
            X_scaled = model._scaler.transform(X) if model._scaler else X

            if model.model_type == "LSTM":
                lookback = model.hyperparams.get("lookback", 10)
                X_full = data[model.feature_cols].values
                X_full_scaled = model._scaler.transform(X_full)
                if len(X_full_scaled) < lookback:
                    return {"signal": 0, "confidence": 0.0, "price": latest_price,
                            "error": f"Need {lookback} rows for LSTM, only have {len(X_full_scaled)}"}
                seq = X_full_scaled[-lookback:].reshape(1, lookback, -1)
                raw = model._model.predict(seq, verbose=0).flatten()[0]
                signal = 1 if raw > 0.5 else 0
                confidence = float(raw if signal == 1 else 1 - raw)
            else:
                pred = model._model.predict(X_scaled)[0]
                signal = int(pred)
                proba = model.predict_proba(X)
                if proba is not None:
                    confidence = float(max(proba[0]))
                else:
                    confidence = 0.5

            return {
                "signal": signal,
                "confidence": round(confidence, 3),
                "price": latest_price,
            }

        except Exception as e:
            return {"signal": 0, "confidence": 0.0, "price": 0.0, "error": str(e)}

    def get_stats(self) -> dict:
        """Return session statistics."""
        wins = [t for t in self.closed_trades if t.pnl > 0]
        losses = [t for t in self.closed_trades if t.pnl <= 0]
        total = len(self.closed_trades)

        return {
            "Session": self.name,
            "Initial Capital": f"${self.initial_capital:,.2f}",
            "Current Equity": f"${self.total_equity:,.2f}",
            "Cash": f"${self.cash:,.2f}",
            "Total Return": f"{self.total_return_pct:+.2f}%",
            "Open Positions": len(self.positions),
            "Closed Trades": total,
            "Win Rate": f"{len(wins) / total * 100:.1f}%" if total else "N/A",
            "Avg Win": f"${np.mean([t.pnl for t in wins]):.2f}" if wins else "N/A",
            "Avg Loss": f"${np.mean([t.pnl for t in losses]):.2f}" if losses else "N/A",
            "Total P&L": f"${self.total_pnl:+,.2f}",
        }

    def save(self):
        data = {
            "session_id": self.session_id,
            "name": self.name,
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions": {
                sym: asdict(pos) for sym, pos in self.positions.items()
            },
            "closed_trades": [asdict(t) for t in self.closed_trades],
            "equity_history": self.equity_history,
            "created_at": self.created_at,
            "commission_pct": self.commission_pct,
        }
        path = SESSIONS_DIR / f"{self.session_id}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, session_id: str) -> "PaperSession":
        path = SESSIONS_DIR / f"{session_id}.json"
        with open(path) as f:
            data = json.load(f)

        session = cls(
            session_id=data["session_id"],
            name=data["name"],
            initial_capital=data["initial_capital"],
            cash=data["cash"],
            created_at=data["created_at"],
            commission_pct=data.get("commission_pct", 0.1),
        )
        for sym, pos_data in data.get("positions", {}).items():
            session.positions[sym] = Position(**pos_data)
        for td in data.get("closed_trades", []):
            session.closed_trades.append(PaperTrade(**td))
        session.equity_history = data.get("equity_history", [])
        return session

    @staticmethod
    def list_sessions() -> list[dict]:
        results = []
        for f in SESSIONS_DIR.glob("*.json"):
            with open(f) as fp:
                data = json.load(fp)
                results.append({
                    "session_id": data["session_id"],
                    "name": data["name"],
                    "created_at": data["created_at"],
                })
        return results
