"""
backtester.py - Backtesting Engine
Runs historical simulations of a trained model's trading signals,
computing ROI, Sharpe, drawdown, win rate, and full trade logs.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from indicators import apply_indicators, INDICATOR_REGISTRY


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    direction: str        # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    cumulative_pnl: float = 0.0


@dataclass
class BacktestResult:
    symbol: str
    model_name: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    best_trade_pct: float
    worst_trade_pct: float
    avg_holding_period: float  # bars
    exposure_pct: float        # % of time in market
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    drawdown_curve: list = field(default_factory=list)
    daily_returns: list = field(default_factory=list)
    buy_hold_return_pct: float = 0.0

    def summary_dict(self) -> dict:
        return {
            "Total Return": f"{self.total_return_pct:+.2f}%",
            "Annual Return": f"{self.annualized_return_pct:+.2f}%",
            "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
            "Sortino Ratio": f"{self.sortino_ratio:.3f}",
            "Max Drawdown": f"{self.max_drawdown_pct:.2f}%",
            "Win Rate": f"{self.win_rate:.1f}%",
            "Profit Factor": f"{self.profit_factor:.2f}",
            "Total Trades": self.total_trades,
            "Avg Trade P&L": f"${self.avg_trade_pnl:.2f}",
            "Avg Win": f"${self.avg_win:.2f}",
            "Avg Loss": f"${self.avg_loss:.2f}",
            "Best Trade": f"{self.best_trade_pct:+.2f}%",
            "Worst Trade": f"{self.worst_trade_pct:+.2f}%",
            "Avg Hold (bars)": f"{self.avg_holding_period:.1f}",
            "Exposure": f"{self.exposure_pct:.1f}%",
            "Buy & Hold": f"{self.buy_hold_return_pct:+.2f}%",
        }


def run_backtest(
    df: pd.DataFrame,
    model,                    # TradingModel instance
    indicator_configs: list[dict],
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05,
    position_size_pct: float = 95.0,  # % of capital per trade
    allow_short: bool = False,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    progress_callback=None,
) -> BacktestResult:
    """
    Walk through the DataFrame bar-by-bar:
      - Compute indicator features
      - Get model prediction (1 = buy, 0 = flat / sell)
      - Execute trades with commission + slippage
      - Track equity curve, drawdown, individual trades
    """
    # ── Prepare data with indicators ────────────────────────────────────
    data = apply_indicators(df.copy(), indicator_configs)
    feature_cols = model.feature_cols
    data.dropna(subset=feature_cols, inplace=True)

    if len(data) < 10:
        raise ValueError("Insufficient data after indicator computation")

    prices = data["Close"].values
    dates = data.index
    n = len(prices)

    # ── Generate predictions ────────────────────────────────────────────
    X = data[feature_cols].values
    X_scaled = model._scaler.transform(X) if model._scaler else X

    if model.model_type == "LSTM":
        lookback = model.hyperparams.get("lookback", 10)
        preds = np.zeros(n, dtype=int)
        for i in range(lookback, n):
            seq = X_scaled[i - lookback:i].reshape(1, lookback, -1)
            raw = model._model.predict(seq, verbose=0).flatten()[0]
            preds[i] = int(raw > 0.5)
    else:
        preds = model._model.predict(X_scaled)

    # ── Simulate trades ─────────────────────────────────────────────────
    capital = initial_capital
    position = 0        # number of shares held (negative = short)
    entry_price = 0.0
    entry_idx = 0
    trades: list[Trade] = []
    equity = np.full(n, initial_capital, dtype=float)
    in_market = np.zeros(n, dtype=bool)

    for i in range(1, n):
        price = prices[i]
        signal = preds[i]

        # ── Check stop-loss / take-profit ───────────────────────────
        if position != 0:
            if position > 0:
                ret = (price - entry_price) / entry_price
            else:
                ret = (entry_price - price) / entry_price

            triggered = False
            if stop_loss_pct is not None and ret <= -stop_loss_pct / 100:
                triggered = True
            if take_profit_pct is not None and ret >= take_profit_pct / 100:
                triggered = True

            if triggered:
                signal = 0  # force exit

        # ── Execute signal ──────────────────────────────────────────
        if signal == 1 and position == 0:
            # Enter LONG
            cost_adj = price * (1 + commission_pct / 100 + slippage_pct / 100)
            max_spend = capital * (position_size_pct / 100)
            shares = max_spend / cost_adj
            if shares > 0:
                position = shares
                entry_price = cost_adj
                entry_idx = i
                capital -= shares * cost_adj

        elif signal == -1 and position == 0 and allow_short:
            # Enter SHORT
            cost_adj = price * (1 - commission_pct / 100 - slippage_pct / 100)
            max_spend = capital * (position_size_pct / 100)
            shares = max_spend / cost_adj
            if shares > 0:
                position = -shares
                entry_price = cost_adj
                entry_idx = i
                capital += shares * cost_adj

        elif signal == 0 and position != 0:
            # Close position
            if position > 0:
                sell_price = price * (1 - commission_pct / 100 - slippage_pct / 100)
                pnl = position * (sell_price - entry_price)
                capital += position * sell_price
            else:
                buy_price = price * (1 + commission_pct / 100 + slippage_pct / 100)
                pnl = abs(position) * (entry_price - buy_price)
                capital -= abs(position) * buy_price

            pnl_pct = (pnl / (abs(position) * entry_price)) * 100

            trades.append(Trade(
                entry_date=str(dates[entry_idx]),
                exit_date=str(dates[i]),
                direction="LONG" if position > 0 else "SHORT",
                entry_price=round(entry_price, 4),
                exit_price=round(price, 4),
                shares=round(abs(position), 6),
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 2),
            ))
            position = 0

        # ── Update equity ───────────────────────────────────────────
        if position > 0:
            equity[i] = capital + position * prices[i]
            in_market[i] = True
        elif position < 0:
            equity[i] = capital - abs(position) * prices[i] + abs(position) * entry_price * 2
            in_market[i] = True
        else:
            equity[i] = capital

        if progress_callback and i % max(1, n // 20) == 0:
            progress_callback(int(i / n * 100), f"Bar {i}/{n}")

    # Force close any remaining position
    if position != 0:
        final_price = prices[-1]
        if position > 0:
            sell_price = final_price * (1 - commission_pct / 100 - slippage_pct / 100)
            pnl = position * (sell_price - entry_price)
            capital += position * sell_price
        else:
            buy_price = final_price * (1 + commission_pct / 100 + slippage_pct / 100)
            pnl = abs(position) * (entry_price - buy_price)
            capital -= abs(position) * buy_price

        pnl_pct = (pnl / (abs(position) * entry_price)) * 100
        trades.append(Trade(
            entry_date=str(dates[entry_idx]),
            exit_date=str(dates[-1]),
            direction="LONG" if position > 0 else "SHORT",
            entry_price=round(entry_price, 4),
            exit_price=round(final_price, 4),
            shares=round(abs(position), 6),
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
        ))
        equity[-1] = capital + pnl  # adjust
        position = 0

    # ── Compute metrics ─────────────────────────────────────────────────
    final_capital = equity[-1]
    total_return = (final_capital - initial_capital) / initial_capital * 100

    # Annualized (assume 252 trading days)
    n_bars = len(equity)
    ann_factor = 252 / max(n_bars, 1)
    ann_return = ((final_capital / initial_capital) ** ann_factor - 1) * 100

    # Daily returns
    eq_series = pd.Series(equity)
    daily_ret = eq_series.pct_change().dropna()
    avg_ret = daily_ret.mean()
    std_ret = daily_ret.std()

    sharpe = (avg_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
    downside = daily_ret[daily_ret < 0].std()
    sortino = (avg_ret / downside * np.sqrt(252)) if downside > 0 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    max_dd = abs(dd.min())

    # Trade stats
    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl <= 0]
    win_rate = (len(winning) / len(trades) * 100) if trades else 0
    gross_profit = sum(t.pnl for t in winning) if winning else 0
    gross_loss = abs(sum(t.pnl for t in losing)) if losing else 0.001
    profit_factor = gross_profit / gross_loss

    avg_win = np.mean([t.pnl for t in winning]) if winning else 0
    avg_loss = np.mean([t.pnl for t in losing]) if losing else 0
    avg_pnl = np.mean([t.pnl for t in trades]) if trades else 0

    pnl_pcts = [t.pnl_pct for t in trades]
    best = max(pnl_pcts) if pnl_pcts else 0
    worst = min(pnl_pcts) if pnl_pcts else 0

    # Buy & hold benchmark
    bh_return = (prices[-1] - prices[0]) / prices[0] * 100

    # Cumulative PnL on trades
    cum = 0.0
    for t in trades:
        cum += t.pnl
        t.cumulative_pnl = round(cum, 2)

    result = BacktestResult(
        symbol=model.symbol,
        model_name=model.name,
        initial_capital=initial_capital,
        final_capital=round(final_capital, 2),
        total_return_pct=round(total_return, 2),
        annualized_return_pct=round(ann_return, 2),
        sharpe_ratio=round(sharpe, 3),
        sortino_ratio=round(sortino, 3),
        max_drawdown_pct=round(max_dd, 2),
        win_rate=round(win_rate, 1),
        profit_factor=round(profit_factor, 2),
        total_trades=len(trades),
        avg_trade_pnl=round(avg_pnl, 2),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        best_trade_pct=round(best, 2),
        worst_trade_pct=round(worst, 2),
        avg_holding_period=0,  # placeholder
        exposure_pct=round(in_market.sum() / n * 100, 1),
        trades=trades,
        equity_curve=equity.tolist(),
        drawdown_curve=dd.tolist(),
        daily_returns=daily_ret.tolist(),
        buy_hold_return_pct=round(bh_return, 2),
    )

    if progress_callback:
        progress_callback(100, "Backtest complete!")

    return result
