"""
gui.py - Main Desktop Application
PyQt5 GUI with tabs for data management, indicators, model training,
backtesting, and paper trading simulation.
"""

import sys
import os
import traceback
from functools import partial
from datetime import datetime

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QComboBox, QPushButton, QTableWidget,
    QTableWidgetItem, QTextEdit, QProgressBar, QCheckBox, QGroupBox,
    QSpinBox, QDoubleSpinBox, QListWidget, QListWidgetItem, QSplitter,
    QMessageBox, QFileDialog, QHeaderView, QAbstractItemView, QStatusBar,
    QFrame, QScrollArea, QTreeWidget, QTreeWidgetItem, QMenu, QAction,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QIcon

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Local modules
from data_manager import (
    fetch_ohlcv, fetch_multiple, get_options_chain, get_ticker_info,
    POPULAR_STOCKS, CRYPTO_SYMBOLS, FUTURES_PROXIES, FOREX_SYMBOLS,
    FOREX_LABELS, PERIODS, INTERVALS, clear_cache,
)
from indicators import (
    INDICATOR_REGISTRY, get_indicators_by_category, apply_indicators,
    get_all_indicator_names,
)
from models import (
    train_model, TradingModel, get_model_types, list_saved_models,
    delete_model, prepare_training_data,
)
from backtester import run_backtest, BacktestResult
from paper_trader import PaperSession


# ── Colour Palette ──────────────────────────────────────────────────────────

DARK_BG     = "#000000"
PANEL_BG    = "#0a0a0a"
ACCENT      = "#2a2a2a"
HIGHLIGHT   = "#3a3a3a"
BORDER      = "#333333"
TEXT        = "#ffffff"
TEXT_DIM    = "#999999"
GREEN       = "#00e676"
RED         = "#ff1744"
CHART_BG    = "#000000"
CHART_GRID  = "#ffffff"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG};
    color: {TEXT};
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background: {DARK_BG};
}}
QTabWidget::tab-bar {{
    alignment: left;
}}
QTabBar {{
    qproperty-expanding: true;
}}
QTabBar::tab {{
    background: {PANEL_BG};
    color: {TEXT_DIM};
    padding: 12px 0px;
    border: 1px solid {BORDER};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{
    background: {ACCENT};
    color: {TEXT};
    font-weight: bold;
    border-bottom: 2px solid {GREEN};
}}
QTabBar::tab:hover {{
    background: {HIGHLIGHT};
    color: {TEXT};
}}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: {TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {TEXT};
}}
QPushButton {{
    background-color: {ACCENT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 8px 18px;
    color: {TEXT};
    font-weight: bold;
}}
QPushButton:hover {{
    background-color: {HIGHLIGHT};
    border-color: {TEXT_DIM};
}}
QPushButton:disabled {{
    background-color: #111;
    color: #444;
    border-color: #222;
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px;
    color: {TEXT};
}}
QComboBox::drop-down {{
    border: none;
}}
QComboBox QAbstractItemView {{
    background-color: {PANEL_BG};
    color: {TEXT};
    selection-background-color: {ACCENT};
    border: 1px solid {BORDER};
}}
QTableWidget {{
    background-color: {PANEL_BG};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 4px;
    color: {TEXT};
    alternate-background-color: #0f0f0f;
}}
QTableWidget::item {{
    padding: 4px;
    color: {TEXT};
}}
QTableWidget::item:selected {{
    background-color: {ACCENT};
}}
QHeaderView::section {{
    background-color: #1a1a1a;
    color: {TEXT};
    padding: 6px;
    border: 1px solid {BORDER};
    font-weight: bold;
}}
QTextEdit {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    color: {TEXT};
}}
QProgressBar {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    text-align: center;
    color: {TEXT};
    background-color: {PANEL_BG};
}}
QProgressBar::chunk {{
    background-color: {GREEN};
    border-radius: 3px;
}}
QListWidget {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    color: {TEXT};
}}
QListWidget::item:selected {{
    background-color: {ACCENT};
}}
QTreeWidget {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    color: {TEXT};
}}
QTreeWidget::item:selected {{
    background-color: {ACCENT};
}}
QTreeWidget::branch {{
    color: {TEXT_DIM};
}}
QCheckBox {{
    spacing: 8px;
    color: {TEXT};
}}
QCheckBox::indicator {{
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {PANEL_BG};
    width: 14px;
    height: 14px;
}}
QCheckBox::indicator:checked {{
    background: {GREEN};
    border-color: {GREEN};
}}
QScrollBar:vertical {{
    background: {PANEL_BG};
    width: 10px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 5px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_DIM};
}}
QScrollBar:horizontal {{
    background: {PANEL_BG};
    height: 10px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 5px;
    min-width: 20px;
}}
QSplitter::handle {{
    background: {BORDER};
}}
QStatusBar {{
    background: {PANEL_BG};
    color: {TEXT_DIM};
    border-top: 1px solid {BORDER};
}}
QLabel {{
    color: {TEXT};
}}
QMenuBar {{
    background: {PANEL_BG};
    color: {TEXT};
    border-bottom: 1px solid {BORDER};
}}
QMenuBar::item:selected {{
    background: {ACCENT};
}}
QMenu {{
    background: {PANEL_BG};
    color: {TEXT};
    border: 1px solid {BORDER};
}}
QMenu::item:selected {{
    background: {ACCENT};
}}
"""


# ── Worker Thread ───────────────────────────────────────────────────────────

class Worker(QThread):
    """Generic background worker thread."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ── Chart Widget ────────────────────────────────────────────────────────────

class ChartWidget(QWidget):
    """Embeddable matplotlib chart with toolbar."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 5), facecolor=CHART_BG)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet(f"background: {DARK_BG}; color: {TEXT}; border-bottom: 1px solid {BORDER};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def clear(self):
        self.figure.clear()
        self.canvas.draw()

    def plot_ohlcv(self, df, symbol="", indicators=None):
        """Plot candlestick-style chart with optional overlays."""
        self.figure.clear()

        if indicators:
            ax1 = self.figure.add_subplot(211, facecolor=CHART_BG)
            ax2 = self.figure.add_subplot(212, facecolor=CHART_BG, sharex=ax1)
        else:
            ax1 = self.figure.add_subplot(111, facecolor=CHART_BG)
            ax2 = None

        dates = df.index
        close = df["Close"]

        # Price line
        ax1.plot(dates, close, color="#00e5ff", linewidth=1.2, label="Close")
        ax1.fill_between(dates, close.min() * 0.98, close, alpha=0.08, color="#00e5ff")
        ax1.set_title(f"{symbol} Price", color=TEXT, fontsize=12)
        ax1.tick_params(colors=TEXT_DIM)
        ax1.set_ylabel("Price", color=TEXT_DIM)
        ax1.grid(True, color=CHART_GRID, alpha=0.12, linewidth=0.5)

        # Overlay indicators on price chart
        overlay_cols = []
        oscillator_cols = []
        if indicators:
            data = apply_indicators(df.copy(), indicators)
            base = {"Open", "High", "Low", "Close", "Volume"}
            for col in data.columns:
                if col in base:
                    continue
                # Heuristic: if values are similar scale to Close, overlay
                vals = data[col].dropna()
                if len(vals) == 0:
                    continue
                ratio = vals.mean() / close.mean() if close.mean() != 0 else 0
                if 0.1 < abs(ratio) < 10:
                    overlay_cols.append(col)
                else:
                    oscillator_cols.append(col)

            colors = plt.cm.Set2(np.linspace(0, 1, max(len(overlay_cols), 1)))
            for i, col in enumerate(overlay_cols):
                ax1.plot(dates, data[col].reindex(dates), label=col,
                         linewidth=0.9, alpha=0.8, color=colors[i % len(colors)])

            if ax2 and oscillator_cols:
                osc_colors = plt.cm.tab10(np.linspace(0, 1, len(oscillator_cols)))
                for i, col in enumerate(oscillator_cols):
                    ax2.plot(dates, data[col].reindex(dates), label=col,
                             linewidth=0.9, color=osc_colors[i])
                ax2.set_title("Oscillators / Indicators", color=TEXT, fontsize=10)
                ax2.tick_params(colors=TEXT_DIM)
                ax2.grid(True, color=CHART_GRID, alpha=0.12, linewidth=0.5)
                ax2.legend(fontsize=7, loc="upper left", facecolor=CHART_BG,
                           edgecolor=BORDER, labelcolor=TEXT)

        ax1.legend(fontsize=7, loc="upper left", facecolor=CHART_BG,
                   edgecolor=BORDER, labelcolor=TEXT)
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_equity_curve(self, result: "BacktestResult"):
        """Plot backtest equity curve and drawdown."""
        self.figure.clear()
        ax1 = self.figure.add_subplot(211, facecolor=CHART_BG)
        ax2 = self.figure.add_subplot(212, facecolor=CHART_BG, sharex=ax1)

        x = range(len(result.equity_curve))
        ax1.plot(x, result.equity_curve, color=GREEN, linewidth=1.2, label="Equity")
        ax1.axhline(y=result.initial_capital, color=TEXT_DIM, linewidth=0.8, linestyle="--", label="Initial Capital")
        ax1.fill_between(x, result.initial_capital, result.equity_curve, alpha=0.15,
                         where=[e >= result.initial_capital for e in result.equity_curve], color=GREEN)
        ax1.fill_between(x, result.initial_capital, result.equity_curve, alpha=0.15,
                         where=[e < result.initial_capital for e in result.equity_curve], color=RED)
        ax1.set_title(f"Equity Curve - {result.model_name}", color=TEXT, fontsize=12)
        ax1.set_ylabel("Portfolio Value ($)", color=TEXT_DIM)
        ax1.tick_params(colors=TEXT_DIM)
        ax1.grid(True, color=CHART_GRID, alpha=0.12, linewidth=0.5)
        ax1.legend(fontsize=8, facecolor=CHART_BG, edgecolor=BORDER, labelcolor=TEXT)

        ax2.fill_between(x, 0, result.drawdown_curve, color=RED, alpha=0.4)
        ax2.set_title("Drawdown %", color=TEXT, fontsize=10)
        ax2.set_ylabel("Drawdown %", color=TEXT_DIM)
        ax2.tick_params(colors=TEXT_DIM)
        ax2.grid(True, color=CHART_GRID, alpha=0.12, linewidth=0.5)

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_paper_equity(self, equity_history: list):
        """Plot paper trading session equity over time."""
        self.figure.clear()
        ax = self.figure.add_subplot(111, facecolor=CHART_BG)

        if not equity_history:
            ax.text(0.5, 0.5, "No equity data yet", transform=ax.transAxes,
                    ha="center", va="center", color=TEXT_DIM, fontsize=14)
            self.canvas.draw()
            return

        times = [h.get("time", "") for h in equity_history]
        equities = [h.get("equity", 0) for h in equity_history]

        ax.plot(range(len(equities)), equities, color=GREEN, linewidth=1.5)
        ax.set_title("Paper Trading Equity", color=TEXT, fontsize=12)
        ax.set_ylabel("Equity ($)", color=TEXT_DIM)
        ax.tick_params(colors=TEXT_DIM)
        ax.grid(True, color=CHART_GRID, alpha=0.12, linewidth=0.5)
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_live_market(self, df, symbol="", indicators=None, show_volume=True):
        """
        Plot a live-style candlestick chart with volume bars and indicator overlays.
        Designed for real-time paper trading view.
        """
        self.figure.clear()

        if df is None or df.empty:
            ax = self.figure.add_subplot(111, facecolor=CHART_BG)
            ax.text(0.5, 0.5, "No market data", transform=ax.transAxes,
                    ha="center", va="center", color=TEXT_DIM, fontsize=14)
            self.canvas.draw()
            return

        # Determine subplot layout
        has_oscillators = False
        overlay_cols = []
        oscillator_cols = []

        if indicators:
            data = apply_indicators(df.copy(), indicators)
            base = {"Open", "High", "Low", "Close", "Volume"}
            close_mean = df["Close"].mean()
            for col in data.columns:
                if col in base:
                    continue
                vals = data[col].dropna()
                if len(vals) == 0:
                    continue
                ratio = vals.mean() / close_mean if close_mean != 0 else 0
                if 0.1 < abs(ratio) < 10:
                    overlay_cols.append(col)
                else:
                    oscillator_cols.append(col)
            has_oscillators = len(oscillator_cols) > 0
        else:
            data = df

        # Layout: price (big) + volume (small) + oscillator (medium, optional)
        if has_oscillators and show_volume:
            gs = self.figure.add_gridspec(3, 1, height_ratios=[5, 1, 2], hspace=0.05)
            ax_price = self.figure.add_subplot(gs[0], facecolor=CHART_BG)
            ax_vol = self.figure.add_subplot(gs[1], facecolor=CHART_BG, sharex=ax_price)
            ax_osc = self.figure.add_subplot(gs[2], facecolor=CHART_BG, sharex=ax_price)
        elif show_volume:
            gs = self.figure.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
            ax_price = self.figure.add_subplot(gs[0], facecolor=CHART_BG)
            ax_vol = self.figure.add_subplot(gs[1], facecolor=CHART_BG, sharex=ax_price)
            ax_osc = None
        else:
            ax_price = self.figure.add_subplot(111, facecolor=CHART_BG)
            ax_vol = None
            ax_osc = None

        n = len(df)
        x = np.arange(n)
        opens = df["Open"].values
        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values
        volumes = df["Volume"].values

        # ── Candlesticks ────────────────────────────────────────────
        up = closes >= opens
        down = ~up

        # Candle bodies
        body_width = 0.6
        ax_price.bar(x[up], closes[up] - opens[up], body_width,
                     bottom=opens[up], color="#26a69a", edgecolor="#26a69a", linewidth=0.5)
        ax_price.bar(x[down], opens[down] - closes[down], body_width,
                     bottom=closes[down], color="#ef5350", edgecolor="#ef5350", linewidth=0.5)

        # Wicks
        ax_price.vlines(x[up], lows[up], opens[up], colors="#26a69a", linewidth=0.6)
        ax_price.vlines(x[up], closes[up], highs[up], colors="#26a69a", linewidth=0.6)
        ax_price.vlines(x[down], lows[down], closes[down], colors="#ef5350", linewidth=0.6)
        ax_price.vlines(x[down], opens[down], highs[down], colors="#ef5350", linewidth=0.6)

        # Current price line
        last_price = closes[-1]
        last_color = "#26a69a" if closes[-1] >= opens[-1] else "#ef5350"
        ax_price.axhline(y=last_price, color=last_color, linewidth=0.8,
                         linestyle="--", alpha=0.7)
        ax_price.text(n - 1, last_price, f" {last_price:.2f}",
                      color=last_color, fontsize=8, fontweight="bold",
                      va="center", ha="left",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor=last_color,
                                edgecolor="none", alpha=0.9),)
        # Fix: make text white on colored background
        ax_price.texts[-1].set_color("white")

        ax_price.set_title(f"{symbol}", color=TEXT, fontsize=12, fontweight="bold", loc="left")
        ax_price.set_ylabel("Price", color=TEXT_DIM, fontsize=9)
        ax_price.tick_params(colors=TEXT_DIM, labelsize=8)
        ax_price.grid(True, color=CHART_GRID, alpha=0.12, linewidth=0.5)
        ax_price.set_xlim(-1, n + 1)

        # X-axis date labels
        if n > 0:
            dates = df.index
            # Show ~8 labels
            step = max(1, n // 8)
            tick_positions = list(range(0, n, step))
            tick_labels = [dates[i].strftime("%m/%d %H:%M") if hasattr(dates[i], "strftime")
                          else str(dates[i])[:10] for i in tick_positions]

        # ── Indicator overlays on price ─────────────────────────────
        if overlay_cols:
            colors_ov = plt.cm.Set2(np.linspace(0, 1, max(len(overlay_cols), 1)))
            for i, col in enumerate(overlay_cols):
                vals = data[col].reindex(df.index)
                ax_price.plot(x, vals.values, label=col, linewidth=1.0,
                              alpha=0.85, color=colors_ov[i % len(colors_ov)])
            ax_price.legend(fontsize=7, loc="upper left", facecolor=CHART_BG,
                            edgecolor=BORDER, labelcolor=TEXT, framealpha=0.8)

        # ── Volume bars ─────────────────────────────────────────────
        if ax_vol is not None:
            vol_colors = np.where(up, "#26a69a", "#ef5350")
            ax_vol.bar(x, volumes, width=body_width, color=vol_colors, alpha=0.7)
            ax_vol.set_ylabel("Vol", color=TEXT_DIM, fontsize=8)
            ax_vol.tick_params(colors=TEXT_DIM, labelsize=7)
            ax_vol.grid(True, color=CHART_GRID, alpha=0.12, linewidth=0.5)
            # Format volume axis
            ax_vol.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda val, pos: f"{val/1e6:.1f}M" if val >= 1e6
                                  else f"{val/1e3:.0f}K" if val >= 1e3 else f"{val:.0f}"))

        # ── Oscillator subplot ──────────────────────────────────────
        if ax_osc is not None and oscillator_cols:
            osc_colors = plt.cm.tab10(np.linspace(0, 1, len(oscillator_cols)))
            for i, col in enumerate(oscillator_cols):
                vals = data[col].reindex(df.index)
                ax_osc.plot(x, vals.values, label=col, linewidth=0.9,
                            color=osc_colors[i])
            # Reference lines for RSI-like oscillators
            for col in oscillator_cols:
                if "RSI" in col:
                    ax_osc.axhline(70, color=RED, linewidth=0.5, linestyle="--", alpha=0.5)
                    ax_osc.axhline(30, color=GREEN, linewidth=0.5, linestyle="--", alpha=0.5)
                    break
            ax_osc.legend(fontsize=7, loc="upper left", facecolor=CHART_BG,
                          edgecolor=BORDER, labelcolor=TEXT, framealpha=0.8)
            ax_osc.tick_params(colors=TEXT_DIM, labelsize=7)
            ax_osc.grid(True, color=CHART_GRID, alpha=0.12, linewidth=0.5)

        # Date labels on bottom axis only
        bottom_ax = ax_osc if ax_osc else (ax_vol if ax_vol else ax_price)
        if n > 0:
            bottom_ax.set_xticks(tick_positions)
            bottom_ax.set_xticklabels(tick_labels, rotation=45, fontsize=7, color=TEXT_DIM)
        # Hide x labels on upper axes
        if ax_vol:
            ax_price.tick_params(labelbottom=False)
        if ax_osc and ax_vol:
            ax_vol.tick_params(labelbottom=False)

        self.figure.tight_layout()
        self.figure.subplots_adjust(hspace=0.05)
        self.canvas.draw()


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1 : DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════

class DataTab(QWidget):
    data_loaded = pyqtSignal(str, object)  # symbol, DataFrame

    def __init__(self):
        super().__init__()
        self._data_cache: dict[str, pd.DataFrame] = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ── Controls ────────────────────────────────────────────────
        ctrl = QGroupBox("Fetch Market Data")
        cg = QGridLayout(ctrl)

        cg.addWidget(QLabel("Symbol:"), 0, 0)
        self.symbol_input = QComboBox()
        self.symbol_input.setEditable(True)
        all_syms = POPULAR_STOCKS + CRYPTO_SYMBOLS + list(FUTURES_PROXIES.values()) + FOREX_SYMBOLS
        self.symbol_input.addItems(all_syms)
        cg.addWidget(self.symbol_input, 0, 1)

        cg.addWidget(QLabel("Period:"), 0, 2)
        self.period_combo = QComboBox()
        self.period_combo.addItems(PERIODS)
        self.period_combo.setCurrentText("1y")
        cg.addWidget(self.period_combo, 0, 3)

        cg.addWidget(QLabel("Interval:"), 0, 4)
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(INTERVALS)
        self.interval_combo.setCurrentText("1d")
        cg.addWidget(self.interval_combo, 0, 5)

        self.fetch_btn = QPushButton("  Fetch Data")
        self.fetch_btn.clicked.connect(self._fetch_data)
        cg.addWidget(self.fetch_btn, 0, 6)

        self.batch_btn = QPushButton("  Batch Fetch (Popular)")
        self.batch_btn.clicked.connect(self._batch_fetch)
        cg.addWidget(self.batch_btn, 0, 7)

        layout.addWidget(ctrl)

        # ── Quick-pick buttons ──────────────────────────────────────
        cats = QHBoxLayout()
        for label, syms in [("Stocks", POPULAR_STOCKS[:8]),
                            ("Crypto", CRYPTO_SYMBOLS[:5])]:
            grp = QGroupBox(label)
            gl = QHBoxLayout(grp)
            for s in syms:
                btn = QPushButton(s)
                btn.setMaximumWidth(80)
                btn.clicked.connect(partial(self._quick_pick, s))
                gl.addWidget(btn)
            cats.addWidget(grp)
        layout.addLayout(cats)

        cats2 = QHBoxLayout()
        for label, syms in [("Futures", list(FUTURES_PROXIES.values())[:5]),
                            ("Forex", list(FOREX_LABELS.items())[:6])]:
            grp = QGroupBox(label)
            gl = QHBoxLayout(grp)
            if label == "Forex":
                for display_name, sym in syms:
                    btn = QPushButton(display_name)
                    btn.setMaximumWidth(80)
                    btn.clicked.connect(partial(self._quick_pick, sym))
                    gl.addWidget(btn)
            else:
                for s in syms:
                    btn = QPushButton(s)
                    btn.setMaximumWidth(80)
                    btn.clicked.connect(partial(self._quick_pick, s))
                    gl.addWidget(btn)
            cats2.addWidget(grp)
        layout.addLayout(cats2)

        # ── Split: Chart / Table ────────────────────────────────────
        splitter = QSplitter(Qt.Vertical)

        self.chart = ChartWidget()
        splitter.addWidget(self.chart)

        # Data table
        table_frame = QWidget()
        tl = QVBoxLayout(table_frame)
        self.info_label = QLabel("No data loaded")
        self.info_label.setStyleSheet(f"color: {TEXT_DIM}; padding: 4px;")
        tl.addWidget(self.info_label)

        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.horizontalHeader().setStretchLastSection(True)
        tl.addWidget(self.data_table)
        splitter.addWidget(table_frame)
        splitter.setSizes([500, 300])

        layout.addWidget(splitter)

        # ── Progress ────────────────────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setMaximum(100)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

    def _quick_pick(self, symbol):
        self.symbol_input.setCurrentText(symbol)
        self._fetch_data()

    def _fetch_data(self):
        symbol = self.symbol_input.currentText().strip().upper()
        if not symbol:
            return
        period = self.period_combo.currentText()
        interval = self.interval_combo.currentText()

        self.fetch_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(10)

        def do_fetch():
            return fetch_ohlcv(symbol, period=period, interval=interval)

        self._worker = Worker(do_fetch)
        self._worker.finished.connect(lambda df: self._on_data(symbol, df))
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _batch_fetch(self):
        self.batch_btn.setEnabled(False)
        self.progress.setVisible(True)

        syms = POPULAR_STOCKS[:10] + CRYPTO_SYMBOLS[:5]

        def do_batch():
            results = {}
            for i, s in enumerate(syms):
                try:
                    results[s] = fetch_ohlcv(s, period="1y", interval="1d")
                except Exception:
                    pass
            return results

        self._worker = Worker(do_batch)
        self._worker.finished.connect(self._on_batch)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_data(self, symbol, df):
        self._data_cache[symbol] = df
        self.data_loaded.emit(symbol, df)
        self.progress.setValue(100)
        self.fetch_btn.setEnabled(True)

        # Update info
        self.info_label.setText(
            f"{symbol}  |  {len(df)} rows  |  "
            f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}  |  "
            f"Last Close: ${df['Close'].iloc[-1]:.2f}"
        )

        # Fill table
        self.data_table.setRowCount(min(len(df), 500))
        cols = ["Open", "High", "Low", "Close", "Volume"]
        self.data_table.setColumnCount(len(cols) + 1)
        self.data_table.setHorizontalHeaderLabels(["Date"] + cols)
        for i in range(min(len(df), 500)):
            row = df.iloc[-(i + 1)]  # most recent first
            self.data_table.setItem(i, 0, QTableWidgetItem(str(df.index[-(i+1)].date())))
            for j, c in enumerate(cols):
                val = row[c]
                text = f"{val:,.2f}" if c != "Volume" else f"{int(val):,}"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.data_table.setItem(i, j + 1, item)

        # Chart
        self.chart.plot_ohlcv(df, symbol)
        self.progress.setVisible(False)

    def _on_batch(self, results):
        self._data_cache.update(results)
        for sym, df in results.items():
            self.data_loaded.emit(sym, df)
        self.batch_btn.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.information(self, "Batch Fetch", f"Loaded {len(results)} symbols successfully.")

    def _on_error(self, msg):
        self.fetch_btn.setEnabled(True)
        self.batch_btn.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.warning(self, "Error", msg)

    def get_cached_data(self) -> dict[str, pd.DataFrame]:
        return self._data_cache


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2 : INDICATORS
# ═══════════════════════════════════════════════════════════════════════════

class IndicatorsTab(QWidget):
    def __init__(self, data_tab: DataTab):
        super().__init__()
        self.data_tab = data_tab
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)

        # ── Left: Indicator selector ────────────────────────────────
        left = QVBoxLayout()
        left.addWidget(QLabel("Select Indicators"))

        self.indicator_tree = QTreeWidget()
        self.indicator_tree.setHeaderLabels(["Indicator", "Category"])
        self.indicator_tree.setRootIsDecorated(True)
        cats = get_indicators_by_category()
        for cat, names in cats.items():
            parent = QTreeWidgetItem([cat, ""])
            parent.setFlags(parent.flags() | Qt.ItemIsUserCheckable)
            parent.setCheckState(0, Qt.Unchecked)
            for name in names:
                child = QTreeWidgetItem([name, cat])
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child.setCheckState(0, Qt.Unchecked)
                parent.addChild(child)
            self.indicator_tree.addTopLevelItem(parent)
        self.indicator_tree.expandAll()
        left.addWidget(self.indicator_tree)

        # Params area
        self.params_group = QGroupBox("Indicator Parameters")
        self.params_layout = QGridLayout(self.params_group)
        self.params_layout.addWidget(QLabel("Select an indicator to see parameters"), 0, 0)
        left.addWidget(self.params_group)

        self.indicator_tree.currentItemChanged.connect(self._show_params)

        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply to Chart")
        self.apply_btn.clicked.connect(self._apply_indicators)
        btn_row.addWidget(self.apply_btn)

        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._select_all)
        btn_row.addWidget(self.select_all_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(self.clear_btn)
        left.addLayout(btn_row)

        # ── Right: Chart ────────────────────────────────────────────
        right = QVBoxLayout()
        self.symbol_combo = QComboBox()
        self.symbol_combo.setPlaceholderText("Select a loaded symbol...")
        right.addWidget(self.symbol_combo)
        self.chart = ChartWidget()
        right.addWidget(self.chart)

        # Data with indicators table
        self.result_table = QTableWidget()
        self.result_table.setMaximumHeight(200)
        right.addWidget(self.result_table)

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setMaximumWidth(380)
        right_w = QWidget()
        right_w.setLayout(right)

        layout.addWidget(left_w)
        layout.addWidget(right_w, stretch=1)

        # Connect data loaded signal
        self.data_tab.data_loaded.connect(self._on_data_loaded)

    def _on_data_loaded(self, symbol, df):
        if self.symbol_combo.findText(symbol) < 0:
            self.symbol_combo.addItem(symbol)
        self.symbol_combo.setCurrentText(symbol)

    def _show_params(self, current, previous):
        # Clear old widgets
        while self.params_layout.count():
            child = self.params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not current or current.childCount() > 0:
            self.params_layout.addWidget(QLabel("Select a specific indicator"), 0, 0)
            return

        name = current.text(0)
        if name not in INDICATOR_REGISTRY:
            return

        info = INDICATOR_REGISTRY[name]
        self.params_layout.addWidget(QLabel(f"{name}"), 0, 0, 1, 2)
        desc = info.get("description", "")
        if desc:
            lbl = QLabel(desc)
            lbl.setStyleSheet(f"color: {TEXT_DIM}; font-style: italic;")
            self.params_layout.addWidget(lbl, 1, 0, 1, 2)

        row = 2
        for param, default in info["default_params"].items():
            self.params_layout.addWidget(QLabel(param), row, 0)
            if isinstance(default, int):
                sb = QSpinBox()
                sb.setRange(1, 500)
                sb.setValue(default)
                sb.setObjectName(param)
                self.params_layout.addWidget(sb, row, 1)
            elif isinstance(default, float):
                sb = QDoubleSpinBox()
                sb.setRange(0.001, 100)
                sb.setDecimals(3)
                sb.setValue(default)
                sb.setObjectName(param)
                self.params_layout.addWidget(sb, row, 1)
            row += 1

    def get_selected_indicators(self) -> list[dict]:
        """Return list of {'name': str, 'params': dict}."""
        selected = []
        root = self.indicator_tree.invisibleRootItem()
        for i in range(root.childCount()):
            cat_item = root.child(i)
            for j in range(cat_item.childCount()):
                child = cat_item.child(j)
                if child.checkState(0) == Qt.Checked:
                    name = child.text(0)
                    info = INDICATOR_REGISTRY.get(name, {})
                    selected.append({
                        "name": name,
                        "params": info.get("default_params", {}),
                    })
        return selected

    def _apply_indicators(self):
        symbol = self.symbol_combo.currentText()
        cached = self.data_tab.get_cached_data()
        if symbol not in cached:
            QMessageBox.warning(self, "No Data", "Load data first in the Data tab.")
            return

        df = cached[symbol]
        indicators = self.get_selected_indicators()
        if not indicators:
            self.chart.plot_ohlcv(df, symbol)
            return

        self.chart.plot_ohlcv(df, symbol, indicators)

        # Also show table preview
        data = apply_indicators(df.copy(), indicators)
        data = data.tail(20)
        cols = [c for c in data.columns if c not in ("Open", "High", "Low", "Volume")]
        self.result_table.setRowCount(len(data))
        self.result_table.setColumnCount(len(cols))
        self.result_table.setHorizontalHeaderLabels(cols)
        for i in range(len(data)):
            for j, c in enumerate(cols):
                val = data.iloc[i][c]
                text = f"{val:.4f}" if pd.notna(val) else ""
                self.result_table.setItem(i, j, QTableWidgetItem(text))

    def _select_all(self):
        root = self.indicator_tree.invisibleRootItem()
        for i in range(root.childCount()):
            cat = root.child(i)
            cat.setCheckState(0, Qt.Checked)
            for j in range(cat.childCount()):
                cat.child(j).setCheckState(0, Qt.Checked)

    def _clear_all(self):
        root = self.indicator_tree.invisibleRootItem()
        for i in range(root.childCount()):
            cat = root.child(i)
            cat.setCheckState(0, Qt.Unchecked)
            for j in range(cat.childCount()):
                cat.child(j).setCheckState(0, Qt.Unchecked)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3 : MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

class TrainingTab(QWidget):
    model_trained = pyqtSignal(object)  # TradingModel

    def __init__(self, data_tab: DataTab, indicators_tab: IndicatorsTab):
        super().__init__()
        self.data_tab = data_tab
        self.indicators_tab = indicators_tab
        self.trained_models: list[TradingModel] = []
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)

        # ── Left: Config ────────────────────────────────────────────
        left = QVBoxLayout()

        # Symbol
        grp1 = QGroupBox("Training Configuration")
        g1 = QGridLayout(grp1)
        g1.addWidget(QLabel("Symbol:"), 0, 0)
        self.symbol_combo = QComboBox()
        g1.addWidget(self.symbol_combo, 0, 1)

        g1.addWidget(QLabel("Model Type:"), 1, 0)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(get_model_types())
        g1.addWidget(self.model_type_combo, 1, 1)

        g1.addWidget(QLabel("Model Name:"), 2, 0)
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("Auto-generated if empty")
        g1.addWidget(self.model_name_input, 2, 1)

        g1.addWidget(QLabel("Target Horizon:"), 3, 0)
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(1, 30)
        self.horizon_spin.setValue(1)
        self.horizon_spin.setToolTip("Predict N bars ahead")
        g1.addWidget(self.horizon_spin, 3, 1)

        g1.addWidget(QLabel("Test Ratio:"), 4, 0)
        self.test_ratio_spin = QDoubleSpinBox()
        self.test_ratio_spin.setRange(0.1, 0.5)
        self.test_ratio_spin.setSingleStep(0.05)
        self.test_ratio_spin.setValue(0.2)
        g1.addWidget(self.test_ratio_spin, 4, 1)

        left.addWidget(grp1)

        # Hyperparams
        grp2 = QGroupBox("Hyperparameters")
        g2 = QGridLayout(grp2)
        g2.addWidget(QLabel("n_estimators:"), 0, 0)
        self.n_estimators = QSpinBox()
        self.n_estimators.setRange(10, 1000)
        self.n_estimators.setValue(200)
        g2.addWidget(self.n_estimators, 0, 1)

        g2.addWidget(QLabel("max_depth:"), 1, 0)
        self.max_depth = QSpinBox()
        self.max_depth.setRange(1, 50)
        self.max_depth.setValue(10)
        g2.addWidget(self.max_depth, 1, 1)

        g2.addWidget(QLabel("learning_rate:"), 2, 0)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.001, 1.0)
        self.learning_rate.setSingleStep(0.01)
        self.learning_rate.setValue(0.1)
        g2.addWidget(self.learning_rate, 2, 1)

        left.addWidget(grp2)

        # Indicators from indicators tab
        grp3 = QGroupBox("Features (from Indicators tab)")
        g3 = QVBoxLayout(grp3)
        self.feature_info = QLabel("Selected indicators will be used as features.\nGo to Indicators tab to select them.")
        self.feature_info.setStyleSheet(f"color: {TEXT_DIM};")
        g3.addWidget(self.feature_info)
        left.addWidget(grp3)

        # Train button
        self.train_btn = QPushButton("  Train Model")
        self.train_btn.setStyleSheet(f"background: {GREEN}; color: #000000; font-size: 15px; padding: 12px;")
        self.train_btn.clicked.connect(self._train)
        left.addWidget(self.train_btn)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        left.addWidget(self.progress)

        self.status_label = QLabel("")
        left.addWidget(self.status_label)

        # ── Right: Results ──────────────────────────────────────────
        right = QVBoxLayout()

        right.addWidget(QLabel("Trained Models"))
        self.models_table = QTableWidget()
        self.models_table.setColumnCount(7)
        self.models_table.setHorizontalHeaderLabels([
            "Name", "Type", "Symbol", "Accuracy", "F1", "Sharpe-Proxy", "Created"
        ])
        self.models_table.horizontalHeader().setStretchLastSection(True)
        self.models_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        right.addWidget(self.models_table)

        btn_row = QHBoxLayout()
        self.save_btn = QPushButton("Save Model")
        self.save_btn.clicked.connect(self._save_selected)
        btn_row.addWidget(self.save_btn)
        self.delete_btn = QPushButton("Delete Model")
        self.delete_btn.clicked.connect(self._delete_selected)
        btn_row.addWidget(self.delete_btn)
        self.load_btn = QPushButton("Load Saved Models")
        self.load_btn.clicked.connect(self._load_models)
        btn_row.addWidget(self.load_btn)
        right.addLayout(btn_row)

        # Classification report
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setMaximumHeight(200)
        self.report_text.setFont(QFont("Consolas", 10))
        right.addWidget(self.report_text)

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setMaximumWidth(350)
        right_w = QWidget()
        right_w.setLayout(right)

        layout.addWidget(left_w)
        layout.addWidget(right_w, stretch=1)

        self.data_tab.data_loaded.connect(self._on_data_loaded)

    def _on_data_loaded(self, symbol, df):
        if self.symbol_combo.findText(symbol) < 0:
            self.symbol_combo.addItem(symbol)

    def _train(self):
        symbol = self.symbol_combo.currentText()
        cached = self.data_tab.get_cached_data()
        if symbol not in cached:
            QMessageBox.warning(self, "No Data", "Load data first in the Data tab.")
            return

        indicators = self.indicators_tab.get_selected_indicators()
        if not indicators:
            QMessageBox.warning(self, "No Indicators", "Select indicators in the Indicators tab first.")
            return

        df = cached[symbol].copy()
        data = apply_indicators(df, indicators)
        base_cols = {"Open", "High", "Low", "Close", "Volume"}
        feature_cols = [c for c in data.columns if c not in base_cols]
        data.dropna(subset=feature_cols, inplace=True)

        if len(data) < 60:
            QMessageBox.warning(self, "Insufficient Data",
                                f"Only {len(data)} rows after computing indicators. Need ≥ 60.")
            return

        model_type = self.model_type_combo.currentText()
        name = self.model_name_input.text().strip()

        hyperparams = {
            "n_estimators": self.n_estimators.value(),
            "max_depth": self.max_depth.value(),
            "learning_rate": self.learning_rate.value(),
        }

        self.train_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("Training...")

        def progress_cb(pct, msg):
            pass  # handled via signal

        def do_train():
            return train_model(
                data, feature_cols,
                model_type=model_type,
                name=name,
                symbol=symbol,
                indicators=[i["name"] for i in indicators],
                hyperparams=hyperparams,
                target_horizon=self.horizon_spin.value(),
                test_ratio=self.test_ratio_spin.value(),
            )

        self._worker = Worker(do_train)
        self._worker.finished.connect(self._on_trained)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_trained(self, model: TradingModel):
        self.trained_models.append(model)
        self.model_trained.emit(model)
        self.train_btn.setEnabled(True)
        self.progress.setValue(100)
        self.status_label.setText(
            f"Model '{model.name}' trained — Accuracy: {model.metrics.accuracy:.1%}, "
            f"F1: {model.metrics.f1:.3f}"
        )
        self.status_label.setStyleSheet(f"color: {GREEN};")

        # Update table
        row = self.models_table.rowCount()
        self.models_table.insertRow(row)
        items = [
            model.name, model.model_type, model.symbol,
            f"{model.metrics.accuracy:.1%}",
            f"{model.metrics.f1:.3f}",
            "N/A",
            model.created_at[:19],
        ]
        for j, text in enumerate(items):
            item = QTableWidgetItem(text)
            if j == 3:
                color = GREEN if model.metrics.accuracy > 0.55 else RED
                item.setForeground(QColor(color))
            self.models_table.setItem(row, j, item)

        # Show report
        self.report_text.setText(
            f"Model: {model.name}\n"
            f"Type: {model.model_type}\n"
            f"Symbol: {model.symbol}\n"
            f"Features: {len(model.feature_cols)}\n"
            f"Indicators: {', '.join(model.indicators)}\n\n"
            f"Classification Report:\n{model.metrics.classification_report}\n\n"
            f"Confusion Matrix:\n{np.array(model.metrics.confusion)}"
        )

    def _on_error(self, msg):
        self.train_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("Training failed!")
        self.status_label.setStyleSheet(f"color: {RED};")
        QMessageBox.critical(self, "Training Error", msg)

    def _save_selected(self):
        row = self.models_table.currentRow()
        if row < 0 or row >= len(self.trained_models):
            QMessageBox.warning(self, "Select Model", "Select a model to save.")
            return
        model = self.trained_models[row]
        model.save()
        QMessageBox.information(self, "Saved", f"Model '{model.name}' saved to disk.")

    def _delete_selected(self):
        row = self.models_table.currentRow()
        if row < 0 or row >= len(self.trained_models):
            return
        model = self.trained_models.pop(row)
        self.models_table.removeRow(row)
        try:
            delete_model(model.model_id)
        except Exception:
            pass

    def _load_models(self):
        saved = list_saved_models()
        count = 0
        for meta in saved:
            try:
                model = TradingModel.load(meta["model_id"])
                self.trained_models.append(model)
                self.model_trained.emit(model)
                row = self.models_table.rowCount()
                self.models_table.insertRow(row)
                items = [
                    model.name, model.model_type, model.symbol,
                    f"{model.metrics.accuracy:.1%}" if model.metrics else "N/A",
                    f"{model.metrics.f1:.3f}" if model.metrics else "N/A",
                    "N/A", model.created_at[:19],
                ]
                for j, text in enumerate(items):
                    self.models_table.setItem(row, j, QTableWidgetItem(text))
                count += 1
            except Exception:
                pass
        QMessageBox.information(self, "Load", f"Loaded {count} saved models.")


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 4 : BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════

class BacktestTab(QWidget):
    def __init__(self, data_tab: DataTab, indicators_tab: IndicatorsTab, training_tab: TrainingTab):
        super().__init__()
        self.data_tab = data_tab
        self.indicators_tab = indicators_tab
        self.training_tab = training_tab
        self.results: list[BacktestResult] = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ── Config row ──────────────────────────────────────────────
        config = QGroupBox("Backtest Configuration")
        cg = QGridLayout(config)

        cg.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        cg.addWidget(self.model_combo, 0, 1)

        cg.addWidget(QLabel("Initial Capital ($):"), 0, 2)
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(100, 10_000_000)
        self.capital_spin.setValue(10_000)
        self.capital_spin.setPrefix("$")
        cg.addWidget(self.capital_spin, 0, 3)

        cg.addWidget(QLabel("Commission (%):"), 1, 0)
        self.commission_spin = QDoubleSpinBox()
        self.commission_spin.setRange(0, 5)
        self.commission_spin.setValue(0.1)
        self.commission_spin.setSuffix("%")
        cg.addWidget(self.commission_spin, 1, 1)

        cg.addWidget(QLabel("Position Size (%):"), 1, 2)
        self.position_spin = QDoubleSpinBox()
        self.position_spin.setRange(1, 100)
        self.position_spin.setValue(95)
        self.position_spin.setSuffix("%")
        cg.addWidget(self.position_spin, 1, 3)

        cg.addWidget(QLabel("Stop Loss (%):"), 2, 0)
        self.stoploss_spin = QDoubleSpinBox()
        self.stoploss_spin.setRange(0, 50)
        self.stoploss_spin.setValue(0)
        self.stoploss_spin.setSpecialValueText("None")
        cg.addWidget(self.stoploss_spin, 2, 1)

        cg.addWidget(QLabel("Take Profit (%):"), 2, 2)
        self.takeprofit_spin = QDoubleSpinBox()
        self.takeprofit_spin.setRange(0, 100)
        self.takeprofit_spin.setValue(0)
        self.takeprofit_spin.setSpecialValueText("None")
        cg.addWidget(self.takeprofit_spin, 2, 3)

        self.short_check = QCheckBox("Allow Short Selling")
        cg.addWidget(self.short_check, 3, 0)

        self.run_btn = QPushButton("  Run Backtest")
        self.run_btn.setStyleSheet(f"background: {RED}; color: #ffffff; font-size: 15px; padding: 12px;")
        self.run_btn.clicked.connect(self._run_backtest)
        cg.addWidget(self.run_btn, 3, 2, 1, 2)

        layout.addWidget(config)

        # ── Progress ────────────────────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # ── Results split ───────────────────────────────────────────
        splitter = QSplitter(Qt.Vertical)

        self.chart = ChartWidget()
        splitter.addWidget(self.chart)

        bottom = QWidget()
        bl = QHBoxLayout(bottom)

        # Metrics table
        metrics_grp = QGroupBox("Performance Metrics")
        ml = QVBoxLayout(metrics_grp)
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        ml.addWidget(self.metrics_table)
        bl.addWidget(metrics_grp)

        # Trade log
        trades_grp = QGroupBox("Trade Log")
        tl = QVBoxLayout(trades_grp)
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(8)
        self.trades_table.setHorizontalHeaderLabels([
            "Entry", "Exit", "Dir", "Entry $", "Exit $", "Shares", "P&L", "P&L %"
        ])
        self.trades_table.horizontalHeader().setStretchLastSection(True)
        tl.addWidget(self.trades_table)
        bl.addWidget(trades_grp)

        splitter.addWidget(bottom)
        splitter.setSizes([500, 300])
        layout.addWidget(splitter)

        # ── Compare button ──────────────────────────────────────────
        self.compare_btn = QPushButton("Compare All Results")
        self.compare_btn.clicked.connect(self._compare)
        layout.addWidget(self.compare_btn)

        self.training_tab.model_trained.connect(self._on_model_added)

    def _on_model_added(self, model):
        self.model_combo.addItem(f"{model.name} ({model.model_type})")
        self.model_combo.setItemData(self.model_combo.count() - 1, model)

    def _run_backtest(self):
        idx = self.model_combo.currentIndex()
        if idx < 0:
            QMessageBox.warning(self, "No Model", "Train a model first.")
            return

        model = self.model_combo.itemData(idx)
        if model is None:
            QMessageBox.warning(self, "No Model", "Train a model first.")
            return
        cached = self.data_tab.get_cached_data()
        if model.symbol not in cached:
            QMessageBox.warning(self, "No Data", f"Load data for {model.symbol} first.")
            return

        df = cached[model.symbol]
        indicator_configs = self.indicators_tab.get_selected_indicators()
        if not indicator_configs:
            # Reconstruct from model's indicator list
            indicator_configs = [{"name": n, "params": {}} for n in model.indicators]

        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)

        sl = self.stoploss_spin.value() or None
        tp = self.takeprofit_spin.value() or None

        def do_backtest():
            return run_backtest(
                df, model, indicator_configs,
                initial_capital=self.capital_spin.value(),
                commission_pct=self.commission_spin.value(),
                position_size_pct=self.position_spin.value(),
                allow_short=self.short_check.isChecked(),
                stop_loss_pct=sl,
                take_profit_pct=tp,
            )

        self._worker = Worker(do_backtest)
        self._worker.finished.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_result(self, result: BacktestResult):
        self.results.append(result)
        self.run_btn.setEnabled(True)
        self.progress.setValue(100)

        # Chart
        self.chart.plot_equity_curve(result)

        # Metrics
        summary = result.summary_dict()
        self.metrics_table.setRowCount(len(summary))
        for i, (k, v) in enumerate(summary.items()):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(k))
            item = QTableWidgetItem(str(v))
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            # Color-code
            try:
                val_str = str(v).replace("%", "").replace("$", "").replace(",", "").replace("+", "")
                num = float(val_str)
                if "Return" in k or "P&L" in k:
                    item.setForeground(QColor(GREEN if num > 0 else RED))
            except ValueError:
                pass
            self.metrics_table.setItem(i, 1, item)

        # Trade log
        self.trades_table.setRowCount(len(result.trades))
        for i, t in enumerate(result.trades):
            items = [
                t.entry_date[:10], t.exit_date[:10], t.direction,
                f"${t.entry_price:.2f}", f"${t.exit_price:.2f}",
                f"{t.shares:.4f}", f"${t.pnl:+.2f}", f"{t.pnl_pct:+.2f}%",
            ]
            for j, text in enumerate(items):
                item = QTableWidgetItem(text)
                if j >= 6:
                    item.setForeground(QColor(GREEN if t.pnl > 0 else RED))
                self.trades_table.setItem(i, j, item)

        self.progress.setVisible(False)

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Backtest Error", msg)

    def _compare(self):
        if len(self.results) < 2:
            QMessageBox.information(self, "Compare", "Run at least 2 backtests to compare.")
            return

        text = f"{'Model':<25} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>7}\n"
        text += "-" * 70 + "\n"
        for r in self.results:
            text += (
                f"{r.model_name:<25} "
                f"{r.total_return_pct:>+9.2f}% "
                f"{r.sharpe_ratio:>8.3f} "
                f"{r.max_drawdown_pct:>7.2f}% "
                f"{r.win_rate:>7.1f}% "
                f"{r.total_trades:>7}\n"
            )

        msg = QMessageBox(self)
        msg.setWindowTitle("Backtest Comparison")
        msg.setText("Model Comparison Results")
        msg.setDetailedText(text)
        msg.setFont(QFont("Consolas", 10))
        msg.exec_()


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 5 : PAPER TRADING
# ═══════════════════════════════════════════════════════════════════════════

class PaperTradingTab(QWidget):
    def __init__(self, data_tab: DataTab, indicators_tab: IndicatorsTab, training_tab: TrainingTab):
        super().__init__()
        self.data_tab = data_tab
        self.indicators_tab = indicators_tab
        self.training_tab = training_tab
        self.session: PaperSession = None
        self._last_df: pd.DataFrame = None       # cached chart data
        self._signal_busy = False                 # prevent overlapping signal calls
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        # ── Top row: Session controls (compact) ────────────────────
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Session:"))
        self.session_name = QLineEdit("My Paper Session")
        self.session_name.setMaximumWidth(180)
        top_row.addWidget(self.session_name)

        top_row.addWidget(QLabel("Capital:"))
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(100, 10_000_000)
        self.capital_spin.setValue(10_000)
        self.capital_spin.setPrefix("$")
        self.capital_spin.setMaximumWidth(140)
        top_row.addWidget(self.capital_spin)

        self.new_session_btn = QPushButton("New")
        self.new_session_btn.clicked.connect(self._new_session)
        top_row.addWidget(self.new_session_btn)
        self.save_session_btn = QPushButton("Save")
        self.save_session_btn.clicked.connect(self._save_session)
        top_row.addWidget(self.save_session_btn)
        self.load_session_btn = QPushButton("Load")
        self.load_session_btn.clicked.connect(self._load_session)
        top_row.addWidget(self.load_session_btn)

        top_row.addStretch()

        # Portfolio summary inline
        self.equity_label = QLabel("Equity: $0.00")
        self.equity_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        top_row.addWidget(self.equity_label)
        self.cash_label = QLabel("Cash: $0.00")
        top_row.addWidget(self.cash_label)
        self.pnl_label = QLabel("P&L: $0.00")
        top_row.addWidget(self.pnl_label)
        self.return_label = QLabel("Ret: 0.00%")
        top_row.addWidget(self.return_label)

        layout.addLayout(top_row)

        # ══════════════════════════════════════════════════════════════
        #  PRICE TICKER BANNER
        # ══════════════════════════════════════════════════════════════
        ticker_frame = QFrame()
        ticker_frame.setStyleSheet(f"""
            QFrame {{
                background: {PANEL_BG};
                border: 1px solid {BORDER};
                border-radius: 4px;
                padding: 4px 12px;
            }}
        """)
        ticker_layout = QHBoxLayout(ticker_frame)
        ticker_layout.setContentsMargins(8, 2, 8, 2)

        self.ticker_symbol_label = QLabel("---")
        self.ticker_symbol_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        ticker_layout.addWidget(self.ticker_symbol_label)

        self.ticker_price_label = QLabel("0.00")
        self.ticker_price_label.setFont(QFont("Segoe UI", 22, QFont.Bold))
        ticker_layout.addWidget(self.ticker_price_label)

        self.ticker_change_label = QLabel("+0.00  +0.00%")
        self.ticker_change_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        ticker_layout.addWidget(self.ticker_change_label)

        ticker_layout.addStretch()

        self.ticker_ohlc_label = QLabel("O: --  H: --  L: --  V: --")
        self.ticker_ohlc_label.setFont(QFont("Consolas", 10))
        self.ticker_ohlc_label.setStyleSheet(f"color: {TEXT_DIM};")
        ticker_layout.addWidget(self.ticker_ohlc_label)

        ticker_layout.addStretch()

        self.ticker_time_label = QLabel("")
        self.ticker_time_label.setStyleSheet(f"color: {TEXT_DIM};")
        ticker_layout.addWidget(self.ticker_time_label)

        # Signal display
        self.signal_label = QLabel("Signal: --")
        self.signal_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.signal_label.setMinimumWidth(280)
        ticker_layout.addWidget(self.signal_label)

        layout.addWidget(ticker_frame)

        # ══════════════════════════════════════════════════════════════
        #  MAIN AREA: Chart (left) + Controls/Tables (right)
        # ══════════════════════════════════════════════════════════════
        main_splitter = QSplitter(Qt.Horizontal)

        # ── LEFT: Live Market Chart ─────────────────────────────────
        chart_panel = QWidget()
        chart_layout = QVBoxLayout(chart_panel)
        chart_layout.setContentsMargins(0, 0, 0, 0)

        # Chart interval selector
        chart_ctrl = QHBoxLayout()
        chart_ctrl.addWidget(QLabel("Symbol:"))
        self.trade_symbol = QComboBox()
        self.trade_symbol.setEditable(True)
        self.trade_symbol.setMinimumWidth(120)
        self.trade_symbol.currentTextChanged.connect(self._on_symbol_changed)
        chart_ctrl.addWidget(self.trade_symbol)

        chart_ctrl.addWidget(QLabel("Interval:"))
        self.chart_interval = QComboBox()
        self.chart_interval.addItems(["1m", "2m", "5m", "15m", "30m", "1h", "1d"])
        self.chart_interval.setCurrentText("5m")
        self.chart_interval.currentTextChanged.connect(self._refresh_chart)
        chart_ctrl.addWidget(self.chart_interval)

        chart_ctrl.addWidget(QLabel("Period:"))
        self.chart_period = QComboBox()
        self.chart_period.addItems(["1d", "5d", "1mo", "3mo", "6mo", "1y"])
        self.chart_period.setCurrentText("1d")
        self.chart_period.currentTextChanged.connect(self._refresh_chart)
        chart_ctrl.addWidget(self.chart_period)

        chart_ctrl.addStretch()
        chart_layout.addLayout(chart_ctrl)

        # The big chart
        self.market_chart = ChartWidget()
        chart_layout.addWidget(self.market_chart)

        main_splitter.addWidget(chart_panel)

        # ── RIGHT: Controls + Positions + History ───────────────────
        right_panel = QWidget()
        right_panel.setMaximumWidth(500)
        rl = QVBoxLayout(right_panel)
        rl.setSpacing(4)

        # Model + Trade controls
        trade_grp = QGroupBox("Trade Controls")
        tg = QGridLayout(trade_grp)
        tg.setSpacing(4)

        tg.addWidget(QLabel("Model:"), 0, 0)
        self.trade_model_combo = QComboBox()
        tg.addWidget(self.trade_model_combo, 0, 1, 1, 3)

        self.get_signal_btn = QPushButton("Get AI Signal")
        self.get_signal_btn.clicked.connect(self._get_signal)
        tg.addWidget(self.get_signal_btn, 1, 0, 1, 2)

        self.auto_signal_check = QCheckBox("Auto Signal")
        self.auto_signal_check.setChecked(True)
        tg.addWidget(self.auto_signal_check, 1, 2)

        self.auto_trade_check = QCheckBox("Auto Trade")
        tg.addWidget(self.auto_trade_check, 1, 3)

        self.buy_btn = QPushButton("BUY")
        self.buy_btn.setStyleSheet(f"background: {GREEN}; color: #000000; font-size: 14px; padding: 8px;")
        self.buy_btn.clicked.connect(self._buy)
        tg.addWidget(self.buy_btn, 2, 0, 1, 2)

        self.sell_btn = QPushButton("SELL / CLOSE")
        self.sell_btn.setStyleSheet(f"background: {RED}; color: #ffffff; font-size: 14px; padding: 8px;")
        self.sell_btn.clicked.connect(self._sell)
        tg.addWidget(self.sell_btn, 2, 2, 1, 2)

        # Refresh controls
        tg.addWidget(QLabel("Refresh:"), 3, 0)
        self.refresh_interval_spin = QSpinBox()
        self.refresh_interval_spin.setRange(5, 3600)
        self.refresh_interval_spin.setValue(5)
        self.refresh_interval_spin.setSuffix(" sec")
        self.refresh_interval_spin.valueChanged.connect(self._update_timer_interval)
        tg.addWidget(self.refresh_interval_spin, 3, 1)

        self.auto_chart_check = QCheckBox("Auto Chart")
        self.auto_chart_check.setChecked(True)
        tg.addWidget(self.auto_chart_check, 3, 2)

        self.refresh_btn = QPushButton("Refresh Now")
        self.refresh_btn.clicked.connect(self._manual_refresh_all)
        tg.addWidget(self.refresh_btn, 3, 3)

        rl.addWidget(trade_grp)

        # Open positions
        rl.addWidget(QLabel("Open Positions"))
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(6)
        self.positions_table.setHorizontalHeaderLabels([
            "Symbol", "Dir", "Entry $", "Shares", "Unreal P&L", "P&L %"
        ])
        self.positions_table.horizontalHeader().setStretchLastSection(True)
        self.positions_table.setMaximumHeight(140)
        rl.addWidget(self.positions_table)

        # Trade history
        rl.addWidget(QLabel("Trade History"))
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(7)
        self.history_table.setHorizontalHeaderLabels([
            "Symbol", "Dir", "Entry", "Exit", "Shares", "P&L", "P&L %"
        ])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        rl.addWidget(self.history_table)

        # Mini equity chart
        self.equity_chart = ChartWidget()
        self.equity_chart.setMaximumHeight(180)
        rl.addWidget(self.equity_chart)

        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([900, 450])

        layout.addWidget(main_splitter)

        # ── Auto-refresh timer (5 seconds default) ─────────────────
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._auto_refresh)
        self.refresh_timer.start(5_000)

        self.data_tab.data_loaded.connect(self._on_data_loaded)
        self.training_tab.model_trained.connect(self._on_model_added)

    # ── Slots ───────────────────────────────────────────────────────

    def _on_data_loaded(self, symbol, df):
        if self.trade_symbol.findText(symbol) < 0:
            self.trade_symbol.addItem(symbol)

    def _on_model_added(self, model):
        self.trade_model_combo.addItem(f"{model.name}")
        self.trade_model_combo.setItemData(self.trade_model_combo.count() - 1, model)

    def _on_symbol_changed(self, text):
        sym = text.strip()
        if sym:
            self.ticker_symbol_label.setText(sym)
            self._refresh_chart()

    # ── Session management ──────────────────────────────────────────

    def _new_session(self):
        self.session = PaperSession(
            name=self.session_name.text(),
            initial_capital=self.capital_spin.value(),
            cash=self.capital_spin.value(),
        )
        self._update_display()

    def _save_session(self):
        if self.session:
            self.session.save()
            QMessageBox.information(self, "Saved", "Paper session saved.")

    def _load_session(self):
        sessions = PaperSession.list_sessions()
        if not sessions:
            QMessageBox.information(self, "No Sessions", "No saved sessions found.")
            return
        self.session = PaperSession.load(sessions[-1]["session_id"])
        self._update_display()

    # ── Chart refresh ───────────────────────────────────────────────

    def _refresh_chart(self):
        symbol = self.trade_symbol.currentText().strip()
        if not symbol:
            return

        period = self.chart_period.currentText()
        interval = self.chart_interval.currentText()

        def do_fetch():
            return fetch_ohlcv(symbol, period=period, interval=interval, cache_hours=0)

        self._chart_worker = Worker(do_fetch)
        self._chart_worker.finished.connect(lambda df: self._on_chart_data(symbol, df))
        self._chart_worker.error.connect(lambda msg: None)  # silent on auto-refresh
        self._chart_worker.start()

    def _on_chart_data(self, symbol, df):
        self._last_df = df
        if df is None or df.empty:
            return

        # Get indicator configs from the selected model
        indicator_configs = self._get_model_indicators()

        self.market_chart.plot_live_market(df, symbol, indicators=indicator_configs)

        # Update price ticker
        last = df.iloc[-1]
        price = last["Close"]
        open_price = last["Open"]
        prev_close = df["Close"].iloc[-2] if len(df) > 1 else open_price
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close != 0 else 0

        self.ticker_price_label.setText(f"{price:,.2f}")
        if change >= 0:
            self.ticker_change_label.setText(f"+{change:,.2f}  +{change_pct:.2f}%")
            self.ticker_change_label.setStyleSheet(f"color: {GREEN};")
            self.ticker_price_label.setStyleSheet(f"color: {GREEN};")
        else:
            self.ticker_change_label.setText(f"{change:,.2f}  {change_pct:.2f}%")
            self.ticker_change_label.setStyleSheet(f"color: {RED};")
            self.ticker_price_label.setStyleSheet(f"color: {RED};")

        vol = last["Volume"]
        vol_str = f"{vol/1e6:.2f}M" if vol >= 1e6 else f"{vol/1e3:.1f}K" if vol >= 1e3 else f"{vol:.0f}"
        self.ticker_ohlc_label.setText(
            f"O: {last['Open']:,.2f}   H: {last['High']:,.2f}   "
            f"L: {last['Low']:,.2f}   V: {vol_str}"
        )
        self.ticker_time_label.setText(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

    def _get_model_indicators(self) -> list[dict]:
        """Get indicator configs from the currently selected model."""
        idx = self.trade_model_combo.currentIndex()
        if idx >= 0:
            model = self.trade_model_combo.itemData(idx)
            if model and hasattr(model, "indicators") and model.indicators:
                return [{"name": n, "params": {}} for n in model.indicators]
        return self.indicators_tab.get_selected_indicators()

    # ── AI Signal ───────────────────────────────────────────────────

    def _get_signal(self):
        if self._signal_busy:
            return
        if not self.session:
            return

        symbol = self.trade_symbol.currentText().strip()
        if not symbol:
            return

        idx = self.trade_model_combo.currentIndex()
        if idx < 0:
            return

        model = self.trade_model_combo.itemData(idx)
        if model is None:
            return

        indicator_configs = [{"name": n, "params": {}} for n in model.indicators]
        self._signal_busy = True

        def do_signal():
            return self.session.get_model_signal(
                symbol, model, indicator_configs, period="1y", interval="1d"
            )

        self._signal_worker = Worker(do_signal)
        self._signal_worker.finished.connect(self._on_signal)
        self._signal_worker.error.connect(lambda msg: self._on_signal_done(f"Error: {msg}"))
        self._signal_worker.start()

    def _on_signal_done(self, err_msg=""):
        self._signal_busy = False
        if err_msg:
            self.signal_label.setText(f"Signal: {err_msg}")
            self.signal_label.setStyleSheet(f"color: {RED};")

    def _on_signal(self, result):
        self._signal_busy = False
        signal = result.get("signal", 0)
        confidence = result.get("confidence", 0)
        price = result.get("price", 0)
        error = result.get("error", "")

        if error:
            self.signal_label.setText(f"Signal: ERROR - {error[:60]}")
            self.signal_label.setStyleSheet(f"color: {RED};")
            return

        if signal == 1:
            text = f"BUY @ ${price:,.2f} (conf: {confidence:.1%})"
            self.signal_label.setText(f"Signal: {text}")
            self.signal_label.setStyleSheet(f"color: {GREEN};")
        else:
            text = f"SELL/HOLD @ ${price:,.2f} (conf: {confidence:.1%})"
            self.signal_label.setText(f"Signal: {text}")
            self.signal_label.setStyleSheet(f"color: {RED};")

        # Auto-trade if enabled
        if self.auto_trade_check.isChecked() and self.session:
            symbol = self.trade_symbol.currentText().strip()
            if signal == 1 and symbol not in self.session.positions:
                self.session.open_position(symbol, "LONG", price)
                self._update_display()
            elif signal == 0 and symbol in self.session.positions:
                self.session.close_position(symbol, price)
                self._update_display()

    # ── Manual trading ──────────────────────────────────────────────

    def _buy(self):
        if not self.session:
            QMessageBox.warning(self, "No Session", "Create a session first.")
            return
        symbol = self.trade_symbol.currentText().strip()
        if not symbol:
            return

        try:
            df = fetch_ohlcv(symbol, period="5d", interval="1d", cache_hours=0)
            price = float(df["Close"].iloc[-1])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not fetch price: {e}")
            return

        pos = self.session.open_position(symbol, "LONG", price)
        if pos:
            self._update_display()
        else:
            QMessageBox.warning(self, "Cannot Buy", "Already have a position or insufficient funds.")

    def _sell(self):
        if not self.session:
            return
        symbol = self.trade_symbol.currentText().strip()
        if symbol not in self.session.positions:
            QMessageBox.warning(self, "No Position", f"No open position for {symbol}.")
            return

        try:
            df = fetch_ohlcv(symbol, period="5d", interval="1d", cache_hours=0)
            price = float(df["Close"].iloc[-1])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not fetch price: {e}")
            return

        trade = self.session.close_position(symbol, price)
        if trade:
            self._update_display()

    # ── Auto refresh ────────────────────────────────────────────────

    def _refresh_prices(self):
        if not self.session or not self.session.positions:
            return
        prices = {}
        for sym in self.session.positions:
            try:
                df = fetch_ohlcv(sym, period="5d", interval="1d", cache_hours=0)
                prices[sym] = float(df["Close"].iloc[-1])
            except Exception:
                pass
        self.session.update_positions(prices)
        self._update_display()

    def _auto_refresh(self):
        if not self.session:
            return
        # Refresh the live chart
        if self.auto_chart_check.isChecked() and self.trade_symbol.currentText().strip():
            self._refresh_chart()
        # Refresh positions
        if self.session.positions:
            self._refresh_prices()
        # Fetch AI signal
        if self.auto_signal_check.isChecked() and self.trade_symbol.currentText().strip():
            self._get_signal()

    def _manual_refresh_all(self):
        """Force refresh everything now."""
        self._refresh_chart()
        if self.session and self.session.positions:
            self._refresh_prices()
        self._get_signal()

    def _update_timer_interval(self, seconds):
        self.refresh_timer.setInterval(seconds * 1000)

    # ── Display update ──────────────────────────────────────────────

    def _update_display(self):
        if not self.session:
            return

        # Summary
        eq = self.session.total_equity
        self.equity_label.setText(f"Equity: ${eq:,.2f}")
        self.cash_label.setText(f"Cash: ${self.session.cash:,.2f}")
        pnl = self.session.total_pnl
        self.pnl_label.setText(f"P&L: ${pnl:+,.2f}")
        self.pnl_label.setStyleSheet(f"color: {GREEN if pnl >= 0 else RED};")
        ret = self.session.total_return_pct
        self.return_label.setText(f"Ret: {ret:+.2f}%")
        self.return_label.setStyleSheet(f"color: {GREEN if ret >= 0 else RED};")

        # Positions table
        positions = self.session.positions
        self.positions_table.setRowCount(len(positions))
        for i, (sym, pos) in enumerate(positions.items()):
            items = [
                sym, pos.direction, f"${pos.entry_price:.2f}",
                f"{pos.shares:.4f}",
                f"${pos.unrealized_pnl:+.2f}",
                f"{pos.unrealized_pnl_pct:+.2f}%",
            ]
            for j, text in enumerate(items):
                item = QTableWidgetItem(text)
                if j >= 4:
                    item.setForeground(QColor(GREEN if pos.unrealized_pnl >= 0 else RED))
                self.positions_table.setItem(i, j, item)

        # History table
        trades = self.session.closed_trades
        self.history_table.setRowCount(len(trades))
        for i, t in enumerate(trades):
            items = [
                t.symbol, t.direction,
                f"${t.entry_price:.2f}", f"${t.exit_price:.2f}",
                f"{t.shares:.4f}", f"${t.pnl:+.2f}", f"{t.pnl_pct:+.2f}%",
            ]
            for j, text in enumerate(items):
                item = QTableWidgetItem(text)
                if j >= 5:
                    item.setForeground(QColor(GREEN if t.pnl > 0 else RED))
                self.history_table.setItem(i, j, item)

        # Equity chart
        self.equity_chart.plot_paper_equity(self.session.equity_history)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Trading Lab — Strategy Builder & Paper Trader")
        self.setMinimumSize(1400, 850)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        # Title bar
        title = QLabel("  AI Trading Lab")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setStyleSheet(f"color: {TEXT}; padding: 8px;")
        layout.addWidget(title)

        # Tab widget
        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        self.data_tab = DataTab()
        self.indicators_tab = IndicatorsTab(self.data_tab)
        self.training_tab = TrainingTab(self.data_tab, self.indicators_tab)
        self.backtest_tab = BacktestTab(self.data_tab, self.indicators_tab, self.training_tab)
        self.paper_tab = PaperTradingTab(self.data_tab, self.indicators_tab, self.training_tab)

        tabs.addTab(self.data_tab, "  Market Data")
        tabs.addTab(self.indicators_tab, "  Indicators")
        tabs.addTab(self.training_tab, "  Train Models")
        tabs.addTab(self.backtest_tab, "  Backtest")
        tabs.addTab(self.paper_tab, "  Paper Trading")

        layout.addWidget(tabs)

        # Status bar
        self.statusBar().showMessage("Ready — Load market data to begin")
        self.statusBar().setStyleSheet(f"color: {TEXT_DIM};")

        # Menu bar
        menubar = self.menuBar()
        menubar.setStyleSheet(f"background: {PANEL_BG}; color: {TEXT};")

        file_menu = menubar.addMenu("File")
        clear_cache_action = QAction("Clear Data Cache", self)
        clear_cache_action.triggered.connect(lambda: (clear_cache(), QMessageBox.information(self, "Cache", "Cache cleared.")))
        file_menu.addAction(clear_cache_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(lambda: QMessageBox.about(
            self, "About AI Trading Lab",
            "AI Trading Lab v1.0\n\n"
            "A desktop application for AI-powered trading strategy\n"
            "development, backtesting, and paper trading.\n\n"
            "Features:\n"
            "• Real-time market data (stocks, crypto, futures)\n"
            "• 20+ technical indicators\n"
            "• ML model training (RF, GBM, SVM, LSTM)\n"
            "• Walk-forward backtesting engine\n"
            "• Paper trading with virtual currency\n\n"
            "DISCLAIMER: This is for educational purposes only.\n"
            "Not financial advice. Trade at your own risk."
        ))
        help_menu.addAction(about_action)


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
