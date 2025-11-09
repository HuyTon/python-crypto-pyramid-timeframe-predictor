import os, yaml
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from functools import lru_cache
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import csv

from broker import Broker
from indicators import ema, rsi
from logs import log_prediction, import_logs
from strategy import PyramidTimeframeStrategy

try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except Exception:
    PLOTLY_EVENTS_AVAILABLE = False

# --- Timezone & storage config for Prediction History ---
SGT = ZoneInfo("Asia/Singapore")
PRED_DIR = Path("data")
PRED_DIR.mkdir(exist_ok=True)
PRED_FILE = PRED_DIR / "prediction_history.csv"

def _pred_to_row(rec: dict) -> list:
    return [
        rec.get("timestamp",""),
        rec.get("symbol",""),
        rec.get("timeframe",""),
        rec.get("side",""),
        rec.get("entry",""),
        rec.get("tp",""),
        rec.get("sl",""),
        rec.get("actual_price",""),
        rec.get("outcome",""),
    ]

def _row_to_pred(row: list) -> dict:
    return {
        "timestamp": row[0],
        "symbol": row[1],
        "timeframe": row[2],
        "side": row[3],
        "entry": float(row[4]) if row[4] != "" else "",
        "tp": float(row[5]) if row[5] != "" else "",
        "sl": float(row[6]) if row[6] != "" else "",
        "actual_price": row[7],
        "outcome": row[8],
    }

def load_pred_history() -> list:
    if not PRED_FILE.exists():
        return []
    out = []
    with PRED_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        if not rows:
            return []
        start_idx = 1 if rows and rows[0] and rows[0][0].lower() == "timestamp" else 0
        for r in rows[start_idx:]:
            if not r:
                continue
            try:
                out.append(_row_to_pred(r))
            except Exception:
                pass
    return out

def save_pred_history(preds: list):
    header = ["timestamp","symbol","timeframe","side","entry","tp","sl","actual_price","outcome"]
    with PRED_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for rec in preds:
            writer.writerow(_pred_to_row(rec))

def clear_pred_history():
    try:
        if PRED_FILE.exists():
            PRED_FILE.unlink()
    except Exception:
        pass
    st.session_state.pred_history = []

st.set_page_config(page_title="Crypto Futures Dashboard", layout="wide")

with open("config.yaml","r") as f:
    CFG = yaml.safe_load(f)

POLL_SEC = int(CFG.get("poll_seconds", 60))
TOP_COINS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","XRPUSDT","DOGEUSDT","MATICUSDT","DOTUSDT","AVAXUSDT"]
timeframes = ["1m","5m","15m","1h","4h","1d","1w"]

tol_cfg = CFG.get("tolerances", {})
ENTRY_TOL = float(tol_cfg.get("entry_pct", 0.03)) / 100.0
HIT_TOL = float(tol_cfg.get("hit_pct", 0.03)) / 100.0

levels_cfg = CFG.get("levels", {})
DRAW_TOP = int(levels_cfg.get("draw_top", 3))
WIN_PCT = float(levels_cfg.get("window_pct", 8))/100.0
TOL_PCT = float(levels_cfg.get("cluster_tol_pct", 0.12))
POOL_MAX = int(levels_cfg.get("max_per_side", 12))

# ----- iOS-safe sidebar (NO Markdown-based labels) -----
st.sidebar.text("Settings")
st.sidebar.text("Top coin")
symbol = st.sidebar.selectbox(
    "",
    TOP_COINS,
    index=TOP_COINS.index(CFG.get("default_symbol","ETHUSDT")) if CFG.get("default_symbol","ETHUSDT") in TOP_COINS else 1,
    label_visibility="collapsed",
)
st.sidebar.text("Timeframe")
tf = st.sidebar.selectbox(
    "",
    timeframes,
    index=timeframes.index("1h"),
    label_visibility="collapsed",
)
st.sidebar.text("Auto refresh")
auto_refresh = st.sidebar.checkbox(
    "",
    value=True,
    label_visibility="collapsed",
)
st.sidebar.text("Refresh seconds")
refresh_sec = st.sidebar.number_input(
    "",
    min_value=10, max_value=300, value=POLL_SEC, step=10,
    label_visibility="collapsed",
)
st.sidebar.text("Draw Top-N levels per side")
draw_top = st.sidebar.number_input(
    "",
    min_value=1, max_value=10, value=DRAW_TOP, step=1,
    label_visibility="collapsed",
)
st.sidebar.text("🔒 Lock zoom when you adjust")
lock_zoom = st.sidebar.toggle(
    "",
    value=True,
    label_visibility="collapsed",
)

st.sidebar.divider()
if st.sidebar.button("🧹 Clear Prediction History", use_container_width=True):
    clear_pred_history()
    st.sidebar.text("Cleared prediction history.")

if draw_top != DRAW_TOP:
    CFG["levels"]["draw_top"] = int(draw_top)
    with open("config.yaml","w") as f: yaml.safe_dump(CFG, f)
    DRAW_TOP = int(draw_top)

broker = Broker(CFG)
engine = PyramidTimeframeStrategy(CFG)

@lru_cache(maxsize=64)
def fetch_df(symbol_key: str, timeframe_key: str):
    data = broker.fetch_ohlcv(symbol_key, timeframe=timeframe_key, limit=1000)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms")
    return df

def _crosses(price_low, price_high, level, tol):
    return (price_low <= level*(1+tol)) and (price_high >= level*(1-tol))

def evaluate_all_predictions(pred_list):
    changed = False
    for i, rec in enumerate(list(pred_list)):
        try:
            if rec.get("outcome"): continue
            sym = rec.get("symbol"); tframe = rec.get("timeframe","1h")
            side = (rec.get("side","")).upper()
            entry = float(rec.get("entry")); tp = float(rec.get("tp")); sl = float(rec.get("sl"))
            ts_raw = rec.get("timestamp", 0)
            if isinstance(ts_raw, (int, float)):
                ts_pred_ms = int(ts_raw) * 1000
            else:
                try:
                    ts_pred_ms = int(datetime.strptime(str(ts_raw), "%Y%m%d %H:%M:%S").timestamp() * 1000)
                except Exception:
                    try: ts_pred_ms = int(float(ts_raw)) * 1000
                    except Exception: ts_pred_ms = 0
            df_tf = fetch_df(sym, tframe); df_after = df_tf[df_tf["ts"] >= ts_pred_ms]
            if df_after.empty or len(df_after) < 2: continue
            entry_idx = None
            for idx, row in df_after.iloc[1:].iterrows():
                if _crosses(float(row["low"]), float(row["high"]), entry, ENTRY_TOL):
                    entry_idx = idx; break
            if entry_idx is None: continue
            after_entry = df_after.loc[entry_idx:]
            hit=None; price_hit=None
            for _, row in after_entry.iloc[1:].iterrows():
                hi = float(row["high"]); lo = float(row["low"]); o = float(row["open"])
                if side == "BUY":
                    tp_hit = hi >= tp*(1 - HIT_TOL); sl_hit = lo <= sl*(1 + HIT_TOL)
                    if tp_hit and sl_hit:
                        if abs(o - sl) < abs(o - tp): hit=("wrong","SL_FIRST"); price_hit=sl
                        else: hit=("correct","TP_FIRST"); price_hit=tp
                        break
                    elif tp_hit: hit=("correct","TP"); price_hit=tp; break
                    elif sl_hit: hit=("wrong","SL"); price_hit=sl; break
                else:
                    tp_hit = lo <= tp*(1 + HIT_TOL); sl_hit = hi >= sl*(1 - HIT_TOL)
                    if tp_hit and sl_hit:
                        if abs(o - sl) < abs(o - tp): hit=("wrong","SL_FIRST"); price_hit=sl
                        else: hit=("correct","TP_FIRST"); price_hit=tp
                        break
                    elif tp_hit: hit=("correct","TP"); price_hit=tp; break
                    elif sl_hit: hit=("wrong","SL"); price_hit=sl; break
            if hit:
                outcome_label, _ = hit
                rec["outcome"] = outcome_label; rec["actual_price"] = f"{price_hit:.4f}"
                from logs import log_prediction
                log_prediction(rec, basename="predictions"); changed=True
        except Exception:
            pass
    return changed

df = fetch_df(symbol, tf)
df_h4 = fetch_df(symbol, "4h")
df_h1 = fetch_df(symbol, "1h")
df_15 = fetch_df(symbol, "15m")

df["ema_s"] = ema(df["close"], int(CFG["strategy"]["ema_short"]))
df["ema_l"] = ema(df["close"], int(CFG["strategy"]["ema_long"]))
df["ema_sig"] = ema(df["close"], int(CFG["strategy"]["ema_signal"]))

side, entry, sl, tp, viz_levels, meta = engine.propose_trade(df_h4, df_h1, df_15, CFG)

# --- Load existing history (persisted) ---
if "pred_history" not in st.session_state:
    st.session_state.pred_history = load_pred_history()

# --- Timestamp in Singapore timezone ---
_ts = datetime.now(SGT).strftime("%Y%m%d %H:%M:%S")

def _same_pred(last, symbol, tf, side, entry, sl, tp):
    try:
        return (last.get("symbol")==symbol and last.get("timeframe")==tf and last.get("side")==side and
                abs(float(last.get("entry",0.0))-float(entry))<1e-9 and
                abs(float(last.get("sl",0.0))-float(sl))<1e-9 and
                abs(float(last.get("tp",0.0))-float(tp))<1e-9)
    except Exception: return False

# --- Append or just update timestamp, then persist to file ---
if st.session_state.pred_history:
    last = st.session_state.pred_history[-1]
    if _same_pred(last, symbol, tf, side, float(entry), float(sl), float(tp)):
        last["timestamp"] = _ts
    else:
        st.session_state.pred_history.append({
            "timestamp":_ts,"symbol":symbol,"side":side,
            "entry":float(entry),"sl":float(sl),"tp":float(tp),
            "timeframe":tf,"actual_price":"","outcome":""
        })
else:
    st.session_state.pred_history.append({
        "timestamp":_ts,"symbol":symbol,"side":side,
        "entry":float(entry),"sl":float(sl),"tp":float(tp),
        "timeframe":tf,"actual_price":"","outcome":""
    })

# Persist current (possibly truncated by your UI preference below)
save_pred_history(st.session_state.pred_history)

# Keep only last N in session (file already saved)
st.session_state.pred_history = st.session_state.pred_history[-int(CFG.get("ui",{}).get("max_predictions",50)):]

def cluster_levels(levels, tol_pct=0.12):
    merged = []
    levels = sorted(levels, key=lambda x: x[0])
    for px, sc in levels:
        if not merged:
            merged.append([px, sc, 1]); continue
        lpx, lsc, cnt = merged[-1]
        if abs(px - lpx)/lpx <= tol_pct/100.0:
            wsum = lsc*cnt + sc
            merged[-1][0] = (lpx*cnt + px)/(cnt+1)
            merged[-1][1] = wsum/(cnt+1)
            merged[-1][2] = cnt+1
        else:
            merged.append([px, sc, 1])
    return [(p, s) for p, s, _ in merged]

fig = go.Figure()
fig.add_candlestick(x=df["dt"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price")
vol_color = np.where(df["close"]>=df["open"], "#16c784", "#ea3943")
fig.add_bar(x=df["dt"], y=df["volume"], name="Volume", opacity=0.25, marker_color=vol_color, yaxis="y2")
fig.add_scatter(x=df["dt"], y=df["ema_s"], name=f"EMA {CFG['strategy']['ema_short']}", line=dict(width=1.5, color="#f39c12"))
fig.add_scatter(x=df["dt"], y=df["ema_l"], name=f"EMA {CFG['strategy']['ema_long']}", line=dict(width=1.5, color="#1abc9c"))
fig.add_scatter(x=df["dt"], y=df["ema_sig"], name=f"EMA {CFG['strategy']['ema_signal']}", line=dict(width=1.5, color="#9b59b6"))
fig.add_scatter(x=[df["dt"].iloc[-1]], y=[entry], mode="markers+text", name=f"Pred {side}",
                marker=dict(size=12, color="#2ecc71" if side=="BUY" else "#e74c3c"),
                text=[f"{side} @ {entry:.2f}"], textposition="top center")
fig.add_hline(y=tp, line_color="#2ecc71", line_dash="dot", annotation_text=f"TP {tp:.2f}")
fig.add_hline(y=sl, line_color="#e74c3c", line_dash="dot", annotation_text=f"SL {sl:.2f}")

price_now = float(df["close"].iloc[-1])
sups = ress = []
if isinstance(viz_levels, dict):
    sup_raw = [(px, sc) for px, sc in viz_levels.get("supports", []) if abs(px - price_now)/price_now <= WIN_PCT]
    res_raw = [(px, sc) for px, sc in viz_levels.get("resists", [])  if abs(px - price_now)/price_now <= WIN_PCT]
    sup_c = cluster_levels(sup_raw, tol_pct=TOL_PCT)
    res_c = cluster_levels(res_raw, tol_pct=TOL_PCT)
    sups_pool = sorted(sup_c, key=lambda x: (-x[1], abs(x[0]-price_now)))[:POOL_MAX]
    ress_pool = sorted(res_c, key=lambda x: (-x[1], abs(x[0]-price_now)))[:POOL_MAX]
    sups = sups_pool[:DRAW_TOP]; ress = ress_pool[:DRAW_TOP]
    for i,(px, score) in enumerate(sups):
        fig.add_hline(y=px, line_color="#16c784", opacity=0.6, line_dash="dot",
                      annotation_text=f"SUP {score:.0f}% @ {px:.2f}", annotation_position="top left" if i%2==0 else "bottom left")
    for i,(px, score) in enumerate(ress):
        fig.add_hline(y=px, line_color="#ea3943", opacity=0.6, line_dash="dot",
                      annotation_text=f"RES {score:.0f}% @ {px:.2f}", annotation_position="top right" if i%2==0 else "bottom right")

fig.update_layout(
    xaxis=dict(rangeslider=dict(visible=False)),
    yaxis=dict(title="Price"),
    yaxis2=dict(title="Vol", overlaying="y", side="right", showgrid=False),
    legend=dict(orientation="h"),
    margin=dict(l=10,r=10,t=40,b=10),
    height=720,
    uirevision="keep-zoom"
)

# Apply saved ranges before render
xr = st.session_state.get("x_range"); yr = st.session_state.get("y_range")
if xr: fig.update_xaxes(range=xr)
if yr: fig.update_yaxes(range=yr)

# Capture zoom events (if available) without relying on markdown components
if PLOTLY_EVENTS_AVAILABLE and lock_zoom:
    try:
        ev = plotly_events(fig, events=["plotly_relayout"], key="chart_ev", override_width="100%", override_height=720)
        if ev:
            for e in ev:
                if isinstance(e, dict):
                    r0 = e.get("xaxis.range[0]"); r1 = e.get("xaxis.range[1]")
                    ry0 = e.get("yaxis.range[0]"); ry1 = e.get("yaxis.range[1]")
                    if r0 is not None and r1 is not None:
                        st.session_state["x_range"] = [r0, r1]
                    if ry0 is not None and ry1 is not None:
                        st.session_state["y_range"] = [ry0, ry1]
                    if e.get("xaxis.autorange"): st.session_state.pop("x_range", None)
                    if e.get("yaxis.autorange"): st.session_state.pop("y_range", None)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)
else:
    st.plotly_chart(fig, use_container_width=True)

# ----- Sidebar blocks WITHOUT Markdown to avoid iOS regex crash -----
st.sidebar.divider()
st.sidebar.text("Top Levels (drawn)")
if sups or ress:
    if sups:
        st.sidebar.text("Supports")
        for px, sc in sups:
            st.sidebar.text(f"- {px:.2f}  · {sc:.0f}%")
    if ress:
        st.sidebar.text("Resistances")
        for px, sc in ress:
            st.sidebar.text(f"- {px:.2f}  · {sc:.0f}%")

st.sidebar.divider()
st.sidebar.text("Prediction History")
hist_df = pd.DataFrame(st.session_state.pred_history)
st.sidebar.dataframe(hist_df.tail(50), use_container_width=True)

# Evaluate unresolved predictions
_ = evaluate_all_predictions(st.session_state.pred_history)

if auto_refresh:
    import time as _t
    _t.sleep(int(refresh_sec))
    st.rerun()
