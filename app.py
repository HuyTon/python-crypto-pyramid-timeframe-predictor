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
import streamlit.components.v1 as components  # HTML divider iOS safe

from broker import Broker
from indicators import ema
from logs import log_prediction
from strategy import PyramidTimeframeStrategy

try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except Exception:
    PLOTLY_EVENTS_AVAILABLE = False


# ---------------------
# iOS SAFE divider
# ---------------------
def ios_safe_divider():
    components.html('<div style="border-top:1px solid #e5e7eb;margin:10px 0;"></div>', height=12)


# ---------------------
# Prediction history local file
# ---------------------
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
        "entry": float(row[4]) if row[4] else "",
        "tp": float(row[5]) if row[5] else "",
        "sl": float(row[6]) if row[6] else "",
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
        start = 1 if rows and rows[0][0].lower() == "timestamp" else 0
        for r in rows[start:]:
            if not r: continue
            try: out.append(_row_to_pred(r))
            except: pass
    return out

def save_pred_history(preds: list):
    with PRED_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","symbol","timeframe","side","entry","tp","sl","actual_price","outcome"])
        for rec in preds:
            writer.writerow(_pred_to_row(rec))

def clear_pred_history():
    if PRED_FILE.exists():
        PRED_FILE.unlink()
    st.session_state.pred_history = []


# ---------------------
# STREAMLIT UI
# ---------------------
st.set_page_config(page_title="Crypto Futures Dashboard", layout="wide")

with open("config.yaml","r") as f:
    CFG = yaml.safe_load(f)

POLL_SEC = int(CFG.get("poll_seconds", 60))
TOP_COINS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","XRPUSDT","DOGEUSDT","MATICUSDT","DOTUSDT","AVAXUSDT"]
timeframes = ["1m","5m","15m","1h","4h","1d","1w"]

tol_cfg = CFG.get("tolerances", {})
ENTRY_TOL = float(tol_cfg.get("entry_pct", 0.03)) / 100
HIT_TOL = float(tol_cfg.get("hit_pct", 0.03)) / 100

lvl_cfg = CFG.get("levels", {})
DRAW_TOP = int(lvl_cfg.get("draw_top", 3))
WIN_PCT = float(lvl_cfg.get("window_pct", 8)) / 100
TOL_PCT = float(lvl_cfg.get("cluster_tol_pct", 0.12))
POOL_MAX = int(lvl_cfg.get("max_per_side", 12))


# ---------------------
# Sidebar (NO MARKDOWN)
# ---------------------
st.sidebar.text("Settings")

st.sidebar.text("Top coin")
symbol = st.sidebar.selectbox("", TOP_COINS, index=1, label_visibility="collapsed")

st.sidebar.text("Timeframe")
tf = st.sidebar.selectbox("", timeframes, index=timeframes.index("1h"), label_visibility="collapsed")

st.sidebar.text("Auto refresh")
auto_refresh = st.sidebar.checkbox("", value=True, label_visibility="collapsed")

st.sidebar.text("Refresh seconds")
refresh_sec = st.sidebar.number_input("", 10, 300, POLL_SEC, step=10, label_visibility="collapsed")

st.sidebar.text("Draw Top-N S/R per side")
draw_top = st.sidebar.number_input("", 1, 10, DRAW_TOP, step=1, label_visibility="collapsed")

st.sidebar.text("Lock zoom")
lock_zoom = st.sidebar.toggle("", value=True, label_visibility="collapsed")


if draw_top != DRAW_TOP:
    CFG["levels"]["draw_top"] = int(draw_top)
    with open("config.yaml","w") as f: yaml.safe_dump(CFG, f)
    DRAW_TOP = draw_top


# ---------------------
# Clear Prediction History via selectbox (SAFE FOR iOS)
# ---------------------
ios_safe_divider()
st.sidebar.text("Actions")

action = st.sidebar.selectbox(
    "",
    ["None", "Clear Prediction History"],
    label_visibility="collapsed"
)

if action == "Clear Prediction History":
    clear_pred_history()
    st.sidebar.text("✅ Cleared.")


# ---------------------
# DATA FETCH
# ---------------------
broker = Broker(CFG)
engine = PyramidTimeframeStrategy(CFG)

@lru_cache(maxsize=64)
def fetch_df(symbol_key, timeframe_key):
    data = broker.fetch_ohlcv(symbol_key, timeframe=timeframe_key, limit=1000)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    # >>> FIX 1: ép float để tránh outlier do string/NaN
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms")
    return df


def _crosses(lo, hi, lv, tol):  # candle cross detection
    return (lo <= lv*(1+tol)) and (hi >= lv*(1-tol))


def evaluate_all_predictions(pred_list):
    changed = False
    for rec in list(pred_list):
        if rec.get("outcome"): continue

        sym, tf_ = rec["symbol"], rec["timeframe"]
        side = rec["side"]
        entry, tp, sl = float(rec["entry"]), float(rec["tp"]), float(rec["sl"])
        ts = rec["timestamp"]

        try:
            ts_ms = int(datetime.strptime(ts,"%Y%m%d %H:%M:%S").timestamp()*1000)
        except:
            continue

        df_tf = fetch_df(sym, tf_)
        df_after = df_tf[df_tf["ts"] >= ts_ms]
        if df_after.empty: continue

        entry_idx = None
        for idx, r in df_after.iloc[1:].iterrows():
            if _crosses(r["low"], r["high"], entry, ENTRY_TOL):
                entry_idx = idx; break
        if entry_idx is None: continue

        hit = None
        for _, r in df_after.loc[entry_idx:].iterrows():
            hi, lo, o = r["high"], r["low"], r["open"]
            if side == "BUY":
                tp_hit, sl_hit = hi >= tp*(1-HIT_TOL), lo <= sl*(1+HIT_TOL)
            else:
                tp_hit, sl_hit = lo <= tp*(1+HIT_TOL), hi >= sl*(1-HIT_TOL)

            if tp_hit and sl_hit:
                hit = ("correct", tp) if abs(o-tp)<abs(o-sl) else ("wrong", sl); break
            if tp_hit: hit=("correct", tp); break
            if sl_hit: hit=("wrong", sl); break

        if hit:
            rec["outcome"] = hit[0]
            rec["actual_price"] = f"{hit[1]:.4f}"
            log_prediction(rec, basename="predictions")
            changed = True
    return changed


# ---------------------
# RUN STRATEGY
# ---------------------
df = fetch_df(symbol, tf)
df_h4 = fetch_df(symbol,"4h")
df_h1 = fetch_df(symbol,"1h")
df_15 = fetch_df(symbol,"15m")

df["ema_s"] = ema(df["close"], int(CFG["strategy"]["ema_short"]))
df["ema_l"] = ema(df["close"], int(CFG["strategy"]["ema_long"]))
df["ema_sig"] = ema(df["close"], int(CFG["strategy"]["ema_signal"]))

side, entry, sl, tp, viz_levels, meta = engine.propose_trade(df_h4, df_h1, df_15, CFG)


# ---------------------
# Prediction History (persisted + session)
# ---------------------
if "pred_history" not in st.session_state:
    st.session_state.pred_history = load_pred_history()

_ts = datetime.now(SGT).strftime("%Y%m%d %H:%M:%S")

def same_pred(last):
    return (
        last["symbol"] == symbol and last["timeframe"] == tf and last["side"] == side and
        abs(last["entry"]-entry)<1e-9 and abs(last["sl"]-sl)<1e-9 and abs(last["tp"]-tp)<1e-9
    )

if st.session_state.pred_history:
    last = st.session_state.pred_history[-1]
    if same_pred(last):
        last["timestamp"] = _ts
    else:
        st.session_state.pred_history.append({"timestamp":_ts,"symbol":symbol,"side":side,"entry":entry,"sl":sl,"tp":tp,"timeframe":tf,"actual_price":"","outcome":""})
else:
    st.session_state.pred_history.append({"timestamp":_ts,"symbol":symbol,"side":side,"entry":entry,"sl":sl,"tp":tp,"timeframe":tf,"actual_price":"","outcome":""})

save_pred_history(st.session_state.pred_history)
st.session_state.pred_history = st.session_state.pred_history[-50:]


# ---------------------
# CHART
# ---------------------
fig = go.Figure()
# Candles (y1)
fig.add_candlestick(x=df["dt"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price")
# Volume (y2)
fig.add_bar(x=df["dt"], y=df["volume"], name="Volume", opacity=0.25, yaxis="y2")
# EMAs (y1)
fig.add_scatter(x=df["dt"], y=df["ema_s"], name="EMA short")
fig.add_scatter(x=df["dt"], y=df["ema_l"], name="EMA long")
fig.add_scatter(x=df["dt"], y=df["ema_sig"], name="EMA signal")
# Pred point (y1)
fig.add_scatter(x=[df["dt"].iloc[-1]], y=[entry], mode="markers+text",
                text=[f"{side} @ {entry:.2f}"], textposition="top center")

# TP/SL (y1)
fig.add_hline(y=tp, line_color="#16c784", line_dash="dot", annotation_text=f"TP {tp:.2f}")
fig.add_hline(y=sl, line_color="#ea3943", line_dash="dot", annotation_text=f"SL {sl:.2f}")

price_now = float(df["close"].iloc[-1])

def cluster(levels, tol_pct):
    merged=[]
    for px,sc in sorted(levels, key=lambda x:x[0]):
        if not merged:
            merged.append([px,sc,1]); continue
        if abs(px-merged[-1][0])/merged[-1][0] <= tol_pct/100:
            mpx,msc,count = merged[-1]
            merged[-1][0]=(mpx*count+px)/(count+1)
            merged[-1][1]=(msc*count+sc)/(count+1)
            merged[-1][2]=count+1
        else:
            merged.append([px,sc,1])
    return [(px,sc) for px,sc,_ in merged]

supports = resistance = []
if isinstance(viz_levels,dict):
    sup_raw = [(p,s) for p,s in viz_levels["supports"] if abs(p-price_now)/price_now<=WIN_PCT]
    res_raw = [(p,s) for p,s in viz_levels["resists"] if abs(p-price_now)/price_now<=WIN_PCT]
    sups = sorted(cluster(sup_raw, TOL_PCT), key=lambda x: (-x[1], abs(x[0]-price_now)))[:POOL_MAX]
    ress = sorted(cluster(res_raw, TOL_PCT), key=lambda x: (-x[1], abs(x[0]-price_now)))[:POOL_MAX]
    supports=sups[:DRAW_TOP]; resistance=ress[:DRAW_TOP]

    for px,sc in supports:
        fig.add_hline(y=px, line_color="#16c784", opacity=0.6, line_dash="dot", annotation_text=f"SUP {sc:.0f}% @ {px:.2f}")
    for px,sc in resistance:
        fig.add_hline(y=px, line_color="#ea3943", opacity=0.6, line_dash="dot", annotation_text=f"RES {sc:.0f}% @ {px:.2f}")

# >>> FIX 2: đảm bảo bar luôn là y2, và y2 không match y1
fig.update_traces(yaxis="y2", selector=dict(type="bar"))
fig.update_layout(
    xaxis=dict(rangeslider=dict(visible=False)),
    yaxis=dict(title="Price", rangemode="tozero"),
    yaxis2=dict(
        title="Vol",
        overlaying="y",
        side="right",
        rangemode="tozero",
        showgrid=False,
        matches=None  # khóa không “match” y1
    ),
    height=740,
    uirevision="keep"
)

# >>> FIX 3: clamp y-axis theo percentile để né outlier
# Dùng cả high/low/close để tính khoảng giá sạch
px_all = np.concatenate([
    df["low"].astype(float).values,
    df["close"].astype(float).values,
    df["high"].astype(float).values,
])
lo = float(np.nanpercentile(px_all, 0.5))
hi = float(np.nanpercentile(px_all, 99.5))
if hi > lo:
    pad = 0.02 * (hi - lo)
    fig.update_yaxes(range=[lo - pad, hi + pad])

# Zoom lock (session remembered)
if "x_range" in st.session_state:
    fig.update_xaxes(range=st.session_state["x_range"])
if "y_range" in st.session_state:
    fig.update_yaxes(range=st.session_state["y_range"])

if PLOTLY_EVENTS_AVAILABLE and lock_zoom:
    ev = plotly_events(fig, events=["plotly_relayout"], key="relayout")
    if ev:
        for e in ev:
            if "xaxis.range[0]" in e:
                st.session_state["x_range"]=[e["xaxis.range[0]"],e["xaxis.range[1]"]]
            if "yaxis.range[0]" in e:
                st.session_state["y_range"]=[e["yaxis.range[0]"],e["yaxis.range[1]"]]
else:
    st.plotly_chart(fig, use_container_width=True)


# Sidebar list
ios_safe_divider()
st.sidebar.text("Supports/Resistances")
# for px,sc in supports: st.sidebar.text(f"  SUP {sc:.0f}% @ {px:.2f}")
# st.sidebar.text("Resistances")
# for px,sc in resistance: st.sidebar.text(f"  RES {sc:.0f}% @ {px:.2f}")
with st.sidebar:
    ios_safe_divider()
    st.text("Supports / Resistances")

    # ---- render supports (GREEN) ----
    for px, sc in supports:
        components.html(
            f"""
            <div style="font-size:14px; color:#16c784; font-weight:500;">
                SUP {sc:.0f}% @ {px:.2f}
            </div>
            """,
            height=22, scrolling=False
        )

    # ---- render resistances (RED) ----
    for px, sc in resistance:
        components.html(
            f"""
            <div style="font-size:14px; color:#ea3943; font-weight:500;">
                RES {sc:.0f}% @ {px:.2f}
            </div>
            """,
            height=22, scrolling=False
        )

ios_safe_divider()
st.sidebar.text("Prediction History")
hist_df = pd.DataFrame(st.session_state.pred_history)
st.sidebar.dataframe(hist_df.tail(50), use_container_width=True)


# evaluate predictions
evaluate_all_predictions(st.session_state.pred_history)


if auto_refresh:
    import time
    time.sleep(refresh_sec)
    st.rerun()
