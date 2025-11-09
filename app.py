import os, yaml, json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from functools import lru_cache
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import csv
import httpx
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
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

PRED_FILE = DATA_DIR / "prediction_history.csv"
TELE_CFG_FILE = DATA_DIR / "telegram.yaml"
TELE_STATE_FILE = DATA_DIR / "telewatch_state.json"

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
        start = 1 if rows and rows[0] and rows[0][0].lower() == "timestamp" else 0
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
# Telegram helpers
# ---------------------
def load_telegram_cfg() -> dict:
    if TELE_CFG_FILE.exists():
        try:
            return yaml.safe_load(TELE_CFG_FILE.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    return {}

def save_telegram_cfg(cfg: dict):
    TELE_CFG_FILE.parent.mkdir(exist_ok=True)
    with TELE_CFG_FILE.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

def _load_tele_state() -> dict:
    if TELE_STATE_FILE.exists():
        try:
            return json.loads(TELE_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_tele_state(state: dict):
    TELE_STATE_FILE.parent.mkdir(exist_ok=True)
    TELE_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

def send_telegram(bot_token: str, chat_id: str, text: str) -> bool:
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        with httpx.Client(timeout=10) as cli:
            r = cli.post(url, data=payload)
            r.raise_for_status()
            return True
    except Exception:
        return False

def get_telegram_token():
    try:
        return st.secrets["TELEGRAM_BOT_TOKEN"]
    except Exception:
        pass
    return os.environ.get("TELEGRAM_BOT_TOKEN")

def get_telegram_chat_id():
    try:
        return st.secrets["TELEGRAM_CHAT_ID"]
    except Exception:
        pass
    return os.environ.get("TELEGRAM_CHAT_ID")

def notify_if_entry_changed(sym: str, side: str, entry: float, tp: float, sl: float,
                            price_now: float, tele_cfg: dict):
    if not tele_cfg.get("enabled"):
        return
    token = get_telegram_token()
    chat_id = get_telegram_chat_id()
    if not token or not chat_id:
        return

    state = _load_tele_state()
    last = state.get(sym, {}).get("last_entry")
    changed = False
    if last is None:
        changed = True
    else:
        try:
            prev = float(last)
            base = prev if prev != 0 else entry
            delta = abs(entry - prev) / max(1e-9, base) * 100.0
            changed = delta >= float(tele_cfg.get("min_change_pct", 0.02))
        except Exception:
            changed = True

    if not changed:
        return

    ts = datetime.now(SGT).strftime("%Y-%m-%d %H:%M:%S")
    emoji = "🟢" if side.upper() == "BUY" else "🔴"
    txt = (
        f"{emoji} <b>{sym}</b> — Entry changed\n"
        f"Time: {ts} SGT\n"
        f"Side: <b>{side}</b>\n"
        f"Entry: <b>{entry:.2f}</b>\n"
        f"TP: {tp:.2f} | SL: {sl:.2f}\n"
        f"Price now: {price_now:.2f}"
    )
    ok = send_telegram(token, chat_id, txt)
    if ok:
        state[sym] = {"last_entry": float(entry)}
        _save_tele_state(state)

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
symbol = st.sidebar.selectbox(
    "", TOP_COINS, index=1, label_visibility="collapsed", key="sel_symbol"
)

st.sidebar.text("Timeframe")
tf = st.sidebar.selectbox(
    "", timeframes, index=timeframes.index("1h"), label_visibility="collapsed", key="sel_tf"
)

st.sidebar.text("Auto refresh")
auto_refresh = st.sidebar.checkbox(
    "", value=True, label_visibility="collapsed", key="auto_refresh"
)

st.sidebar.text("Refresh seconds")
refresh_sec = st.sidebar.number_input(
    "", 10, 300, POLL_SEC, step=10, label_visibility="collapsed", key="refresh_sec"
)

st.sidebar.text("Draw Top-N S/R per side")
draw_top = st.sidebar.number_input(
    "", 1, 10, DRAW_TOP, step=1, label_visibility="collapsed", key="draw_top"
)

st.sidebar.text("Lock zoom")
lock_zoom = st.sidebar.toggle(
    "", value=True, label_visibility="collapsed", key="lock_zoom"
)

if draw_top != DRAW_TOP:
    CFG["levels"]["draw_top"] = int(draw_top)
    with open("config.yaml","w") as f: yaml.safe_dump(CFG, f)
    DRAW_TOP = draw_top

# ---------------------
# Actions (add 'Send test Telegram' here)
# ---------------------
ios_safe_divider()
st.sidebar.text("Actions")

action = st.sidebar.selectbox(
    "",
    ["None", "Clear Prediction History", "Send test Telegram"],
    label_visibility="collapsed",
    key="actions_select"
)
if action == "Clear Prediction History":
    clear_pred_history()
    st.sidebar.text("✅ Cleared.")
elif action == "Send test Telegram":
    token = get_telegram_token()
    chat_id = get_telegram_chat_id()
    ts = datetime.now(SGT).strftime("%Y-%m-%d %H:%M:%S")
    ok = send_telegram(token, chat_id, f"✅ Test from app at {ts} SGT")
    st.sidebar.text("✅ Sent." if ok else "❌ Failed (check token/chat id)")

# ---------------------
# Telegram Alerts (sidebar, iOS-safe)
# (No Save / No Send buttons anymore — using secrets + existing YAML for non-secret fields)
# ---------------------
ios_safe_divider()
st.sidebar.text("Telegram Alerts")

_tele_cfg = load_telegram_cfg()
en_default      = bool(_tele_cfg.get("enabled", False))
token_default   = get_telegram_token()
chat_default    = get_telegram_chat_id()
watch_default   = _tele_cfg.get("watch_symbols", ["ETHUSDT"])
minchg_default  = float(_tele_cfg.get("min_change_pct", 0.02))  # %

enable_alerts = st.sidebar.checkbox(
    "", value=en_default, label_visibility="collapsed", key="telegram_enable_alerts"
)

# st.sidebar.text("Bot Token (from secrets)")
# st.sidebar.text_input(
#     "", value=token_default or "", type="password", label_visibility="collapsed", key="tg_token",
#     disabled=True
# )

# st.sidebar.text("Chat ID (from secrets)")
# st.sidebar.text_input(
#     "", value=chat_default or "", label_visibility="collapsed", key="tg_chat_id",
#     disabled=True
# )

st.sidebar.text("Watch Symbols")
watch_syms = st.sidebar.multiselect(
    "", TOP_COINS, default=watch_default, label_visibility="collapsed", key="tg_watch"
)

st.sidebar.text("Min Entry Change (%)")
min_change_pct = st.sidebar.number_input(
    "", min_value=0.0, max_value=5.0, value=float(minchg_default), step=0.01,
    label_visibility="collapsed", key="tg_min_change"
)

# Auto-persist non-secret prefs on change (enabled, watchlist, min_change)
new_cfg = {
    "enabled": bool(enable_alerts),
    # bot_token/chat_id are not persisted (read from secrets)
    "watch_symbols": list(watch_syms),
    "min_change_pct": float(min_change_pct),
}
save_telegram_cfg(new_cfg)

# ---------------------
# DATA FETCH
# ---------------------
broker = Broker(CFG)
engine = PyramidTimeframeStrategy(CFG)

@lru_cache(maxsize=64)
def fetch_df(symbol_key, timeframe_key):
    data = broker.fetch_ohlcv(symbol_key, timeframe=timeframe_key, limit=1000)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    # ép float để tránh outlier do string/NaN
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
# RUN STRATEGY (current symbol)
# ---------------------
df = fetch_df(symbol, tf)
df_h4 = fetch_df(symbol,"4h")
df_h1 = fetch_df(symbol,"1h")
df_15 = fetch_df(symbol,"15m")

df["ema_s"] = ema(df["close"], int(CFG["strategy"]["ema_short"]))
df["ema_l"] = ema(df["close"], int(CFG["strategy"]["ema_long"]))
df["ema_sig"] = ema(df["close"], int(CFG["strategy"]["ema_signal"]))

side, entry, sl, tp, viz_levels, meta = engine.propose_trade(df_h4, df_h1, df_15, CFG)

# --- Telegram notify for current symbol
tele_cfg_runtime = load_telegram_cfg()
if tele_cfg_runtime.get("enabled"):
    price_now_current = float(df["close"].iloc[-1])
    notify_if_entry_changed(symbol, side, float(entry), float(tp), float(sl), price_now_current, tele_cfg_runtime)

# --- Telegram notify for watch list (other symbols)
watch_list = tele_cfg_runtime.get("watch_symbols", [])
if tele_cfg_runtime.get("enabled") and watch_list:
    for sym in [s for s in watch_list if s != symbol]:
        try:
            df_sym = fetch_df(sym, tf)
            df_h4_s = fetch_df(sym, "4h")
            df_h1_s = fetch_df(sym, "1h")
            df_15_s = fetch_df(sym, "15m")
            s_side, s_entry, s_sl, s_tp, _, _ = engine.propose_trade(df_h4_s, df_h1_s, df_15_s, CFG)
            price_now_s = float(df_sym["close"].iloc[-1])
            notify_if_entry_changed(sym, s_side, float(s_entry), float(s_tp), float(s_sl), price_now_s, tele_cfg_runtime)
        except Exception:
            pass

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

# Giữ y2 riêng, tránh match y1
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
        matches=None
    ),
    height=740,
    uirevision="keep"
)

# clamp y theo percentile để né outlier
px_all = np.concatenate([df["low"].values, df["close"].values, df["high"].values]).astype(float)
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

# ---- render supports + resistances on the SAME line ----
max_len = max(len(supports), len(resistance))
components.html('<div style="border-top:2px solid #000000;"></div>', height=16)
components.html('<div style="margin:5px 0;font-size:16px;font-family:Arial, sans-serif;">Supports / Resistances</div>', height=25)

for i in range(max_len):
    sup_txt = ""
    res_txt = ""

    if i < len(supports):
        px, sc = supports[i]
        sup_txt = f'<span style="color:#16c784;font-family:Arial, sans-serif;font-weight:500;">SUP {sc:.0f}% @ {px:.2f}</span>'

    if i < len(resistance):
        px, sc = resistance[i]
        res_txt = f'<span style="color:#ea3943;font-family:Arial, sans-serif;font-weight:500;">RES {sc:.0f}% @ {px:.2f}</span>'

    components.html(
        f"""
        <div style="
            display:flex;            
            font-size:14px;
            width:100%;
        ">
            <div>{sup_txt}</div>
            <div style="margin-left:10px;">{res_txt}</div>
        </div>
        """,
        height=24,
        scrolling=False
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
