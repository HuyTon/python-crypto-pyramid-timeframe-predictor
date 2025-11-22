import os, yaml, json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import csv
import httpx
import streamlit.components.v1 as components  # HTML divider iOS safe
from typing import Optional

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
MAX_PRED_RECORDS_PER_SYMBOL = 3

def _safe_float(val):
    try:
        if val in ("", None):
            return ""
        return float(val)
    except (TypeError, ValueError):
        return ""

def _build_entry_slots(primary_entry, candidates):
    slots = []
    primary_val = _safe_float(primary_entry)
    for cand in candidates or []:
        price = _safe_float((cand or {}).get("price"))
        if price == "":
            continue
        score = _safe_float((cand or {}).get("score"))
        slots.append((price, score))
    if primary_val != "":
        match_idx = next(
            (idx for idx, (price, _) in enumerate(slots) if price != "" and abs(price - primary_val) < 1e-6),
            None
        )
        if match_idx is not None:
            slots.insert(0, slots.pop(match_idx))
        else:
            if slots:
                slots[0] = (primary_val, slots[0][1])
            else:
                slots.append((primary_val, ""))
    if not slots:
        slots.append(("", ""))
    while len(slots) < 3:
        slots.append(("", ""))
    return slots[:3]

def _keep_last_symbol_records(history: list, symbol: str, keep: int):
    keep = max(keep, 0)
    matches = [idx for idx, rec in enumerate(history) if rec.get("symbol") == symbol]
    drop = max(0, len(matches) - keep)
    if drop <= 0:
        return
    new_history = []
    removed = 0
    for rec in history:
        if rec.get("symbol") == symbol and removed < drop:
            removed += 1
            continue
        new_history.append(rec)
    history[:] = new_history

def enforce_symbol_record_limit(history: list, limit: int = MAX_PRED_RECORDS_PER_SYMBOL):
    if limit <= 0:
        history.clear()
        return
    symbols = {rec.get("symbol") for rec in history}
    for sym in symbols:
        _keep_last_symbol_records(history, sym, limit)

def _pred_to_row(rec: dict) -> list:
    return [
        rec.get("timestamp",""),
        rec.get("symbol",""),
        rec.get("timeframe",""),
        rec.get("side",""),
        rec.get("entry",""),
        rec.get("tp",""),
        rec.get("sl",""),
        rec.get("market",""),
        rec.get("actual_price",""),
        rec.get("outcome",""),
        rec.get("entry1_score",""),
        rec.get("entry2",""),
        rec.get("entry2_score",""),
        rec.get("entry3",""),
        rec.get("entry3_score",""),
    ]

def _row_to_pred(row: list) -> dict:
    # legacy rows without market column
    timestamp = row[0] if len(row) > 0 else ""
    symbol = row[1] if len(row) > 1 else ""
    timeframe = row[2] if len(row) > 2 else ""
    side = row[3] if len(row) > 3 else ""
    entry = float(row[4]) if len(row) > 4 and row[4] else ""
    tp = float(row[5]) if len(row) > 5 and row[5] else ""
    sl = float(row[6]) if len(row) > 6 and row[6] else ""
    market = float(row[7]) if len(row) > 7 and row[7] else ""
    actual_price = row[8] if len(row) > 8 else ""
    outcome = row[9] if len(row) > 9 else ""
    entry1_score = _safe_float(row[10]) if len(row) > 10 else ""
    entry2 = _safe_float(row[11]) if len(row) > 11 else ""
    entry2_score = _safe_float(row[12]) if len(row) > 12 else ""
    entry3 = _safe_float(row[13]) if len(row) > 13 else ""
    entry3_score = _safe_float(row[14]) if len(row) > 14 else ""
    # handle old format where market column didn't exist
    if len(row) == 9:
        actual_price = row[7]
        outcome = row[8]
        market = ""
    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "timeframe": timeframe,
        "side": side,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "market": market,
        "actual_price": actual_price,
        "outcome": outcome,
        "entry1_score": entry1_score,
        "entry2": entry2,
        "entry2_score": entry2_score,
        "entry3": entry3,
        "entry3_score": entry3_score,
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
    enforce_symbol_record_limit(out)
    return out

def save_pred_history(preds: list):
    with PRED_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp","symbol","timeframe","side","entry","tp","sl","market","actual_price","outcome",
            "entry1_score","entry2","entry2_score","entry3","entry3_score"
        ])
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
st.set_page_config(
    page_title="Crypto Futures Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
SIDEBAR_WIDTH = "50vw"
SIDEBAR_WIDTH_MOBILE = "100vw"
components.html(
    f"""
    <style>
    [data-testid="stSidebar"] {{
        width: {SIDEBAR_WIDTH} !important;
        min-width: {SIDEBAR_WIDTH} !important;
    }}
    div[data-testid="stAppViewContainer"] > .main {{
        margin-left: {SIDEBAR_WIDTH};
    }}
    @media screen and (max-width: 900px) {{
        [data-testid="stSidebar"] {{
            width: {SIDEBAR_WIDTH_MOBILE} !important;
            min-width: {SIDEBAR_WIDTH_MOBILE} !important;
        }}
        div[data-testid="stAppViewContainer"] > .main {{
            margin-left: {SIDEBAR_WIDTH_MOBILE};
        }}
    }}
    </style>
    """,
    height=0
)

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
cache_cfg = CFG.get("cache", {}) or {}
try:
    DATA_CACHE_TTL = int(cache_cfg.get("data_ttl_seconds", 180))
except (TypeError, ValueError):
    DATA_CACHE_TTL = 180
if DATA_CACHE_TTL < 0:
    DATA_CACHE_TTL = 0

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

st.sidebar.text("Show chart")
chart_visible = st.sidebar.checkbox(
    "", value=False, label_visibility="collapsed", key="chart_visibility"
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

@st.cache_data(show_spinner=False, ttl=DATA_CACHE_TTL)
def fetch_df(symbol_key, timeframe_key, limit: int = 1000):
    data = broker.fetch_ohlcv(symbol_key, timeframe=timeframe_key, limit=limit)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    # ép float để tránh outlier do string/NaN
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms")
    return df

def get_market_price(symbol_key: str, fallback_df: Optional[pd.DataFrame] = None) -> float:
    try:
        return float(broker.fetch_price(symbol_key))
    except Exception:
        pass
    if fallback_df is not None and not fallback_df.empty:
        try:
            return float(fallback_df["close"].iloc[-1])
        except Exception:
            pass
    try:
        df_latest = fetch_df(symbol_key, "1m")
        return float(df_latest["close"].iloc[-1])
    except Exception:
        return float("nan")

def _crosses(lo, hi, lv, tol):  # candle cross detection
    return (lo <= lv*(1+tol)) and (hi >= lv*(1-tol))

def evaluate_all_predictions(pred_list):
    changed = False
    for rec in list(pred_list):
        if rec.get("outcome"): continue

        sym = rec.get("symbol")
        tf_ = rec.get("timeframe") or tf
        side = (rec.get("side") or "").upper()
        try:
            entry = float(rec["entry"])
            tp = float(rec["tp"])
            sl = float(rec["sl"])
        except (TypeError, ValueError):
            continue
        ts = rec.get("timestamp","")

        try:
            ts_ms = int(datetime.strptime(ts,"%Y%m%d %H:%M:%S").timestamp()*1000)
        except:
            continue

        try:
            df_tf = fetch_df(sym, tf_)
        except Exception:
            continue
        df_after = df_tf[df_tf["ts"] >= ts_ms]
        if df_after.empty: continue

        entry_idx = None
        for idx, r in df_after.iloc[1:].iterrows():
            if _crosses(r["low"], r["high"], entry, ENTRY_TOL):
                entry_idx = idx; break
        if entry_idx is None: continue

        outcome = None
        hit_price = None
        half_tp = (entry + tp) / 2.0
        half_sl = (entry + sl) / 2.0

        def _tp_hit(hi, lo):
            return hi >= tp*(1-HIT_TOL) if side == "BUY" else lo <= tp*(1+HIT_TOL)

        def _sl_hit(hi, lo):
            return lo <= sl*(1+HIT_TOL) if side == "BUY" else hi >= sl*(1-HIT_TOL)

        def _tp_half(hi, lo):
            return hi >= half_tp if side == "BUY" else lo <= half_tp

        def _sl_half(hi, lo):
            return lo <= half_sl if side == "BUY" else hi >= half_sl

        for _, r in df_after.loc[entry_idx:].iterrows():
            hi, lo, o = float(r["high"]), float(r["low"]), float(r["open"])
            tp_hit = _tp_hit(hi, lo)
            sl_hit = _sl_hit(hi, lo)
            if tp_hit and sl_hit:
                prefer_tp = abs(o - tp) <= abs(o - sl)
                outcome = "correct" if prefer_tp else "wrong"
                hit_price = tp if prefer_tp else sl
                break
            if tp_hit:
                outcome = "correct"; hit_price = tp; break
            if sl_hit:
                outcome = "wrong"; hit_price = sl; break

            tp_half_hit = _tp_half(hi, lo)
            sl_half_hit = _sl_half(hi, lo)
            if tp_half_hit:
                outcome = "correct"; hit_price = half_tp; break
            if sl_half_hit:
                outcome = "wrong"; hit_price = half_sl; break

        if outcome:
            rec["outcome"] = outcome
            rec["actual_price"] = f"{hit_price:.4f}" if hit_price is not None else ""
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
current_market_price = get_market_price(symbol, df)

# --- Telegram notify for current symbol
tele_cfg_runtime = load_telegram_cfg()
if tele_cfg_runtime.get("enabled"):
    price_now_current = current_market_price
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
            price_now_s = get_market_price(sym, df_sym)
            notify_if_entry_changed(sym, s_side, float(s_entry), float(s_tp), float(s_sl), price_now_s, tele_cfg_runtime)
        except Exception:
            pass

# ---------------------
# Prediction History (persisted + session)
# ---------------------
if "pred_history" not in st.session_state:
    st.session_state.pred_history = load_pred_history()

_ts = datetime.now(SGT).strftime("%Y%m%d %H:%M:%S")
entry_candidates = meta.get("entry_candidates") or []
entry_slots = _build_entry_slots(entry, entry_candidates)
entry1_score = entry_slots[0][1]
entry2_val, entry2_score = entry_slots[1]
entry3_val, entry3_score = entry_slots[2]

def same_pred(last):
    return (
        last["symbol"] == symbol and last["timeframe"] == tf and last["side"] == side and
        abs(last["entry"]-entry)<1e-9 and abs(last["sl"]-sl)<1e-9 and abs(last["tp"]-tp)<1e-9
    )

if st.session_state.pred_history:
    last = st.session_state.pred_history[-1]
    if same_pred(last):
        last["timestamp"] = _ts
        last["market"] = current_market_price
        last["entry1_score"] = entry1_score
        last["entry2"] = entry2_val
        last["entry2_score"] = entry2_score
        last["entry3"] = entry3_val
        last["entry3_score"] = entry3_score
    else:
        _keep_last_symbol_records(
            st.session_state.pred_history,
            symbol,
            MAX_PRED_RECORDS_PER_SYMBOL - 1
        )
        st.session_state.pred_history.append({
            "timestamp":_ts,
            "symbol":symbol,
            "side":side,
            "entry":entry,
            "sl":sl,
            "tp":tp,
            "market": current_market_price,
            "timeframe":tf,
            "actual_price":"",
            "outcome":"",
            "entry1_score": entry1_score,
            "entry2": entry2_val,
            "entry2_score": entry2_score,
            "entry3": entry3_val,
            "entry3_score": entry3_score,
        })
else:
    st.session_state.pred_history.append({
        "timestamp":_ts,
        "symbol":symbol,
        "side":side,
        "entry":entry,
        "sl":sl,
        "tp":tp,
            "market": current_market_price,
        "timeframe":tf,
        "actual_price":"",
        "outcome":"",
        "entry1_score": entry1_score,
        "entry2": entry2_val,
        "entry2_score": entry2_score,
        "entry3": entry3_val,
        "entry3_score": entry3_score,
    })

save_pred_history(st.session_state.pred_history)
st.session_state.pred_history = st.session_state.pred_history[-50:]

# ---------------------
# CHART + LEVEL SUMMARY
# ---------------------
price_for_levels = float(df["close"].iloc[-1])

def _cluster(levels, tol_pct):
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

supports = []
resistance = []
if isinstance(viz_levels,dict):
    sup_raw = [(p,s) for p,s in viz_levels["supports"] if abs(p-price_for_levels)/price_for_levels<=WIN_PCT]
    res_raw = [(p,s) for p,s in viz_levels["resists"] if abs(p-price_for_levels)/price_for_levels<=WIN_PCT]
    sups = sorted(_cluster(sup_raw, TOL_PCT), key=lambda x: (-x[1], abs(x[0]-price_for_levels)))[:POOL_MAX]
    ress = sorted(_cluster(res_raw, TOL_PCT), key=lambda x: (-x[1], abs(x[0]-price_for_levels)))[:POOL_MAX]
    supports = sups[:DRAW_TOP]
    resistance = ress[:DRAW_TOP]

if chart_visible:
    fig = go.Figure()
    fig.add_candlestick(x=df["dt"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price")
    fig.add_bar(x=df["dt"], y=df["volume"], name="Volume", opacity=0.25, yaxis="y2")
    fig.add_scatter(x=df["dt"], y=df["ema_s"], name="EMA short")
    fig.add_scatter(x=df["dt"], y=df["ema_l"], name="EMA long")
    fig.add_scatter(x=df["dt"], y=df["ema_sig"], name="EMA signal")
    fig.add_scatter(x=[df["dt"].iloc[-1]], y=[entry], mode="markers+text",
                    text=[f"{side} @ {entry:.2f}"], textposition="top center")
    fig.add_hline(y=tp, line_color="#16c784", line_dash="dot", annotation_text=f"TP {tp:.2f}")
    fig.add_hline(y=sl, line_color="#ea3943", line_dash="dot", annotation_text=f"SL {sl:.2f}")
    for px,sc in supports:
        fig.add_hline(y=px, line_color="#16c784", opacity=0.6, line_dash="dot", annotation_text=f"SUP {sc:.0f}% @ {px:.2f}")
    for px,sc in resistance:
        fig.add_hline(y=px, line_color="#ea3943", opacity=0.6, line_dash="dot", annotation_text=f"RES {sc:.0f}% @ {px:.2f}")

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

    px_all = np.concatenate([df["low"].values, df["close"].values, df["high"].values]).astype(float)
    lo = float(np.nanpercentile(px_all, 0.5))
    hi = float(np.nanpercentile(px_all, 99.5))
    if hi > lo:
        pad = 0.02 * (hi - lo)
        fig.update_yaxes(range=[lo - pad, hi + pad])

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
else:
    components.html(
        """
        <div style="border-left:4px solid #2d8cff;padding:8px 12px;background:#eef4ff;border-radius:4px;">
            <strong>Chart hidden.</strong> Use the "Show Chart" option in the sidebar to display it.
        </div>
        """,
        height=60
    )

# max_len = max(len(supports), len(resistance))
# components.html('<div style="border-top:2px solid #000000;"></div>', height=16)
# components.html('<div style="margin:5px 0;font-size:16px;font-family:Arial, sans-serif;">Supports / Resistances</div>', height=25)

# for i in range(max_len):
#     sup_txt = ""
#     res_txt = ""

#     if i < len(supports):
#         px, sc = supports[i]
#         sup_txt = f'<span style="color:#16c784;font-family:Arial, sans-serif;font-weight:500;">SUP {sc:.0f}% @ {px:.2f}</span>'

#     if i < len(resistance):
#         px, sc = resistance[i]
#         res_txt = f'<span style="color:#ea3943;font-family:Arial, sans-serif;font-weight:500;">RES {sc:.0f}% @ {px:.2f}</span>'

#     components.html(
#         f"""
#         <div style="
#             display:flex;
#             font-size:14px;
#             width:100%;
#         ">
#             <div>{sup_txt}</div>
#             <div style="margin-left:10px;">{res_txt}</div>
#         </div>
#         """,
#         height=24,
#         scrolling=False
#     )

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
if not hist_df.empty:
    if "market" in hist_df.columns:
        hist_df["market"] = pd.to_numeric(hist_df["market"], errors="coerce")
    def _fmt_entry(row, price_col, score_col):
        price = row.get(price_col, "")
        score = row.get(score_col, "")
        try:
            price_f = float(price)
        except (TypeError, ValueError):
            return ""
        if np.isnan(price_f):
            return ""
        base = f"{price_f:.2f}"
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = ""
        if score_f == "" or np.isnan(score_f):
            return base
        return f"{base} ({score_f:.0f}%)"
    hist_df["entry1"] = hist_df.apply(lambda r: _fmt_entry(r, "entry", "entry1_score"), axis=1)
    hist_df["entry2"] = hist_df.apply(lambda r: _fmt_entry(r, "entry2", "entry2_score"), axis=1)
    hist_df["entry3"] = hist_df.apply(lambda r: _fmt_entry(r, "entry3", "entry3_score"), axis=1)
    display_cols = ["timestamp","symbol","market","side","entry1","entry2","entry3","tp","sl","actual_price","outcome"]
    available_cols = [c for c in display_cols if c in hist_df.columns]
    st.sidebar.dataframe(hist_df[available_cols].tail(50), use_container_width=True)
else:
    st.sidebar.dataframe(hist_df, use_container_width=True)

# evaluate predictions
evaluate_all_predictions(st.session_state.pred_history)

if auto_refresh:
    import time
    time.sleep(refresh_sec)
    st.rerun()
