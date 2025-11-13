from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from indicators import ema, rsi

class PyramidTimeframeStrategy:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def _bias_from_series(
        self,
        close: pd.Series,
        fast: int,
        slow: int,
        signal: int,
        rsi_period: int,
    ) -> Tuple[str, float]:
        ema_fast = ema(close, fast)
        ema_slow = ema(close, slow)
        ema_sig = ema(close, signal)
        r = rsi(close, rsi_period)

        slope_window = min(5, len(ema_fast) - 1)
        slope = 0.0
        if slope_window > 0:
            slope = float(ema_fast.iloc[-1] - ema_fast.iloc[-slope_window - 1])

        score = 0.0
        score += 1.0 if ema_fast.iloc[-1] > ema_slow.iloc[-1] else -1.0
        score += 0.5 if ema_slow.iloc[-1] > ema_sig.iloc[-1] else -0.5
        if slope != 0:
            slope_pct = slope / max(1e-9, close.iloc[-1]) * 1000.0
            score += np.clip(slope_pct, -1.0, 1.0) * 0.5
        if r.iloc[-1] > 55:
            score += 0.5
        elif r.iloc[-1] < 45:
            score -= 0.5

        if abs(score) < 0.25:
            return "NEUTRAL", float(score)
        return ("BUY" if score > 0 else "SELL"), float(score)

    def detect_bias_h4(self, df_h4: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
        close = df_h4["close"]
        ema21 = ema(close, 21)
        ema55 = ema(close, 55)
        ema200 = ema(close, 200)
        r = rsi(close, 14)
        bias, bias_score = self._bias_from_series(close, 21, 55, 200, 14)

        meta = {
            "ema21": float(ema21.iloc[-1]),
            "ema55": float(ema55.iloc[-1]),
            "ema200": float(ema200.iloc[-1]),
            "rsi14": float(r.iloc[-1]),
            "bias_score": float(bias_score),
        }
        return bias, meta

    def _swing_points(self, df: pd.DataFrame, lookback: int = 3):
        highs = df["high"].values
        lows = df["low"].values
        res_points, sup_points = [], []
        for i in range(lookback, len(df)-lookback):
            if highs[i] == max(highs[i-lookback:i+lookback+1]):
                res_points.append((i, highs[i]))
            if lows[i] == min(lows[i-lookback:i+lookback+1]):
                sup_points.append((i, lows[i]))
        return res_points, sup_points

    def _score_levels(self, df: pd.DataFrame, levels: List[Tuple[int,float]]) -> List[Tuple[float,float]]:
        close = df["close"]
        ema21 = ema(close, 21); ema55 = ema(close, 55); ema200 = ema(close, 200)
        last_idx = len(df)-1
        out = []
        for idx, px in levels:
            band = px * 0.0015
            touch = ((df["high"] >= px - band) & (df["low"] <= px + band)).sum()
            touch_score = min(40, int(touch) * 5)
            recency = max(0.0, 1 - (last_idx - idx)/500.0)
            recency_score = recency * 25.0
            ema_dist = min(abs(px-ema21.iloc[-1]), abs(px-ema55.iloc[-1]), abs(px-ema200.iloc[-1]))
            conf_score = max(0.0, 20.0 - (ema_dist/px)*10000.0)
            rn = round(px/50)*50
            rn_score = max(0.0, 15.0 - abs(px-rn)/px*10000.0)
            score = float(np.clip(touch_score + recency_score + conf_score + rn_score, 0, 100))
            out.append((float(px), score))
        return out

    def nearest_levels(self, df: pd.DataFrame, current_price: float, k: int = 12):
        res_points, sup_points = self._swing_points(df, lookback=3)
        res_scored = self._score_levels(df, res_points)
        sup_scored = self._score_levels(df, sup_points)
        res_sorted = sorted(res_scored, key=lambda x: abs(x[0]-current_price))[:k]
        sup_sorted = sorted(sup_scored, key=lambda x: abs(x[0]-current_price))[:k]
        return sup_sorted, res_sorted

    def propose_trade(self, df_h4: pd.DataFrame, df_h1: pd.DataFrame, df_15m: pd.DataFrame, cfg: Dict[str, Any]):
        bias_h4, meta_h4 = self.detect_bias_h4(df_h4)
        strat_cfg = cfg.get("strategy", {})
        fast = int(strat_cfg.get("ema_short", 21))
        slow = int(strat_cfg.get("ema_long", 55))
        signal = int(strat_cfg.get("ema_signal", 100))
        rsi_period = int(strat_cfg.get("rsi_period", 14))

        bias_h1, score_h1 = self._bias_from_series(df_h1["close"], fast, slow, signal, rsi_period)
        fast_m15 = max(5, fast // 2)
        slow_m15 = max(fast, slow)
        signal_m15 = max(signal // 2, slow_m15)
        bias_m15, score_m15 = self._bias_from_series(df_15m["close"], fast_m15, slow_m15, max(signal_m15, slow_m15 + 5), max(9, rsi_period // 2))

        score_h4 = meta_h4.get("bias_score", 0.0)

        def _weighted(score: float, weight: float) -> float:
            return np.sign(score) * min(abs(score), 2.0) * weight

        total_score = sum([
            _weighted(score_h4 if bias_h4 != "NEUTRAL" else 0.0, 2.5),
            _weighted(score_h1 if bias_h1 != "NEUTRAL" else 0.0, 1.5),
            _weighted(score_m15 if bias_m15 != "NEUTRAL" else 0.0, 1.0),
        ])

        price_now = float(df_15m["close"].iloc[-1])
        if abs(total_score) < 0.3:
            ema_fast_15 = ema(df_15m["close"], fast_m15)
            side = "BUY" if price_now >= float(ema_fast_15.iloc[-1]) else "SELL"
        else:
            side = "BUY" if total_score > 0 else "SELL"

        sups, ress = self.nearest_levels(df_h1, price_now, k=20)
        tol_cfg = cfg.get("tolerances", {})
        entry_buffer = max(0.0001, float(tol_cfg.get("entry_pct", 0.03)) / 100.0)
        lvl_cfg = cfg.get("levels", {})
        window_pct = max(entry_buffer * 2, float(lvl_cfg.get("window_pct", 8)) / 100.0)

        if side == "BUY":
            upper_entry = price_now * (1 - entry_buffer)
            lower_entry = price_now * (1 - window_pct)
            below = [lvl for lvl in sups if lvl[0] <= upper_entry]
            pick = sorted(below, key=lambda x: (-x[1], abs(x[0]-price_now)))[:1] or sups[:1]
            entry = float(pick[0][0]) if pick else price_now * (1 - entry_buffer)
            entry = float(np.clip(entry, lower_entry, upper_entry))
            sl = entry * (1 - strat_cfg.get("stop_loss_pct", 0.8)/100.0)
            tp = entry * (1 + strat_cfg.get("take_profit_pct", 1.2)/100.0)
        else:
            lower_entry = price_now * (1 + entry_buffer)
            upper_entry = price_now * (1 + window_pct)
            above = [lvl for lvl in ress if lvl[0] >= lower_entry]
            pick = sorted(above, key=lambda x: (-x[1], abs(x[0]-price_now)))[:1] or ress[:1]
            entry = float(pick[0][0]) if pick else price_now * (1 + entry_buffer)
            entry = float(np.clip(entry, lower_entry, upper_entry))
            sl = entry * (1 + strat_cfg.get("stop_loss_pct", 0.8)/100.0)
            tp = entry * (1 - strat_cfg.get("take_profit_pct", 1.2)/100.0)

        meta = {
            "h4_bias": bias_h4,
            "h1_bias": bias_h1,
            "m15_bias": bias_m15,
            "h4_score": float(score_h4),
            "h1_score": float(score_h1),
            "m15_score": float(score_m15),
            "combined_score": float(total_score),
            **meta_h4
        }
        viz_levels = {"supports": sups, "resists": ress}
        return side, entry, sl, tp, viz_levels, meta

    def should_flip(self, current_side: str, new_side: str, meta: Dict[str, Any]) -> bool:
        h4 = meta.get("h4_bias","")
        h1 = meta.get("h1_bias","")
        return (new_side != current_side) and (h4 == new_side and h1 == new_side)
