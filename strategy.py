from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from indicators import ema, rsi

class PyramidTimeframeStrategy:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def detect_bias_h4(self, df_h4: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
        close = df_h4["close"]
        ema21 = ema(close, 21)
        ema55 = ema(close, 55)
        ema200 = ema(close, 200)
        r = rsi(close, 14)

        if ema21.iloc[-1] > ema55.iloc[-1] > ema200.iloc[-1] and r.iloc[-1] > 50:
            bias = "BUY"
        elif ema21.iloc[-1] < ema55.iloc[-1] < ema200.iloc[-1] and r.iloc[-1] < 50:
            bias = "SELL"
        else:
            bias = "BUY" if ema21.iloc[-1] > ema55.iloc[-1] else "SELL"

        meta = {
            "ema21": float(ema21.iloc[-1]),
            "ema55": float(ema55.iloc[-1]),
            "ema200": float(ema200.iloc[-1]),
            "rsi14": float(r.iloc[-1])
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
        bias, meta_h4 = self.detect_bias_h4(df_h4)
        close1 = df_h1["close"]
        e21_1 = ema(close1, 21); e55_1 = ema(close1, 55); r1 = rsi(close1, 14)
        bias_ltf = "BUY" if (e21_1.iloc[-1] > e55_1.iloc[-1] and r1.iloc[-1] > 50) else "SELL"
        side = bias if bias == bias_ltf else bias
        price_now = float(df_15m["close"].iloc[-1])
        sups, ress = self.nearest_levels(df_h1, price_now, k=20)
        if side == "BUY":
            below = [lvl for lvl in sups if lvl[0] <= price_now*1.01]
            pick = sorted(below, key=lambda x: (-x[1], abs(x[0]-price_now)))[:1] or sups[:1]
            entry = float(pick[0][0]) if pick else price_now
            sl = entry * (1 - cfg["strategy"].get("stop_loss_pct",0.8)/100.0)
            tp = entry * (1 + cfg["strategy"].get("take_profit_pct",1.2)/100.0)
        else:
            above = [lvl for lvl in ress if lvl[0] >= price_now*0.99]
            pick = sorted(above, key=lambda x: (-x[1], abs(x[0]-price_now)))[:1] or ress[:1]
            entry = float(pick[0][0]) if pick else price_now
            sl = entry * (1 + cfg["strategy"].get("stop_loss_pct",0.8)/100.0)
            tp = entry * (1 - cfg["strategy"].get("take_profit_pct",1.2)/100.0)
        meta = {"h4_bias": bias, "h1_bias": bias_ltf, **meta_h4}
        viz_levels = {"supports": sups, "resists": ress}
        return side, entry, sl, tp, viz_levels, meta

    def should_flip(self, current_side: str, new_side: str, meta: Dict[str, Any]) -> bool:
        h4 = meta.get("h4_bias","")
        h1 = meta.get("h1_bias","")
        return (new_side != current_side) and (h4 == new_side and h1 == new_side)
