import time
import ccxt
from typing import Dict, Any

class Broker:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.mode = cfg.get("mode", "paper")
        market = cfg.get("default_market","USDM").upper()
        self.is_usdm = market == "USDM"
        self._init_exchange(cfg)
        self.paper_positions = {}

    def _init_exchange(self, cfg):
        params = {"enableRateLimit": True}
        if self.is_usdm:
            ex = ccxt.binanceusdm(params)
        else:
            ex = ccxt.binance(params)
        if self.mode in ["testnet","paper"]:
            ex.set_sandbox_mode(True)
        api = cfg.get("exchange",{})
        if api.get("apiKey"): ex.apiKey = api["apiKey"]
        if api.get("secret"): ex.secret = api["secret"]
        if api.get("password"): ex.password = api["password"]
        self.ex = ex

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 500):
        return self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_tickers(self):
        return self.ex.fetch_tickers()

    def place_order(self, symbol: str, side: str, amount: float, price=None, params=None):
        params = params or {}
        side = side.lower()
        if self.mode == "paper":
            last = self.ex.fetch_ticker(symbol)["last"]
            fill_price = price or last
            key = (symbol, side)
            self.paper_positions.setdefault(key, []).append({"price": fill_price, "amount": amount, "ts": time.time()})
            return {"status":"filled","side":side,"price":fill_price,"amount":amount,"symbol":symbol,"paper":True}
        else:
            order_type = "market" if price is None else "limit"
            return self.ex.create_order(symbol, order_type, side, amount, price, params or {})
