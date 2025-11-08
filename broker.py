# broker.py — HTTP client thuần cho Spot & USDM Futures (OHLCV + ticker + paper order)
from typing import Dict, Any, List
import httpx

_TF = {
    "1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
    "1h":"1h","2h":"2h","4h":"4h","6h":"6h","8h":"8h","12h":"12h",
    "1d":"1d","3d":"3d","1w":"1w","1M":"1M"
}

SPOT_BASE = "https://api.binance.com"
USDM_BASE = "https://fapi.binance.com"
USDM_TEST = "https://testnet.binancefuture.com"

class Broker:
    """
    - default_market: 'USDM' (Futures) hoặc 'SPOT'
    - mode: 'paper' | 'testnet' | 'live'
    Lưu ý: place_order hiện đang giả lập (paper). Nếu muốn order thật, bổ sung HMAC theo API docs.
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.mode = (cfg.get("mode","paper") or "paper").lower()
        self.market = (cfg.get("default_market","USDM") or "USDM").upper()
        self.paper_positions = []
        self.client = httpx.Client(timeout=15)

        api = cfg.get("exchange", {}) or {}
        self.api_key = api.get("apiKey","")
        self.api_secret = api.get("secret","")

        if self.market == "USDM":
            self.base = USDM_TEST if self.mode == "testnet" else USDM_BASE
        else:
            self.base = SPOT_BASE

    def _iv(self, tf: str) -> str:
        return _TF.get(tf, tf)

    def fetch_ohlcv(self, symbol: str, timeframe: str="1m", limit: int=500) -> List[List[float]]:
        iv = self._iv(timeframe)
        url = f"{self.base}/fapi/v1/klines" if self.market=="USDM" else f"{self.base}/api/v3/klines"
        r = self.client.get(url, params={"symbol": symbol, "interval": iv, "limit": limit})
        r.raise_for_status()
        kl = r.json()
        out = []
        for k in kl:
            # k: [Open time, Open, High, Low, Close, Volume, Close time, ...]
            ts = int(k[0])
            o = float(k[1]); h = float(k[2]); l = float(k[3]); c = float(k[4]); v = float(k[5])
            out.append([ts,o,h,l,c,v])
        return out

    def fetch_tickers(self):
        url = f"{self.base}/fapi/v1/ticker/price" if self.market=="USDM" else f"{self.base}/api/v3/ticker/price"
        r = self.client.get(url)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return {row["symbol"]: float(row["price"]) for row in data}
        return {data["symbol"]: float(data["price"])}

    def place_order(self, symbol: str, side: str, amount: float, price=None, params=None):
        # Paper: giả lập fill tại last price
        side = side.upper()
        tickers = self.fetch_tickers()
        last = tickers.get(symbol)
        if last is None:
            url = f"{self.base}/fapi/v1/ticker/price" if self.market=="USDM" else f"{self.base}/api/v3/ticker/price"
            r = self.client.get(url, params={"symbol": symbol})
            r.raise_for_status()
            last = float(r.json()["price"])
        fill_price = price or last
        rec = {"status":"filled","side":side,"price":fill_price,"amount":amount,"symbol":symbol,"paper":True}
        self.paper_positions.append(rec)
        return rec
