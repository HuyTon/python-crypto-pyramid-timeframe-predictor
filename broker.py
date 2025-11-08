# broker.py — HTTP client thuần cho Spot & USDM Futures với fallback host
from typing import Dict, Any, List, Optional
import httpx

_TF = {
    "1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
    "1h":"1h","2h":"2h","4h":"4h","6h":"6h","8h":"8h","12h":"12h",
    "1d":"1d","3d":"3d","1w":"1w","1M":"1M"
}

# Chính
SPOT_BASE = "https://api.binance.com"
USDM_BASE = "https://fapi.binance.com"
# Testnet
USDM_TEST = "https://testnet.binancefuture.com"
# Mirror public (proxy của Binance Vision) — có cùng path REST
VISION_MIRROR = "https://data-api.binance.vision"

UA = {"User-Agent": "streamlit-app/1.0 (+https://streamlit.io)"}  # giúp tránh một số lớp chặn thô

class Broker:
    """
    - default_market: 'USDM' (Futures) hoặc 'SPOT'
    - mode: 'paper' | 'testnet' | 'live'
    - Tự động fallback host khi gặp 451/403/429/Connect lỗi.
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.mode = (cfg.get("mode","paper") or "paper").lower()
        self.market = (cfg.get("default_market","USDM") or "USDM").upper()
        self.client = httpx.Client(timeout=20, headers=UA, follow_redirects=True)

        # API key/secret chỉ cần nếu bạn mở trade thật (hiện đang paper)
        api = cfg.get("exchange", {}) or {}
        self.api_key = api.get("apiKey","")
        self.api_secret = api.get("secret","")

        # Danh sách host thử theo thứ tự
        self.hosts: List[str] = []
        if self.market == "USDM":
            # Ưu tiên main → testnet → mirror
            self.hosts = [USDM_BASE]
            if self.mode == "testnet":
                self.hosts.insert(0, USDM_TEST)  # testnet lên đầu khi mode=testnet
            self.hosts += [USDM_TEST, VISION_MIRROR]
        else:
            # SPOT: main → mirror
            self.hosts = [SPOT_BASE, VISION_MIRROR]

    def _iv(self, tf: str) -> str:
        return _TF.get(tf, tf)

    def _request(self, path: str, params: Optional[dict] = None):
        """
        Thử tuần tự qua các host; bỏ qua lỗi 451/403/429 và lỗi kết nối để thử host kế tiếp.
        """
        last_err: Optional[Exception] = None
        for base in self.hosts:
            url = f"{base}{path}"
            try:
                r = self.client.get(url, params=params or {})
                # Nếu 2xx => ok
                if 200 <= r.status_code < 300:
                    return r
                # Nếu bị chặn/limit, thử host tiếp theo
                if r.status_code in (451, 403, 429):
                    last_err = httpx.HTTPStatusError(
                        f"{r.status_code} on {url}", request=r.request, response=r
                    )
                    continue
                # Các mã khác: raise luôn
                r.raise_for_status()
                return r
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                last_err = e
                continue
        # Hết host để thử
        if last_err:
            raise last_err
        raise RuntimeError("No hosts configured for request")

    def fetch_ohlcv(self, symbol: str, timeframe: str="1m", limit: int=500) -> List[List[float]]:
        iv = self._iv(timeframe)
        if self.market == "USDM":
            # chung path cho main/testnet/mirror
            path = "/fapi/v1/klines"
        else:
            path = "/api/v3/klines"

        r = self._request(path, params={"symbol": symbol, "interval": iv, "limit": limit})
        kl = r.json()
        out = []
        for k in kl:
            # k: [Open time, Open, High, Low, Close, Volume, Close time, ...]
            ts = int(k[0])
            o = float(k[1]); h = float(k[2]); l = float(k[3]); c = float(k[4]); v = float(k[5])
            out.append([ts,o,h,l,c,v])
        return out

    def fetch_tickers(self):
        if self.market == "USDM":
            path = "/fapi/v1/ticker/price"
        else:
            path = "/api/v3/ticker/price"
        r = self._request(path)
        data = r.json()
        if isinstance(data, list):
            return {row["symbol"]: float(row["price"]) for row in data}
        return {data["symbol"]: float(data["price"])}

    def place_order(self, symbol: str, side: str, amount: float, price=None, params=None):
        """
        Paper: giả lập fill ở giá last. Không gọi endpoint trade thật trong bản này.
        """
        side = side.upper()
        last = self.fetch_tickers().get(symbol)
        if last is None:
            # fallback single ticker
            if self.market == "USDM":
                path = "/fapi/v1/ticker/price"
            else:
                path = "/api/v3/ticker/price"
            r = self._request(path, params={"symbol": symbol})
            last = float(r.json()["price"])
        fill_price = price or last
        rec = {"status":"filled","side":side,"price":fill_price,"amount":amount,"symbol":symbol,"paper":True}
        return rec
