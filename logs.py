import os, csv, json
from typing import List, Dict, Any

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def log_prediction(record: Dict[str, Any], basename="predictions"):
    csv_path = os.path.join(LOG_DIR, f"{basename}.csv")
    txt_path = os.path.join(LOG_DIR, f"{basename}.txt")
    header = ["timestamp","symbol","market","side","entry","sl","tp","timeframe","actual_price","outcome"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header: w.writeheader()
        w.writerow({k: record.get(k, "") for k in header})
    with open(txt_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return csv_path, txt_path

def import_logs(csv_file: str) -> List[Dict[str, Any]]:
    out = []
    with open(csv_file, newline="") as f:
        r = csv.DictReader(f)
        for row in r: out.append(row)
    return out
