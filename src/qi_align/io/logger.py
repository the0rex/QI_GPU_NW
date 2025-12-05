# ================================================================
# logger.py
# Simple logger with timestamps
# ================================================================
from datetime import datetime
import sys

def log(msg):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sys.stdout.write(f"[{t}] {msg}\n")
    sys.stdout.flush()
