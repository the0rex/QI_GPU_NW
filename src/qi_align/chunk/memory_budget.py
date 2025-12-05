# ================================================================
# memory_budget.py
# Estimate RAM usage of chunk generation and enforce limits
# ================================================================
import psutil

class MemoryBudget:
    """
    Tracks RAM usage and enforces a soft upper limit.
    """
    def __init__(self, max_gb=4):
        self.max_gb = max_gb
        self.process = psutil.Process()
        self.warned = False

    def record_chunk(self):
        """
        Called after writing each chunk. Warn the user if approaching limit.
        """
        mem = self.process.memory_info().rss / (1024**3)
        if mem > self.max_gb and not self.warned:
            print(f"[WARN] RAM usage exceeded {self.max_gb} GB during chunking.")
            self.warned = True
