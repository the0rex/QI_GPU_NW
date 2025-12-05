# ================================================================
# disk_array.py
# Disk-backed append-only storage for large chunk lists
# ================================================================
import tempfile
import pickle

class DiskArray:
    """
    Append objects to disk (via pickle).
    Later iterates through all objects.
    More memory-efficient than keeping list in RAM.
    """
    def __init__(self):
        self.tmp = tempfile.NamedTemporaryFile(delete=False)
        self.count = 0

    def append(self, obj):
        pickle.dump(obj, self.tmp)
        self.count += 1

    def __iter__(self):
        self.tmp.seek(0)
        for _ in range(self.count):
            yield pickle.load(self.tmp)
