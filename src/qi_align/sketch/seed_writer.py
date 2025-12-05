# ================================================================
# seed_writer.py
# Write seeds to disk (for huge genomes)
# ================================================================
import struct

class SeedWriter:
    """
    Write (hash, pos_ref, pos_qry) seed entries to binary file.
    """
    def __init__(self, path):
        self.f = open(path, "wb")

    def add(self, h, rpos, qpos):
        self.f.write(struct.pack("<QQQ", h, rpos, qpos))

    def close(self):
        self.f.close()
