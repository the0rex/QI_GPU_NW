#!/usr/bin/env python3
import subprocess
import time
import psutil
import json
from pathlib import Path

REF = "genomes/human_chr1.fa"
QRY = "genomes/chimp_chr1.fa"
OUT = "bench_output"
CONFIG = "config/human-chimp.yaml"

def get_gpu_usage():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        return mem.used/1024**2, util.gpu
    except:
        return None, None

start_time = time.time()

cmd = [
    "qi-align",
    "--ref", REF,
    "--qry", QRY,
    "--out", OUT,
    "--config", CONFIG
]

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

max_ram = 0
while proc.poll() is None:
    ram = psutil.Process(proc.pid).memory_info().rss / (1024**2)
    max_ram = max(max_ram, ram)
    time.sleep(1)

total_time = time.time() - start_time

stats = json.load(open(Path(OUT)/"stats.json"))

print("======== Benchmark Finished ========")
print(f"Total time: {total_time/60:.2f} minutes")
print(f"Max RAM: {max_ram:.1f} MB")
print("Identity:", stats["identity"])
print("Divergence:", stats["divergence"])
