import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Plot GPU memory usage log.")
parser.add_argument("--title", type=str, default=None, help="Custom title for the plot")
args = parser.parse_args()

log_file = "gpu_usage_log.csv"
df = pd.read_csv(log_file)

plt.figure(figsize=(10, 6))
for session_id, group in df.groupby("session_id"):
    plt.plot(group["timestamp"], group["gpu_memory_used"], label=session_id)

plt.xlabel("Time (seconds)")
plt.ylabel("GPU Memory")

if args.title:
    plt.title(args.title)

plt.legend()
plt.grid(True)

plt.savefig("gpu_memory_usage_comparison.png")
plt.show()