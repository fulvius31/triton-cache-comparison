import pandas as pd
import matplotlib.pyplot as plt

log_file = "gpu_usage_log.csv"
df = pd.read_csv(log_file)

plt.figure(figsize=(10, 6))
for session_id, group in df.groupby("session_id"):
    plt.plot(group["timestamp"], group["gpu_memory_used"], label=session_id)

plt.xlabel("Time (seconds)")
plt.ylabel("GPU Memory Used (MB)")
plt.title("GPU Memory Usage and startup time on vllm + granite-3.0 + Triton flash attention (w/ Triton cache vs w/o Triton cache)")
plt.legend()
plt.grid(True)

plt.savefig("gpu_memory_usage_comparison.png")
plt.show()
