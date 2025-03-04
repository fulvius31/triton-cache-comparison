import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Plot GPU memory usage log.")
parser.add_argument("--title", type=str, default=None, help="Custom title for the plot")
args = parser.parse_args()

log_file = "gpu_usage_log.csv"
df = pd.read_csv(log_file)

plt.style.use('seaborn-v0_8-dark')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'lines.linewidth': 2.5,
    'grid.linewidth': 1,
})

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each session and add a vertical line and annotation at the last point
for session_id, group in df.groupby("session_id"):
    # Plot the line with markers and transparency
    line, = ax.plot(
        group["timestamp"], group["gpu_memory_used"],
        label=session_id,
        marker='o',
        markersize=5,
        alpha=0.8,
        zorder=2
    )
    # Get the last timestamp and GPU memory usage value
    last_timestamp = group["timestamp"].iloc[-1]
    last_value = group["gpu_memory_used"].iloc[-1]
    
    # Plot a dashed vertical line from the x-axis to the last value at the last timestamp
    ax.plot(
        [last_timestamp, last_timestamp],
        [0, last_value],
        '--',
        color=line.get_color(),
        alpha=0.8
    )
    
    # Annotate with an offset so that the text does not overlap with the x-axis.
    ax.annotate(
        f"{last_timestamp:.2f}",
        xy=(last_timestamp, 0),
        xytext=(0, 30),            # 10 points below the x-axis
        textcoords="offset points",
        ha='center',
        va='top',
        rotation=90,
        fontsize=12,
        color=line.get_color(),
        backgroundcolor='white'
    )

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("GPU Memory Used (MB)")

if args.title:
    ax.set_title(args.title)

ax.legend()
ax.grid(True)

fig.tight_layout()
fig.savefig("gpu_memory_usage_comparison.png", dpi=300)
plt.show()