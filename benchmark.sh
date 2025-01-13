#!/bin/bash

# Use Triton Flash attention with custom vLLM https://github.com/cmagina/vllm/tree/triton
export VLLM_ATTENTION_BACKEND=TRITON_FLASH

LOG_FILE="gpu_usage_log.csv"
echo "session_id,timestamp,gpu_memory_used" > $LOG_FILE

log_usage() {
	local SESSION_ID=$1
	local START=$(date +%s)
	
	while true; do
		CURRENT_TIME=$(date +%s)
		
		ELAPSED_TIME=$((CURRENT_TIME - START))
		
		GPU_MEMORY_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
		echo "$SESSION_ID,$ELAPSED_TIME,$GPU_MEMORY_USED" >> $LOG_FILE
		sleep 1
	done
}

log_usage "no-cache" &
LOG_PID1=$!

# Remove the previous cache
echo "Removing Triton cache..."
rm -rf ~/.triton/cache

python ./scripts/flash_test.py

kill $LOG_PID1

log_usage "cache" &
LOG_PID2=$!

python ./scripts/flash_test.py

kill $LOG_PID2

echo "GPU memory usage data saved in $LOG_FILE"
echo "Saving the plot..."

python ./scripts/plot_benchmark.py
