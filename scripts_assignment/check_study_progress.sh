#!/bin/bash
echo "=========================================="
echo "CarRacing Study Progress Check"
echo "=========================================="
echo "Time: $(date)"
echo ""

# Active processes
active=$(ps aux | grep "objectrl.*main.*--model.name" | grep -v grep | wc -l)
echo "🔄 Active processes: $active"
echo ""

# Check each algorithm
for algo in ppo sac td3; do
  echo "${algo^^}:"
  completed=0
  total=5

  for seed in 1 1234 22 3407 42; do
    eval_file="_logs/car-racing/$algo/seed_$seed/*/eval_results.npy"

    if [ -f "$eval_file" ]; then
      result=$(python3 -c "
import numpy as np
d = np.load('$eval_file', allow_pickle=True).item()
s = sorted(d.keys())
if s:
  latest = s[-1]
  progress = (latest / 20000) * 100
  print(f'  Seed {seed:4}: {latest:6} steps ({progress:5.1f}%) = {d[latest].mean():7.2f}', flush=True)
  if [ "$latest" -ge "20000" ]; then
    completed=$((completed + 1))
  fi
" 2>/dev/null)
      echo "$result" | head -1
    fi
  done

  echo "  Progress: $completed/$total seeds completed"
  echo ""
done

echo "=========================================="
echo "Logs: scripts/run_*_seed_*_comparison.log"
echo "Plot will be: plots/comparison_car_racing.png"
echo "=========================================="
