#!/bin/bash
echo "=========================================="
echo "CarRacing V6 Final Study Progress"
echo "=========================================="
echo "Time: $(date)"
echo ""

# Active processes
active=$(ps aux | grep "V6_final_comparison" | grep "python -m objectrl" | grep -v grep | wc -l)
echo "🔄 Active processes: $active"
echo ""

# Check each algorithm
for algo in ppo sac td3; do
  echo "${algo^^}:"
  completed=0
  total=5

  for seed in 1 1234 22 3407 42; do
    eval_file="_logs/V6_final_comparison/car-racing/$algo/seed_$seed/*/eval_results.npy"

    if ls $eval_file 2>/dev/null | head -1 | grep -q .; then
      result=$(python3 -c "
import numpy as np
import glob
import os
files = glob.glob('$eval_file')
if files:
  latest = max(files, key=os.path.getmtime)
  d = np.load(latest, allow_pickle=True).item()
  s = sorted(d.keys())
  if s:
    latest_step = s[-1]
    progress = (latest_step / 120000) * 100
    reward = d[latest_step].mean()
    print(f'{latest_step:7} steps ({progress:5.1f}%) = {reward:7.2f}')
  else:
    print('No data')
" 2>/dev/null)
      echo "  Seed $seed: $result"
      if echo "$result" | grep -q "120000"; then
        completed=$((completed + 1))
      fi
    else
      echo "  Seed $seed: Not started"
    fi
  done

  echo "  Progress: $completed/$total seeds completed"
  echo ""
done

echo "=========================================="
echo "Logs: scripts/run_*_car-racing_seed_*_V6.log"
echo "Results: _logs/V6_final_comparison/"
echo "=========================================="
