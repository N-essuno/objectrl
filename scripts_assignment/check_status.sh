#!/bin/bash
echo "=== CarRacing Experiment Status ==="
echo "Time: $(date)"
echo ""

echo "🔄 Running Processes:"
ps aux | grep "objectrl.*car-racing" | grep -v grep | wc -l
echo ""

echo "📊 Latest Results:"
for algo in sac td3 ppo; do
  echo "$algo:"
  find "_logs/car-racing/$algo" -name "eval_results.npy" -printf "%T@ %p\n" | sort -rn | head -3 | cut -d' ' -f2 | while read file; do
    seed=$(echo "$file" | cut -d'/' -f4)
    result=$(python3 -c "import numpy as np; d=np.load('$file', allow_pickle=True).item(); s=sorted(d.keys()); print(f'{len(s)} evals, {s[-1]} steps = {d[s[-1]].mean():.2f} ± {d[s[-1]].std():.2f}') if s else 'No data'")
    echo "  $seed: $result"
  done
  echo ""
done

echo "📝 Recent Log Activity:"
for algo in sac td3 ppo; do
  latest_log=$(find "_logs/car-racing/$algo" -name "log.log" -printf "%T@ %p\n" | sort -rn | head -1 | cut -d' ' -f2)
  if [ -n "$latest_log" ]; then
    echo "$algo: $(grep 'EVALUATION' "$latest_log" | tail -1)"
  fi
done
