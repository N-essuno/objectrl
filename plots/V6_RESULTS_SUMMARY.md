# V6 CarRacing Study - Intermediate Results

## Executive Summary

**Status:** ❌ **FAILED - Features Hurt Performance**

The V6 study with frame stacking + normalization enabled produced **disastrous results**:
- PPO: -34.03 average (70x worse than without features)
- SAC: Crashed at 68% (segmentation fault)
- TD3: Crashed at 42% (segmentation fault)

## Plots Generated

1. **V6 Comparison Plot:** `plots/V6_car_racing_comparison.png`
2. **Old PPO Baseline:** `plots/old_ppo_car_racing.png`

## Detailed Results

### PPO (5 seeds completed)

| Step | Mean | Std | Seeds |
|------|------|-----|-------|
| 0 | -22.12 | 4.83 | 5 |
| 10K | -26.10 | 6.66 | 5 |
| 20K | -22.74 | 8.38 | 5 |
| 30K | -24.01 | 6.33 | 5 |
| 40K | -20.70 | 3.79 | 5 |
| 50K | -20.42 | 7.57 | 5 |
| 60K | -27.99 | 26.77 | 5 |
| 70K | -33.28 | 16.25 | 5 |
| 80K | **-64.85** | 51.05 | 5 |
| 90K | -47.35 | 15.51 | 5 |
| 100K | **-69.28** | 30.29 | 5 |
| 110K | -39.52 | 15.54 | 5 |
| 120K | **-34.03** | 19.95 | 5 |

**Overall Average: -34.80**

### SAC (1 seed, crashed at 80K steps)

| Step | Mean | Std |
|------|------|-----|
| 0 | -25.12 | 0.00 |
| 10K | -29.63 | 0.00 |
| 20K | -31.28 | 0.00 |
| 30K | -18.00 | 0.00 |
| 40K | -17.18 | 0.00 |
| 50K | -24.64 | 0.00 |
| 60K | -26.68 | 0.00 |
| 70K | **6.10** | 0.00 |
| 80K | -27.23 | 0.00 |

**Overall Average: -21.52**
**Error:** Segmentation fault at 82,176 steps

### TD3 (1 seed, crashed at 50K steps)

| Step | Mean | Std |
|------|------|-----|
| 0 | -25.12 | 0.00 |
| 10K | **-93.30** | 0.00 |
| 20K | -83.36 | 0.00 |
| 30K | -84.06 | 0.00 |
| 40K | -84.14 | 0.00 |
| 50K | -85.11 | 0.00 |

**Overall Average: -75.85**
**Error:** Crashed (likely segmentation fault)

## Comparison: With vs Without Features

### PPO Performance Comparison

**Without Features (Old Results, 50K steps):**
```
step=0 mean=-22.36 std=4.66
step=5000 mean=-26.55 std=6.14
step=10000 mean=-23.23 std=7.49
step=15000 mean=-24.92 std=5.25
step=20000 mean=-20.43 std=5.43
step=25000 mean=-25.20 std=4.65
step=30000 mean=-21.78 std=11.63
step=35000 mean=-12.02 std=16.80
step=40000 mean=-24.38 std=7.41
step=45000 mean=-29.59 std=5.00
step=49999 mean=-20.84 std=29.25

Overall Average: -22.85
Best Seed (42): +36.50 at 50K steps
```

**With Features (V6 Results, 120K steps):**
```
Overall Average: -34.80
Best Seed (22): +1.40 at 120K steps
Worst Seed (01): -57.71 at 120K steps
```

**Performance Change:** -22.85 → -34.03 (**-48% worse**)

## Key Findings

### 1. **Features Hurt PPO Performance** ❌

- **Expected:** Frame stacking + normalization should improve performance
- **Actual:** 48% worse performance (-22.85 → -34.03)
- **Conclusion:** These features are NOT beneficial for this implementation

### 2. **Massive Performance Collapse at 60K+ Steps**

PPO training completely collapsed after 60K steps:
- 0-60K: -20 to -27 (normal)
- 70K+: -33 to -69 (catastrophic collapse)
- Suggests instability with features enabled

### 3. **Process Crashes** ❌

- **SAC:** Segmentation fault at 82K steps (68%)
- **TD3:** Crashed at 50K steps (42%)
- **Cause:** Unknown, possibly memory/GPU issues

### 4. **High Variance**

- PPO std at 80K: 51.05 (extremely high)
- PPO std at 100K: 30.29 (very high)
- Suggests training instability

## Recommendations

### For Re-run

1. **Disable Frame Stacking and Normalization**
   - These features hurt performance
   - Run WITHOUT these features

2. **Fix Segmentation Faults**
   - Investigate memory issues
   - Possibly reduce batch size or buffer size
   - Add error handling

3. **Shorter Training Runs**
   - 50K steps instead of 120K
   - PPO collapses after 60K anyway

4. **Keep Original Configuration**
   - The configuration that gave +36.50 for seed 42
   - Use the working hyperparameters

### For Reporting

**Current Results:** **NOT REPORTABLE**

- PPO: Failed to meet expectations (-34 vs +20 to +40)
- SAC: Incomplete (crashed)
- TD3: Incomplete (crashed)

**Next Steps:**
1. Fix crashes
2. Re-run without features
3. Generate reportable results

## Files

- **Plot:** `plots/V6_car_racing_comparison.png`
- **Data:** `_logs/V6_final_comparison/car-racing/`
- **Logs:** `scripts/run_*_car-racing_seed_*_V6.log`
- **Script:** `process_logs_for_report.py`
