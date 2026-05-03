import numpy as np, pathlib as p

env = "acrobot-swingup-v0"
algo = "td3"
seeds = ["01", "22", "42", "1234", "3407"]

for seed in seeds:
    run = p.Path(sorted(p.Path(f"../_logs/{env}/{algo}/seed_{seed}").glob("*"))[-1])
    print(f"\n\nseed: {seed}")
    print("run:", run)

    eval_dict = np.load(run / "eval_results.npy", allow_pickle=True).item()

    print("eval steps:", sorted(eval_dict.keys()))

    for k in sorted(eval_dict):
        v = eval_dict[k]
        print(f"step={k} mean={float(v.mean()):.3f} std={float(v.std()):.3f}")
    overall_mean = float(np.concatenate(list(eval_dict.values())).mean())
    overall_std = float(np.concatenate(list(eval_dict.values())).std())
    print(f"overall mean: {overall_mean:.3f}")
    print(f"overall std: {overall_std:.3f}")


    if (run / "episode_rewards.npy").exists():
        ep = np.load(run / "episode_rewards.npy", allow_pickle=True)
        print("episodes:", ep.shape[0], "last reward:", float(ep[-1]))
