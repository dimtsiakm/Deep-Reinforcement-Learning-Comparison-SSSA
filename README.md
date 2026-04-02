# Deep Reinforcement Learning Comparison

Comparison of PPO, SAC, and TD3 on MuJoCo Gymnasium tasks (default: `HalfCheetah-v5`).

Neural Networks & Deep Learning - 2025  
Scuola Superiore Sant'Anna, Pisa  
Instructor: Prof. Giorgio Buttazzo

Dimitrios Tsiakmakis  
PhD Candidate, Biorobotics Institute  
dimitrios.tsiakmakis@santannapisa.it

## Run

```bash
python main.py --env HalfCheetah-v5 --timesteps 1000 --eval_episodes 10 --seeds 42 123 456 --algos PPO SAC TD3 --output results
```

## Output

- `results/summary.csv`
- `results/detailed_results.csv`
- `results/statistical_tests.txt`
- plots (`mean_reward.png`, `train_time.png`, `learning_curves.png`, `reward_distribution.png`)
- per-algorithm/per-seed models and videos