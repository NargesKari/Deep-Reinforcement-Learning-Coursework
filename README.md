# Deep-Reinforcement-Learning-Coursework

Five homework sets covering the standard arc of deep RL, from value-based methods through actor-critic and model-based planning to contextual bandits, each with a written report, a runnable notebook, and recorded rollout videos of the trained policies.

## What's in here

- **HW1 - Value-based methods**: DQN, Double DQN, Dueling DQN, Prioritized Experience Replay DQN, and a "HalfRainbow" combining several of these extensions, compared against a random-rollout baseline with recorded videos for each variant.
- **HW2 - Policy-based methods**: REINFORCE and Natural Policy Gradient, with an ablation on variance-reduction choices (no baseline, constant baseline, learned baseline, with/without reward-to-go causality) on CartPole.
- **HW3 - Actor-critic methods**: DPG, DDPG, and TD3, trained on a custom Gymnasium environment (`moving_target_reacher_env.py`, ~1200 lines) where the target follows a configurable moving trajectory rather than sitting still.
- **HW4 - Model-based RL**: Dyna-Q with prioritized sweeping, and MCTS/MuZero-style planning.
- **HW5 - Multi-armed bandits**: contextual bandits applied to a real text-classification task (hate-speech/offensive-language detection on the TweetEval dataset) rather than a synthetic bandit setup.

Each homework folder has `doc/` (assignment + reference papers), `practical/` (the notebook(s) and result videos), and a `report.pdf`.

## Tech stack

Python, PyTorch, Gymnasium (including a custom environment for HW3), Jupyter notebooks, NumPy/matplotlib.

## Running it

Each homework's practical notebook is self-contained:

```bash
jupyter notebook "HOMEWORK 1_VALUE-BASED METHODS/practical/HalfRainbowDQN.ipynb"
jupyter notebook "HOMEWORK 2_POLICY-BASED METHODS/practical/HW2_REINFORCE.ipynb"
jupyter notebook "HOMEWORK 3_ACTOR-CRITIC METHODS/practical/HW3_DPGs.ipynb"
jupyter notebook "HOMEWORK 4_MODEL-BASED RL/practical/HW4_dynaq.ipynb"
jupyter notebook "HOMEWORK 5_MULTI-ARMED BANDITS/practical/Ctx_Bandits.ipynb"
```

HW3 additionally requires `moving_target_reacher_env.py` on the path (it lives alongside the notebook) and `gymnasium[mujoco]`.
