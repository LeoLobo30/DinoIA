# DinoIA

DinoIA is a small reinforcement-learning project for Chrome Dino-style control in a simulated Gymnasium environment.

The active path is DQN with a 13-value vector state. There is no real browser control, computer vision, PPO, or NEAT flow in this version.

## Setup

Use the project virtual environment, then install dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Check the environment:

```powershell
python -m dinoia doctor
```

## Train

```powershell
python -m dinoia train-sim-dqn --total-timesteps 250000
```

The agent uses `stable_baselines3.DQN("MlpPolicy", DinoEnv, ...)` with vector observations and three discrete actions:

- `0`: noop
- `1`: jump
- `2`: duck

Artifacts are saved under `artifacts/dqn/`:

- `sim_model.zip`
- `best_sim_model.zip`
- `training_config.json`
- `training_history.json`
- `evaluation_history.json`
- `latest_eval.json`
- `best_eval.json`
- `latest_summary.txt`

## Evaluate

```powershell
python -m dinoia eval-sim-dqn --best --episodes 10
```

## Play

Render the simulator:

```powershell
python -m dinoia play-sim-dqn --best --duration 30
```

Run headless:

```powershell
python -m dinoia play-sim-dqn --best --duration 30 --no-display
```

## Results

```powershell
python -m dinoia results
```

## Structure

- `dinoia/sim/env.py`: Gymnasium simulator, `DinoEnv`.
- `dinoia/observations.py`: 13-value vector observation builder.
- `dinoia/dqn_agent.py`: train, evaluate, play, and summarize DQN artifacts.
- `dinoia/cli.py`: command-line interface.
