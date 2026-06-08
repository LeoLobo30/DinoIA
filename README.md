# DinoIA

DinoIA is a small reinforcement-learning project for Chrome Dino-style control in a simulated Gymnasium environment, with an optional real-game playback adapter for `chrome://dino/`.

The active path is QR-DQN with a 13-value vector state. Real-game control is available for playback only through Selenium and JavaScript state extraction from the Dino runtime.

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

## Play In The Real Game

Install dependencies first, including Selenium:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Then run the trained model against `chrome://dino/`:

```powershell
python -m dinoia play-real-dqn --best --duration 30
```

Useful options:

- `--url chrome://dino/`
- `--chrome-binary "C:\Program Files\Google\Chrome\Application\chrome.exe"`
- `--headless`
- `--no-restart-on-crash`

## Results

```powershell
python -m dinoia results
```

## Structure

- `dinoia/sim/env.py`: Gymnasium simulator, `DinoEnv`.
- `dinoia/observations.py`: 13-value vector observation builder.
- `dinoia/dqn_agent.py`: train, evaluate, play, and summarize DQN artifacts.
- `dinoia/real_game.py`: Selenium bridge for real-game playback.
- `dinoia/cli.py`: command-line interface.
