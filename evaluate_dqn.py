from dinoia.cli import run_evaluate


if __name__ == "__main__":
    run_evaluate(None, episodes=5, render=False, seed=42, latest=True)
