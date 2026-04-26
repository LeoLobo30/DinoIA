from dinoia.cli import run_benchmark


if __name__ == "__main__":
    run_benchmark(timesteps=3_000, device="auto")
