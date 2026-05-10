from dinoia.cli import main


if __name__ == "__main__":
    # Legacy wrapper kept for comparing PPO against the old NEAT workflow.
    raise SystemExit(main(["evolve-real"]))
