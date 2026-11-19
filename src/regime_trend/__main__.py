from .cli import main

if __name__ == "__main__":  # guard required: HMM fitting uses worker processes (spawn on Windows)
    raise SystemExit(main())
