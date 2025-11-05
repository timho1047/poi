import argparse
from pathlib import Path

from poi.llm.ddp_utils import load_run_config
from poi.llm.trainer import train_llm_ddp_single_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    run = load_run_config(Path(args.config))

    train_llm_ddp_single_run(run)
