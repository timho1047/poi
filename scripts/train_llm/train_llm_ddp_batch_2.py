"""
Run command (We need to activate the virtual environment to ensure torchrun available):
source .venv/bin/activate
python scripts/train_llm/train_llm_batch.py

"""

from poi import settings
from poi.llm.ddp_utils import CreateRunBatchItem, create_run_batch
from poi.llm.trainer import train_llm_ddp_batch

NYC_DS = settings.DATASETS_DIR / "NYC_Exploration" / "LLM Dataset"
TKY_DS = settings.DATASETS_DIR / "TKY_Exploration" / "LLM Dataset"

########################################################
### List of runs to train
########################################################
RUN_ITEMS: list[CreateRunBatchItem] = [
    CreateRunBatchItem(
        run_name="new-llama3-nyc-exploration-base",
        dataset_dir=NYC_DS / "Nrqvae-NYC_Exploration-div0.25-commit0.25-lr1e-3",
    ),
    CreateRunBatchItem(
        run_name="new-llama3-tky-exploration-base",
        dataset_dir=TKY_DS / "Nrqvae-TKY_Exploration-div0.25-commit0.25-lr1e-3",
    ),
]

########################################################
### Training configurations
########################################################
NUM_EPOCHS = 8
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
DO_DDP = True
DO_EVAL = True
MAX_EXAMPLES = None
PUSH_TO_HUB = True
FORCE_PUSH = False

SCRIPT_PATH = settings.ROOT_DIR / "scripts/train_llm/train_llm_ddp_single.py"
NPROC_PER_NODE = 8

if __name__ == "__main__":
    runs = create_run_batch(RUN_ITEMS, NUM_EPOCHS, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, DO_DDP, DO_EVAL, MAX_EXAMPLES, PUSH_TO_HUB, FORCE_PUSH)

    print(f"Created {len(runs)} runs")
    for run in runs:
        print(f"Run: {run['config'].run_name}")
        print(f"Train dataset: {run['train_dataset_path']}")
        print(f"Eval dataset: {run['eval_dataset_path']}")
        print(f"Max examples: {run['max_examples']}")
        print(f"Push to hub: {run['push_to_hub']}")
        print(f"Force push: {run['force_push']}")
        print("-" * 70)

    train_llm_ddp_batch(runs, script_path=SCRIPT_PATH, nproc_per_node=NPROC_PER_NODE)
