"""
Run command (We need to activate the virtual environment to ensure torchrun available):
source .venv/bin/activate
python scripts/train_llm/train_llm_batch.py

"""

from poi import settings
from poi.llm.ddp_utils import CreateRunBatchItem, create_run_batch
from poi.llm.trainer import train_llm_ddp_batch

NYC_DS = settings.DATASETS_DIR / "NYC" / "New LLM Dataset"
TKY_DS = settings.DATASETS_DIR / "TKY" / "New LLM Dataset"

########################################################
### List of runs to train
########################################################
RUN_ITEMS: list[CreateRunBatchItem] = [
    ###########
    # NYC
    ##########
    CreateRunBatchItem(
        run_name="new-llama3-nyc-base",
        dataset_dir=NYC_DS / "Nrqvae-NYC-div0.25-commit0.25-lr1e-3",
    ),
    CreateRunBatchItem(
        run_name="new-llama3-nyc-no-time",
        dataset_dir=NYC_DS / "ablation without Time",
    ),
    CreateRunBatchItem(
        run_name="new-llama3-nyc-no-div",
        dataset_dir=NYC_DS / "Nrqvae-NYC-div0.00-commit0.25-lr1e-3",
    ),
    CreateRunBatchItem(
        run_name="new-llama3-nyc-no-quant",
        dataset_dir=NYC_DS / "Nrqvae-without_L_quant-NYC-div0.25-commit0.25-lr1e-3",
    ),
    CreateRunBatchItem(
        run_name="new-llama3-nyc-div-0.5",
        dataset_dir=NYC_DS / "Nrqvae-NYC-div0.50-commit0.25-lr1e-3",
    ),
    CreateRunBatchItem(
        run_name="new-llama3-nyc-div-0.75",
        dataset_dir=NYC_DS / "Nrqvae-NYC-div0.75-commit0.25-lr1e-3",
    ),
    CreateRunBatchItem(
        run_name="new-llama3-nyc-kl",
        dataset_dir=NYC_DS / "Nrqvae-withKL-NYC-div0.25-commit0.25-lr1e-3",
    ),
    ###########
    # TKY
    ##########
    CreateRunBatchItem(
        run_name="new-llama3-tky-base",
        dataset_dir=TKY_DS / "Nrqvae-TKY-div0.25-commit0.25-lr1e-3",
    ),
    CreateRunBatchItem(
        run_name="new-llama3-tky-no-time",
        dataset_dir=TKY_DS / "ablation without Time",
    ),
    CreateRunBatchItem(
        run_name="new-llama3-tky-no-div",
        dataset_dir=TKY_DS / "Nrqvae-TKY-div0.00-commit0.25-lr1e-3",
    ),
    CreateRunBatchItem(
        run_name="new-llama3-tky-no-quant",
        dataset_dir=TKY_DS / "Nrqvae-without_L_quant-TKY-div0.25-commit0.25-lr1e-3",
    ),
    CreateRunBatchItem(
        run_name="new-llama3-tky-div-0.5",
        dataset_dir=TKY_DS / "Nrqvae-TKY-div0.50-commit0.25-lr1e-3",
    ),
    CreateRunBatchItem(
        run_name="new-llama3-tky-div-0.75",
        dataset_dir=TKY_DS / "Nrqvae-TKY-div0.75-commit0.25-lr1e-3",
    ),
    CreateRunBatchItem(
        run_name="new-llama3-tky-kl",
        dataset_dir=TKY_DS / "Nrqvae-withKL-TKY-div0.25-commit0.25-lr1e-3",
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
