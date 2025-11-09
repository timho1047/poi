## Evaluation Results

The following LLMs are trained with 64 effective batch size, 1e-5 learning rate, 8 epochs with early stopping, lora rank 16, lora alpha 32, lora dropout 0.1, 4-bit quantization if not specified. The test models are trained using the processed dataset from the original paper. `Test Accuracy` is the main column we care about.

### NYC

| ID | Model | Validation Accuracy | Test All Accuracy | Train Accuracy | Test Accuracy |
|-------|-------|-------|-------|-------|-------|
| llama3-nyc-test | Paper Base |  - | - | 0.3711 | 0.3368 |
| llama3-nyc-test-no-sid | Paper Base w/o sid | - | - | 0.3933 | 0.3204 |
| llama3-nyc-test-full-fintune | Paper Base (Unquantized) | - | - | 0.3743 | 0.3276 | 
| new-llama3-nyc-base | Our Base | 0.2963 | 0.3271 | 0.3869 | 0.3155 |
| new-llama3-nyc-no-quant | Our Ablation w/o L_quant | 0.2902 | 0.3429 | 0.4080 | 0.3161 |
| new-llama3-nyc-no-div | Our Ablation w/o L_div | 0.2939 | 0.3318 | 0.4129 | 0.3100 |
| new-llama3-nyc-no-time | Our Ablation w/o time | 0.2183 | 0.2268 | 0.2542 | 0.2149 |
| llama3-nyc-no-sid | Our Ablation w/o sid | 0.3024 | 0.3253 | 0.4041 | 0.3216 |
| new-llama3-nyc-div-0.5 | Our div 0.5 | 0.2890 | 0.3309 | 0.3796 | 0.3039 |
| new-llama3-nyc-div-0.75 | Our div 0.75  | 0.2963 | 0.3327 | 0.3803 | 0.3100 |
| new-llama3-nyc-kl | Our KL | 0.2720 | 0.3234 | 0.4038 | 0.3074 |
| new-llama3-nyc-exploration-base | Our exploration | 0.3305 | 0.3671 | 0.4754 | 0.3606 |


### TKY

| ID | Model | Validation Accuracy | Test All Accuracy | Train Accuracy | Test Accuracy |
|-------|----------------|---------------|------------------|-------|-------|
| new-llama3-tky-base | Our Base | 0.2439 | 0.2752 | 0.3322 | 0.2476 |
| new-llama3-tky-no-quant | Our Ablation w/o L_quant | 0.2485 | 0.2695 | 0.3388 | 0.2456 |
| new-llama3-tky-no-div | Our Ablation w/o L_div | 0.2241 | 0.2590 | 0.3071 | 0.2204 |
| new-llama3-tky-no-time | Our Ablation w/o time | 0.1780 | 0.1994 | 0.2323 | 0.1823 |
| llama3-tky-no-sid | Our Ablation w/o sid | 0.2378 | 0.2756 | 0.3800 | 0.2436 |
| new-llama3-tky-div-0.5 | Our div 0.5 | 0.2465 | 0.2857 | 0.3567 | 0.2530 |
| new-llama3-tky-div-0.75 | Our div 0.75  | 0.2434 | 0.2822 | 0.3571 | 0.2450 |
| new-llama3-tky-kl | Our KL | 0.2444 | 0.2691 | 0.3168 | 0.2308 |
| new-llama3-tky-exploration-base | Our exploration | 0.2728 | 0.3195 | 0.3413 | 0.2879 |

### Cross-Evaluation
Using base models.

| Train Data\Test Data | NYC | TKY |
|-------|-------|-------|
| NYC | 0.3155 | 0.2455 |
| TKY | 0.3118 | 0.2475 |



## Setup

### Manage Dependencies and run scripts

This repo use `uv` as package manager. `uv` is a tool that helps you manage the dependencies of your project, written in `rust`. It is much faster than `pip`, and use standard `pyproject.toml` file to manage the project (rather than `requirements.txt`). 

`uv` is installed in this cloud instance already.

To install the dependencies, run:

```bash
uv sync
```

To add a new dependency, run:
```bash
uv add <dependency>
```

To remove a dependency, run:
```bash
uv remove <dependency>
```

To run a python script, run:
```bash
uv run path/to/your/script.py
```

If you want to run a script using the local python manually without using `uv`, you can activate the virtual environment:
```bash
source .venv/bin/activate  # activate the virtual environment, 
                           # you should see (poi) in the beginning 
                           # of the prompt.

python path/to/your/script.py
```


All dependencies listed in `pyproject.toml` will be installed in `.venv` (the virtual environment) directory.

### Local Machine

If you are using local machine, you can install `uv` via `pip` by running:
```bash
pip install uv
```

Then you can use `uv` command to manage the dependencies as mentioned above.

### Check GPU Utilization

To check the GPU utilization (cuda), you can use the following command:
```bash
nvidia-smi # will show the GPU utilization of all GPUs
nvidia-smi -l 1 # will show the GPU utilization of all GPUs every 1 second

nvtop # will show the GPU utilization in a prettier way, which can be exited by pressing `q`
```

## Development Guidelines

### Source and Scripts

We build the library under `src/poi` directory. All reusable code should be well organized and placed under this directory.

If we want to train or test the model or modules, we should create a new script under `scripts` directory, whether it be `.py` or `.ipynb` file. You can import anything in our library in these scripts by `from poi import <module_name>`:
```python
from poi.rqvae import RQVAE, RQVAEConfig
from poi.settings import DEVICE
from poi.dataset.rqvae import get_dataloader

config = RQVAEConfig()

train_loader = get_dataloader(config.dataset_path, batch_size=config.batch_size, num_workers=config.num_dataloader_workers, device=DEVICE)

model = RQVAE(
    embedding_dim=config.embedding_dim,
    vae_hidden_dims=config.vae_hidden_dims,
    vector_dim=config.vector_dim,
    vector_num=config.vector_num,
    codebook_num=config.codebook_num,
    commitment_weight=config.commitment_weight,
    random_state=config.random_state,
).to(DEVICE)

model.eval()
```

### Configurations

We should put global configurations in `src/poi/settings.py` file.

For model training, we should create a corresponding config class to initialize the default parameters for the model. If we need to use different parameters, just instantiate the config with different parameters. For example:
```python
@dataclass
class RQVAEConfig:
    # Training parameters
    dataset_name: Literal["NYC", "TKY", "GWL"] = "GWL"
    batch_size: int = 128
    epoch_num: int = 50
    lr: float = 1e-5
    run_name: str = "rqvae-1"

    num_dataloader_workers: int = 4  # 数据加载并行进程数，可根据 CPU 核数调整（2~8）
    device: Literal["cpu", "cuda", "mps"] = settings.DEVICE

    # Model parameters
    codebook_num: int = 3
    vector_num: int = 64
    vector_dim: int = 64
    vae_hidden_dims: list[int] = field(default_factory=lambda: [128, 512, 1024])
    ...

config = RQVAEConfig(run_name="rqvae-tky-lr-1e-5")
train_rqvae(config)
```

### API
1. See `scripts/inference_rqvae_example.py` for the API of encoding POI sids.
2. See `scripts/inference_llm.py` for the API of LLM inference.
3. See `scripts/train_llm/train_llm.py` for the API of training LLM.
4. See `scripts/train_rqvae.py` for the API of training RQVAE.

### Dataset
All datasets should be placed under `datasets` directory. We can download the datasets from Hugging Face by `uv run scripts/download_datasets.py`.
