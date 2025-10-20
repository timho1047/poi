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
1. See `scripts/inference_rqvae.py` for the API of encoding POI sids.
2. See `scripts/inference_llm.py` for the API of LLM inference.
3. See `scripts/train_llm.py` for the API of training LLM.
4. See `scripts/train_rqvae.py` for the API of training RQVAE.

### Dataset format
All datasets should be placed under `datasets` directory in the following format (tentative):
```
datasets/
├── NYC/
│   ├── train_codebook.json
│   ├── test_codebook.json
│   ├── poi_features.pt
│   ├── metadata.json
├── TKY/
│   ├── train_codebook.json
│   ├── test_codebook.json
│   ├── poi_features.pt
│   ├── metadata.json
├── GWL/
│   ├── train_codebook.json
│   ├── test_codebook.json
│   ├── poi_features.pt
│   ├── metadata.json
```