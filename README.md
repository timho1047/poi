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

