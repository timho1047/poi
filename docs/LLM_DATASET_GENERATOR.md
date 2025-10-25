# LLM Dataset Generator

This document describes the refactored LLM dataset generation system, which has been split into reusable components in `src/poi/` and simple scripts in `scripts/`.

## Architecture

### Core Module (`src/poi/dataset/llm_dataset_generator.py`)

The core functionality is organized into four main classes:

#### 1. `DataDownloader`
- **Purpose**: Handles downloading of codebook and data files from Hugging Face
- **Key Methods**:
  - `download_codebook_file()`: Downloads specific codebook CSV files
  - `download_data_files()`: Downloads all data CSV files for a dataset

#### 2. `JSONGenerator`
- **Purpose**: Generates JSON format LLM training data
- **Key Methods**:
  - `generate_json()`: Main method for generating JSON data
  - Supports both ID and codebook formats
  - Supports both time-aware and time-agnostic formats

#### 3. `HFUploader`
- **Purpose**: Handles file uploads to Hugging Face with retry mechanisms
- **Key Methods**:
  - `upload_json_to_hf()`: Upload to model-specific folders
  - `upload_json_to_hf_notime()`: Upload to "ablation without Time" folder
  - `upload_json_to_hf_id()`: Upload to "ablation without SID" folder
  - `_upload_with_retry()`: Internal retry mechanism

#### 4. `LLMDatasetGenerator`
- **Purpose**: Main orchestrator class that coordinates all operations
- **Key Methods**:
  - `generate_id_files()`: Generate dataset-level ID files
  - `process_model()`: Process a single model
  - `generate_all_datasets()`: Generate all datasets for all models

### Script (`scripts/generate_llm_datasets.py`)

A simple wrapper script that:
- Imports the `LLMDatasetGenerator` class
- Sets up the HF token
- Calls the main generation method

## Usage

### Basic Usage
```bash
# Set environment variable
export HF_TOKEN="your_huggingface_token_here"

# Run the script
uv run scripts/generate_llm_datasets.py
```

### Programmatic Usage
```python
from poi.dataset.llm_dataset_generator import LLMDatasetGenerator

# Create generator
generator = LLMDatasetGenerator(hf_token="your_token")

# Generate all datasets
generator.generate_all_datasets()

# Or generate specific components
generator.generate_id_files("NYC", temp_dir)
generator.process_model("NYC", "rqvae-nyc-div0.25-commit0.5-lr5e-5", temp_dir)
```

### Individual Component Usage
```python
from poi.dataset.llm_dataset_generator import DataDownloader, JSONGenerator, HFUploader

# Use individual components
downloader = DataDownloader(hf_token)
json_gen = JSONGenerator()
uploader = HFUploader(hf_token)

# Download data
data_dir = downloader.download_data_files("NYC", temp_dir)

# Generate JSON
json_data, filename = json_gen.generate_json(
    mode="train", 
    target_dataset="NYC", 
    codebook_path=None, 
    data_dir=data_dir, 
    idorcodebook="id", 
    include_time=True
)

# Upload to HF
uploader.upload_json_to_hf_id(json_data, filename, "NYC")
```

## Output Structure

The generator creates the following structure on Hugging Face:

```
comp5331poi/{dataset}/LLM Dataset/
├── ablation without SID/
│   ├── train_id.json
│   ├── val_id.json
│   ├── test_id.json
│   └── test_all_id.json
├── ablation without Time/
│   ├── train_codebook_notime.json
│   ├── val_codebook_notime.json
│   ├── test_codebook_notime.json
│   └── test_all_codebook_notime.json
└── rqvae-{model_name}/
    ├── train_codebook.json
    ├── val_codebook.json
    ├── test_codebook.json
    └── test_all_codebook.json
```

## File Counts

- **ID files**: 2 datasets × 4 files = 8 files
- **Codebook files**: 10 models × 4 files = 40 files
- **Notime files**: 2 models × 4 files = 8 files
- **Total**: 56 files

## Special Notes

- **ID files**: Generated once per dataset (same for all models)
- **Notime files**: Only generated for specific models:
  - `rqvae-nyc-div0.25-commit0.5-lr5e-5`
  - `rqvae-tky-div0.25-commit0.25-lr5e-5`
- **Retry mechanism**: All uploads include automatic retry on failure
- **Network requirements**: Stable internet connection required for HF operations

## Testing

Run the test script to verify functionality:
```bash
uv run scripts/test_llm_generator.py
```

This will test:
- JSON generation with both ID and codebook formats
- Data downloading from Hugging Face
- Class instantiation and basic functionality

## Benefits of Refactoring

1. **Reusability**: Core components can be imported and used independently
2. **Maintainability**: Clear separation of concerns
3. **Testability**: Individual components can be tested in isolation
4. **Extensibility**: Easy to add new features or modify existing ones
5. **Documentation**: Well-documented classes and methods
6. **Simplicity**: Script remains simple and focused
