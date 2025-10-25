# RQ-VAE Batch Inference

This document describes the refactored RQ-VAE batch inference system, which has been split into reusable components in `src/poi/rqvae/` and simple scripts in `scripts/`.

## Architecture

### Core Module (`src/poi/rqvae/batch_inference.py`)

The core functionality is organized into three main classes:

#### 1. `ModelConfigParser`
- **Purpose**: Parses Hugging Face model IDs to extract RQ-VAE configuration parameters
- **Key Methods**:
  - `parse_model_name()`: Static method to parse model ID and extract configuration
  - Extracts dataset name, div_weight, commitment_weight, learning rate, and quant_weight
  - Handles special flags like 'without_L_quant'

#### 2. `BatchInferenceProcessor`
- **Purpose**: Handles the complete inference pipeline for a single RQ-VAE model
- **Key Methods**:
  - `process_model()`: Main method for processing a single model
  - `_process_batches()`: Internal method for batch processing
  - `_upload_results()`: Internal method for uploading results to HF
  - Handles model loading, data processing, and result upload

#### 3. `BatchInferenceManager`
- **Purpose**: Manages batch processing of multiple RQ-VAE models
- **Key Methods**:
  - `process_models()`: Process a list of models
  - `process_single_model()`: Process a single model for testing
  - `get_model_list()`: Get all target models
  - `get_nyc_models()`: Get NYC-specific models
  - `get_tky_models()`: Get TKY-specific models

### Script (`scripts/inference_rqvae.py`)

A simple wrapper script that:
- Imports the `BatchInferenceManager` class
- Sets up the HF token
- Calls the main processing method

## Usage

### Basic Usage
```bash
# Set environment variable
export HF_TOKEN="your_huggingface_token_here"

# Run the script
uv run scripts/inference_rqvae.py
```

### Programmatic Usage
```python
from poi.rqvae.batch_inference import BatchInferenceManager

# Create manager
manager = BatchInferenceManager(hf_token="your_token")

# Process all models
manager.process_models()

# Or process specific models
custom_models = ["comp5331poi/rqvae-tky-div0.0-commit0.25-lr5e-5"]
manager.process_models(custom_models)

# Process single model for testing
manager.process_single_model("comp5331poi/rqvae-nyc-div0.25-commit0.5-lr5e-5")
```

### Individual Component Usage
```python
from poi.rqvae.batch_inference import ModelConfigParser, BatchInferenceProcessor

# Parse model configuration
config = ModelConfigParser.parse_model_name("comp5331poi/rqvae-tky-div0.25-commit0.25-lr5e-5")
print(f"Dataset: {config['dataset_name']}")
print(f"Div Weight: {config['div_weight']}")

# Use processor for single model
processor = BatchInferenceProcessor(hf_token)
processor.process_model("comp5331poi/rqvae-tky-div0.0-commit0.25-lr5e-5")
```

## Model Configuration

The system supports 10 pre-trained RQ-VAE models:

### NYC Models (5 models)
- `comp5331poi/rqvae-nyc-div0.0-commit0.5-lr5e-5`
- `comp5331poi/rqvae-nyc-div0.25-commit0.5-lr5e-5`
- `comp5331poi/rqvae-nyc-div0.25-commit0.5-lr5e-5-without_L_quant`
- `comp5331poi/rqvae-nyc-div0.5-commit0.5-lr5e-5`
- `comp5331poi/rqvae-nyc-div0.75-commit0.5-lr5e-5`

### TKY Models (5 models)
- `comp5331poi/rqvae-tky-div0.0-commit0.25-lr5e-5`
- `comp5331poi/rqvae-tky-div0.25-commit0.25-lr5e-5`
- `comp5331poi/rqvae-tky-div0.25-commit0.25-lr5e-5-without_L_quant`
- `comp5331poi/rqvae-tky-div0.5-commit0.25-lr5e-5`
- `comp5331poi/rqvae-tky-div0.75-commit0.25-lr5e-5`

## Output Structure

The inference process generates CSV files with the following structure:

```
comp5331poi/{dataset}/LLM Dataset/Intermediate Files/
└── codebooks-rqvae-{model_name}.csv
```

Each CSV file contains:
- **Pid**: Mapped POI ID
- **Codebook**: SID (Semantic ID) as list of integers
- **Vector**: Raw feature vector

## Key Features

1. **PID Mapping**: Ensures correct POI-to-codebook correspondence
2. **No Shuffling**: Maintains consistent data order for reproducible results
3. **Batch Processing**: Efficient processing of large datasets
4. **Error Handling**: Robust error handling with detailed logging
5. **Progress Tracking**: Real-time progress updates during processing
6. **Temporary File Management**: Automatic cleanup of temporary files

## Testing

Run the test script to verify functionality:
```bash
uv run scripts/test_batch_inference.py
```

This will test:
- Model configuration parsing
- Manager instantiation and model list retrieval
- Processor instantiation
- All model configurations

## Benefits of Refactoring

1. **Reusability**: Core components can be imported and used independently
2. **Maintainability**: Clear separation of concerns
3. **Testability**: Individual components can be tested in isolation
4. **Extensibility**: Easy to add new features or modify existing ones
5. **Documentation**: Well-documented classes and methods
6. **Simplicity**: Script remains simple and focused

## Performance Notes

- **Memory Usage**: Processes data in batches to manage memory efficiently
- **Network Requirements**: Requires stable internet for model downloads and uploads
- **Processing Time**: Varies by dataset size (minutes to hours)
- **Disk Space**: Uses temporary files during processing
