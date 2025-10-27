The following LLMs are trained with 64 effective batch size, 1e-5 learning rate, 8 epochs with early stopping, lora rank 16, lora alpha 32, lora dropout 0.1, 4-bit quantization if not specified.

### NYC

| ID | Model | Train Accuracy | Test Accuracy | Test All Accuracy |
|-------|-------|-------|-------|-------|
| llama3-nyc-test-full-fintune | Paper Base (Unquantized) | 0.3743 | 0.3276 | - |
| llama3-nyc-test | Paper Base | 0.3680 | 0.3276 | - |
| llama3-nyc-base | Our Base | - | - | - |
| - | Our Ablation w/o L_quant | - | - | - |
| - | Our Ablation w/o L_div | - | - | - |
| - | Our Ablation w/o time | - | - | - |
| - | Our Ablation w/o sid | - | - | - |
| - | Our div 0.5 | - | - | - |
| Our div 0.75  | - | - | - |


### TKY

| ID | Model | Train Accuracy | Test Accuracy | Test All Accuracy |
|-------|----------------|---------------|------------------|-------|
| - | Our Base | - | - | - |
| - | Our Ablation w/o L_quant | - | - | - |
| - | Our Ablation w/o L_div | - | - | - |
| - | Our Ablation w/o time | - | - | - |
| - | Our Ablation w/o sid | - | - | - |
| - | Our div 0.5 | - | - | - |
| - | Our div 0.75  | - | - | - |