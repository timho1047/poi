The following LLMs are trained with 64 effective batch size, 1e-5 learning rate, 8 epochs with early stopping, lora rank 16, lora alpha 32, lora dropout 0.1, 4-bit quantization if not specified.

### NYC

| ID | Model | Validation Accuracy | Test All Accuracy | Train Accuracy | Test Accuracy |
|-------|-------|-------|-------|-------|-------|
| llama3-nyc-test-full-fintune | Paper Base (Unquantized) | - | - | 0.3743 | 0.3276 | 
| llama3-nyc-test | Paper Base |  - | - | 0.3680 | 0.3276 |
| llama3-nyc-base | Our Base | 0.2744 | 0.3467 | 0.3859 | 0.3184 |
| llama3-nyc-no-quant | Our Ablation w/o L_quant | 0.2890 | 0.3178 | 0.3827 | 0.3099 |
| llama3-nyc-no-div | Our Ablation w/o L_div | 0.2720 | 0.3383 | 0.3676 | 0.2966 |
| llama3-nyc-no-time | Our Ablation w/o time | 0.2207 | 0.2481 | 0.2525 | 0.2107 |
| llama3-nyc-no-sid | Our Ablation w/o sid | 0.2890 | 0.3346 | 0.4077 | 0.3257 |
| llama3-nyc-div-0.5 | Our div 0.5 | 0.2768 | 0.3299 | 0.3912 | 0.3184 |
| llama3-nyc-div-0.75 | Our div 0.75  | 0.2805 | 0.3281 | 0.3894 | 0.3160 |


### TKY

| ID | Model | Validation Accuracy | Test All Accuracy | Train Accuracy | Test Accuracy |
|-------|----------------|---------------|------------------|-------|-------|
| llama3-tky-base | Our Base | 0.2561 | 0.2730 | 0.3651 | 0.2453 |
| llama3-tky-no-quant | Our Ablation w/o L_quant | 0.2515 | 0.2805 | 0.3380 | 0.2433 |
| llama3-tky-no-div | Our Ablation w/o L_div | 0.2394 | 0.2708 | 0.3273 | 0.2458 |
| llama3-tky-no-time | Our Ablation w/o time | 0.1831 | 0.2090 | 0.2241 | 0.2000 |
| llama3-tky-no-sid | Our Ablation w/o sid | 0.2525 | 0.2717 | 0.3786 | 0.2504 |
| llama3-tky-div-0.5 | Our div 0.5 | 0.2454 | 0.2770 | 0.3255 | 0.2499 |
| llama3-tky-div-0.75 | Our div 0.75  | 0.2394 | 0.2708 | 0.3317 | 0.2448 |