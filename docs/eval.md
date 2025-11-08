The following LLMs are trained with 64 effective batch size, 1e-5 learning rate, 8 epochs with early stopping, lora rank 16, lora alpha 32, lora dropout 0.1, 4-bit quantization if not specified.

## New RQVAE

### NYC

| ID | Model | Validation Accuracy | Test All Accuracy | Train Accuracy | Test Accuracy |
|-------|-------|-------|-------|-------|-------|
| llama3-nyc-test | Paper Base |  - | - | 0.3711 | 0.3368 |
| llama3-nyc-test-no-sid | Paper Base w/o sid | - | - | 0.3933 | 0.3204 |
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

| Train Data\Test Data | NYC | TKY |
|-------|-------|-------|
| NYC | 0.3155 | 0.2455 |
| TKY | 0.3118 | 0.2475 |

## Old RQVAE

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