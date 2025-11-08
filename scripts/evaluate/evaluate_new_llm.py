from poi import settings
from poi.llm import LLMConfig
from poi.llm.evaluate import evaluate_all_and_save

NYC_DS = settings.DATASETS_DIR / "NYC" / "New LLM Dataset"
TKY_DS = settings.DATASETS_DIR / "TKY" / "New LLM Dataset"


# NYC
CONFIG_DS_DIR_PAIRS = [
    (LLMConfig(run_name="new-llama3-nyc-base"), NYC_DS / "Nrqvae-NYC-div0.25-commit0.25-lr1e-3"),
    (LLMConfig(run_name="new-llama3-nyc-no-time"), NYC_DS / "ablation without Time"),
    (LLMConfig(run_name="new-llama3-nyc-no-div"), NYC_DS / "Nrqvae-NYC-div0.00-commit0.25-lr1e-3"),
    (LLMConfig(run_name="new-llama3-nyc-no-quant"), NYC_DS / "Nrqvae-without_L_quant-NYC-div0.25-commit0.25-lr1e-3"),
    (LLMConfig(run_name="new-llama3-nyc-div-0.5"), NYC_DS / "Nrqvae-NYC-div0.50-commit0.25-lr1e-3"),
    (LLMConfig(run_name="new-llama3-nyc-div-0.75"), NYC_DS / "Nrqvae-NYC-div0.75-commit0.25-lr1e-3"),
    (LLMConfig(run_name="new-llama3-nyc-kl"), NYC_DS / "Nrqvae-withKL-NYC-div0.25-commit0.25-lr1e-3"),
]

# TKY
CONFIG_DS_DIR_PAIRS += [
    (LLMConfig(run_name="new-llama3-tky-base"), TKY_DS / "Nrqvae-TKY-div0.25-commit0.25-lr1e-3"),
    (LLMConfig(run_name="new-llama3-tky-no-time"), TKY_DS / "ablation without Time"),
    (LLMConfig(run_name="new-llama3-tky-no-div"), TKY_DS / "Nrqvae-TKY-div0.00-commit0.25-lr1e-3"),
    (LLMConfig(run_name="new-llama3-tky-no-quant"), TKY_DS / "Nrqvae-without_L_quant-TKY-div0.25-commit0.25-lr1e-3"),
    (LLMConfig(run_name="new-llama3-tky-div-0.5"), TKY_DS / "Nrqvae-TKY-div0.50-commit0.25-lr1e-3"),
    (LLMConfig(run_name="new-llama3-tky-div-0.75"), TKY_DS / "Nrqvae-TKY-div0.75-commit0.25-lr1e-3"),
    (LLMConfig(run_name="new-llama3-tky-kl"), TKY_DS / "Nrqvae-withKL-TKY-div0.25-commit0.25-lr1e-3"),
]


if __name__ == "__main__":
    evaluate_all_and_save(CONFIG_DS_DIR_PAIRS, settings.OUTPUT_DIR / "new_llm_metrics.csv")
