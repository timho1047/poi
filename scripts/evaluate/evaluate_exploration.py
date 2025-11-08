from poi import settings
from poi.llm import LLMConfig
from poi.llm.evaluate import evaluate_all_and_save

NYC_DS = settings.DATASETS_DIR / "NYC_Exploration" / "LLM Dataset"
TKY_DS = settings.DATASETS_DIR / "TKY_Exploration" / "LLM Dataset"


# NYC
CONFIG_DS_DIR_PAIRS = [
    (LLMConfig(run_name="new-llama3-nyc-exploration-base"), NYC_DS / "Nrqvae-NYC_Exploration-div0.25-commit0.25-lr1e-3"),
    (LLMConfig(run_name="new-llama3-tky-exploration-base"), TKY_DS / "Nrqvae-TKY_Exploration-div0.25-commit0.25-lr1e-3"),
]



if __name__ == "__main__":
    evaluate_all_and_save(CONFIG_DS_DIR_PAIRS, settings.OUTPUT_DIR / "new_llm_metrics.csv")
