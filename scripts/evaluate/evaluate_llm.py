from poi import settings
from poi.llm import LLMConfig
from poi.llm.evaluate import evaluate_all_and_save

if __name__ == "__main__":
    NYC_BASE_DS = settings.DATASETS_DIR / "NYC" / "LLM Dataset" / "rqvae-nyc-div0.25-commit0.5-lr5e-5"
    NYC_NO_TIME_DS = settings.DATASETS_DIR / "NYC" / "LLM Dataset" / "ablation without Time"
    NYC_NO_DIV_DS = settings.DATASETS_DIR / "NYC" / "LLM Dataset" / "rqvae-nyc-div0.0-commit0.5-lr5e-5"
    NYC_NO_SID_DS = settings.DATASETS_DIR / "NYC" / "LLM Dataset" / "ablation without SID"
    NYC_NO_QUANT_DS = settings.DATASETS_DIR / "NYC" / "LLM Dataset" / "rqvae-nyc-div0.25-commit0.5-lr5e-5-without_L_quant"
    NYC_DIV_0_5_DS = settings.DATASETS_DIR / "NYC" / "LLM Dataset" / "rqvae-nyc-div0.5-commit0.5-lr5e-5"
    NYC_DIV_0_75_DS = settings.DATASETS_DIR / "NYC" / "LLM Dataset" / "rqvae-nyc-div0.75-commit0.5-lr5e-5"

    TKY_BASE_DS = settings.DATASETS_DIR / "TKY" / "LLM Dataset" / "rqvae-tky-div0.25-commit0.25-lr5e-5"
    TKY_NO_TIME_DS = settings.DATASETS_DIR / "TKY" / "LLM Dataset" / "ablation without Time"
    TKY_NO_DIV_DS = settings.DATASETS_DIR / "TKY" / "LLM Dataset" / "rqvae-tky-div0.0-commit0.25-lr5e-5"
    TKY_NO_SID_DS = settings.DATASETS_DIR / "TKY" / "LLM Dataset" / "ablation without SID"
    TKY_NO_QUANT_DS = settings.DATASETS_DIR / "TKY" / "LLM Dataset" / "rqvae-tky-div0.25-commit0.25-lr5e-5-without_L_quant"
    TKY_DIV_0_5_DS = settings.DATASETS_DIR / "TKY" / "LLM Dataset" / "rqvae-tky-div0.5-commit0.25-lr5e-5"
    TKY_DIV_0_75_DS = settings.DATASETS_DIR / "TKY" / "LLM Dataset" / "rqvae-tky-div0.75-commit0.25-lr5e-5"

    CONFIG_DS_DIR_PAIRS = [
        (LLMConfig(run_name="llama3-nyc-base"), NYC_BASE_DS),
        (LLMConfig(run_name="llama3-nyc-no-time"), NYC_NO_TIME_DS),
        (LLMConfig(run_name="llama3-nyc-no-div"), NYC_NO_DIV_DS),
        (LLMConfig(run_name="llama3-nyc-no-sid"), NYC_NO_SID_DS),
        (LLMConfig(run_name="llama3-nyc-no-quant"), NYC_NO_QUANT_DS),
        (LLMConfig(run_name="llama3-nyc-div-0.5"), NYC_DIV_0_5_DS),
        (LLMConfig(run_name="llama3-nyc-div-0.75"), NYC_DIV_0_75_DS),
    ]

    CONFIG_DS_DIR_PAIRS += [
        (LLMConfig(run_name="llama3-tky-base"), TKY_BASE_DS),
        (LLMConfig(run_name="llama3-tky-no-time"), TKY_NO_TIME_DS),
        (LLMConfig(run_name="llama3-tky-no-div"), TKY_NO_DIV_DS),
        (LLMConfig(run_name="llama3-tky-no-sid"), TKY_NO_SID_DS),
        (LLMConfig(run_name="llama3-tky-no-quant"), TKY_NO_QUANT_DS),
        (LLMConfig(run_name="llama3-tky-div-0.5"), TKY_DIV_0_5_DS),
        (LLMConfig(run_name="llama3-tky-div-0.75"), TKY_DIV_0_75_DS),
    ]
    evaluate_all_and_save(CONFIG_DS_DIR_PAIRS, settings.OUTPUT_DIR / "llm_metrics.csv")
