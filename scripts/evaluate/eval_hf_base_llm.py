from tqdm import tqdm

from poi import settings
from poi.dataset.llm import load_prompt_completion_llm_dataset
from poi.llm import LLMConfig, inference, load_fast_inference_model


def top_one_accuracy(model, ds):
    total = len(ds)
    correct = 0
    for i in tqdm(range(total)):
        res = ""
        retry_count = 0
        while retry_count < 5 and res == "":
            res = inference(config, model, ds[i]["prompt"] + "<a_").strip()  # provide <a_ as prefix hint
            retry_count += 1
        res = "<a_" + res
        if ds[i]["completion"] in res:
            correct += 1
    return correct / total


if __name__ == "__main__":
    config = LLMConfig(run_name="llama3-nyc-base")
    model = load_fast_inference_model(config, from_hub=True)
    eval_dataset = load_prompt_completion_llm_dataset(
        settings.DATASETS_DIR / "NYC" / "LLM Dataset" / "rqvae-nyc-div0.25-commit0.5-lr5e-5" / "test_codebook.json"
    )
    metrics = top_one_accuracy(model, eval_dataset)
    print(metrics)
