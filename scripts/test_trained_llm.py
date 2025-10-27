from poi import settings
from poi.dataset.llm import load_prompt_completion_llm_dataset
from poi.llm import LLMConfig, inference, load_fast_inference_model


def format_input(ex):
    text = ex["instruction"] + "\n" + ex["input"] + " "
    return {"input": text, "output": ex["output"]}


if __name__ == "__main__":
    config = LLMConfig(run_name="llama3-nyc-test", num_epochs=8, batch_size=4, gradient_accumulation_steps=4, do_eval=True)

    DATASET_DIR = settings.DATASETS_DIR / "NYC" / "LLM Dataset" / "test_codebook.json"
    ds = load_prompt_completion_llm_dataset(DATASET_DIR)

    index = 1

    model = load_fast_inference_model(config)
    res = inference(config, model, ds[index]["prompt"])

    print("-------------input--------------")
    print(ds[index]["prompt"])
    print("-------------response--------------")
    print(res)
    print("-------------correct--------------")
    print(ds[index]["completion"])
