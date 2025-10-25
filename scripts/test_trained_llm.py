from datasets import Dataset
from poi import settings
from poi.llm import LLMConfig, inference, load_fast_inference_model


def format_input(ex):
    text = ex["instruction"] + "\n" + ex["input"] + " "
    return {"input": text, "output": ex["output"]}


if __name__ == "__main__":
    config = LLMConfig(run_name="llama3-nyc-test", num_epochs=8, batch_size=4, gradient_accumulation_steps=4, do_eval=True)

    DATASET_DIR = settings.DATASETS_DIR / "NYC" / "LLM Dataset" / "train_codebook.json"
    ds = Dataset.from_json(DATASET_DIR.as_posix()).map(format_input).remove_columns(["instruction"])

    index = 1

    model = load_fast_inference_model(config)
    res = inference(config, model, ds[index]["input"])

    print("-------------input--------------")
    print(ds[index]["input"])
    print("-------------response--------------")
    print(res)
    print("-------------correct--------------")
    print(ds[index]["output"])
