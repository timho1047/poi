from poi.llm import LLMConfig, load_inference_model, inference


if __name__ == "__main__":
    config = LLMConfig()
    model = load_inference_model(config)
    prompt = "Hey how are you doing today?"
    response = inference(config, model, prompt)
    print(response)