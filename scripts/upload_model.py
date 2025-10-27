from poi import settings
from poi.dataset.llm import load_tokenized_llm_dataset
from poi.llm import LLMConfig
from poi.llm.trainer import train_full_llm_fast
from huggingface_hub import create_repo, repo_exists, upload_file, upload_folder

if __name__ == "__main__":
    config = LLMConfig(
        run_name="llama3-nyc-test-full-fintune", num_epochs=8, batch_size=4, gradient_accumulation_steps=16, do_eval=True, resume_from_checkpoint=True
    )
    
    print(f"Uploading model to {config.hub_id}...")
    if not repo_exists(config.hub_id, token=settings.HF_TOKEN):
        create_repo(repo_id=config.hub_id, token=settings.HF_TOKEN, private=False)

    upload_folder(
        folder_path=str(config.output_dir),
        repo_id=config.hub_id,
        token=settings.HF_TOKEN,
        commit_message=f"Training completed for {config.run_name}",
        ignore_patterns=["checkpoint-*"],
    )