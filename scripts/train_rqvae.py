from poi.rqvae import RQVAEConfig, train_rqvae

if __name__ == "__main__":
    rqvae_config = RQVAEConfig(run_name="auto")
    if rqvae_config.run_name == "auto":
        auto_name = rqvae_config.generate_run_name()
        rqvae_config.set_run_name(auto_name)
        print(f"[RunName] Using auto-generated run name: {rqvae_config.run_name}")
    train_rqvae(rqvae_config, push_to_hub=False)