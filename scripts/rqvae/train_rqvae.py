from poi.rqvae import RQVAEConfig, train_rqvae

if __name__ == "__main__":
    rqvae_config = RQVAEConfig(run_name="Nrqvae-NYC-div0.25-commit0.25-lr1e-3", dataset_name="NYC", div_weight=0.25, commitment_weight=0.25)
    train_rqvae(rqvae_config, push_to_hub=False)