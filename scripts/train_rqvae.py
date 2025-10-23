from poi.rqvae import RQVAEConfig, train_rqvae

if __name__ == "__main__":
    rqvae_config = RQVAEConfig(run_name='rqvae-nyc-div0.1-commit0.5-lr5e-5', dataset_name='NYC', lr=5e-5, div_weight=0.1, commitment_weight=0.5)
    train_rqvae(rqvae_config, push_to_hub=True)