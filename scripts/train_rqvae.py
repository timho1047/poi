from poi.rqvae import RQVAEConfig, train_rqvae

if __name__ == "__main__":
    rqvae_config = RQVAEConfig(run_name="rqvae-4")
    train_rqvae(rqvae_config)