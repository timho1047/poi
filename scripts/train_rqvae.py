from poi.train.config import RQVAEConfig
from poi.train.rqvae import train_rqvae

if __name__ == "__main__":
    rqvae_config = RQVAEConfig()
    train_rqvae(rqvae_config)