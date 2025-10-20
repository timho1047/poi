from poi.dataset.rqvae import get_dataloader
from poi.rqvae import RQVAEConfig
from poi.rqvae.inference import encode_poi_sid, load_model
from poi.settings import DEVICE

config = RQVAEConfig()

# Get a batch of data to run
train_loader = get_dataloader(config.dataset_path, batch_size=config.batch_size, num_workers=config.num_dataloader_workers, device=DEVICE)
batch = next(iter(train_loader)).to(DEVICE)

# Load model from config
model = load_model(config, from_checkpoint=False)  # Set from_checkpoint to True to load the model from the best checkpoint

# Encode POI sids
sids = encode_poi_sid(model, batch)
print(sids)
