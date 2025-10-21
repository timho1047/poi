from poi.dataset.rqvae import get_dataloader
from poi.rqvae import RQVAEConfig
from poi.rqvae.inference import encode_poi_sid, load_inference_model
from poi.settings import DEVICE

config = RQVAEConfig(run_name="rqvae-4")

# Get a batch of data to run
train_loader = get_dataloader(config.dataset_path, batch_size=config.batch_size, num_workers=config.num_dataloader_workers, device=DEVICE)
batch = next(iter(train_loader)).to(DEVICE)

# Set from_hub to True to load the model from the Hugging Face hub, or False to load the model from the local checkpoint
model = load_inference_model(config, from_hub=True)  

# Encode POI sids
sids = encode_poi_sid(model, batch)
print(sids)
