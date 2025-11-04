from poi.dataset.rqvae import get_dataloader
from poi.rqvae import RQVAEConfig
from poi.rqvae.inference import encode_poi_sid, load_inference_model

if __name__ == "__main__":
    config = RQVAEConfig(run_name="Nrqvae-NYC-div0.00-commit0.25-lr1e-3", dataset_name="NYC", div_weight=0.0, commitment_weight=0.25)

    model = load_inference_model(config, from_hub=True)

    ds = get_dataloader(config.dataset_path, batch_size=config.batch_size, num_workers=config.num_dataloader_workers, device=config.device)

    batch = next(iter(ds))
    results = encode_poi_sid(model, batch.to(config.device))
    print(results)
