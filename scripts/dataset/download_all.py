from poi.dataset.download import download_dataset

if __name__ == "__main__":
    download_dataset("NYC")
    download_dataset("TKY")
    download_dataset("NYC_Exploration")
    download_dataset("TKY_Exploration")