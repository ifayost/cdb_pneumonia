import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_files('paultimothymooney/chest-xray-pneumonia', path='./', unzip=True)