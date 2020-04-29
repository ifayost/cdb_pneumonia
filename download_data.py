import kaggle
import shutil

kaggle.api.authenticate()

kaggle.api.dataset_download_files('paultimothymooney/chest-xray-pneumonia', path='./', unzip=True)

try:
	shutil.rmtree("./chest_xray/chest_xray")
except:
	print("Error removing duplicate images.")
