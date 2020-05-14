import kaggle
import shutil

kaggle.api.authenticate()

kaggle.api.dataset_download_files('paultimothymooney/chest-xray-pneumonia', path='./', unzip=True)

try:
	shutil.rmtree("./chest_xray/chest_xray")
except:
	print("Error removing duplicate images.")

kaggle.api.dataset_download_files('tawsifurrahman/covid19-radiography-database', path='./', unzip=True)

try:
  shutil.move("./COVID-19 Radiography Database/COVID-19", 
              "./cdb_pneumonia/chest_xray")
  shutil.rmtree("./COVID-19 Radiography Database")
except:
	print("Error removing non Covid19 images.")