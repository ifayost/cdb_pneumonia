import os
import shutil
import sys

PATH = sys.argv[1]
for i in tqdm(os.listdir(PATH)):
  if i != '__MACOSX':
    os.mkdir(os.path.join(PATH, i, "BACTERIA"))
    os.mkdir(os.path.join(PATH, i, "VIRUS"))
    for j in os.listdir(os.path.join(PATH, i, "PNEUMONIA")):
      image_path = os.path.join(PATH, i, "PNEUMONIA", j)
      if "bacteria" in j:
        shutil.move(image_path, os.path.join(PATH, i , "BACTERIA"))
      elif "virus" in j:
        shutil.move(image_path, os.path.join(PATH, i, "VIRUS"))

    remaining = os.listdir(os.path.join(PATH, i, "PNEUMONIA"))
    if len(remaining)!=0:
      print("The following documents has been removed:")
      print(remaining)
    shutil.rmtree(os.path.join(PATH, i, "PNEUMONIA"))