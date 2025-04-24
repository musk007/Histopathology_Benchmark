import sys
sys.path.insert(1, '/home/roba.majzoub/Histopathology_Benchmark')
import os
import numpy as np
import pandas as pd
from plip.reproducibility.generate_validation_datasets.artifacts import *

dest_dir1 = "/l/users/roba.majzoub/plip_data/artifacts/level1"
dest_dir2 = "/l/users/roba.majzoub/plip_data/artifacts/level2"
dest_dir3 = "/l/users/roba.majzoub/plip_data/artifacts/level3"
dest_dir4 = "/l/users/roba.majzoub/plip_data/artifacts/level4"
df_path = "/l/users/roba.majzoub/plip_data/LC25000_lung_test.csv"
# df_path = "/l/users/roba.majzoub/plip_data/MSI_MSS_cancerType_test.csv"
df = pd.read_csv(df_path, index_col=False)


os.makedirs(dest_dir1, exist_ok=True)
os.makedirs(dest_dir2, exist_ok=True)
os.makedirs(dest_dir3, exist_ok=True)
os.makedirs(dest_dir4, exist_ok=True)
counter = 0
finished_ids_1 = os.listdir(dest_dir1)
finished_ids_2 = os.listdir(dest_dir2)

finished_ids_1 = [i.split("_")[0] for i in finished_ids_1]
finished_ids_1 = [i.split("_")[0] for i in finished_ids_1]
for i in range(len(df)):
    image_path = df.iloc[i]["image"]

    im_name = image_path.split("/")[-1].split(".")[0]
    ####### creating artfacts
    luminance(image_path, "contrast", "bright", im_name,dest_dir2)
    luminance(image_path, "contrast", "dark", im_name,dest_dir2)
    luminance(image_path, "brightness", "bright", im_name,dest_dir3)
    luminance(image_path, "brightness", "dark", im_name, dest_dir3)
    # counter += 2
    # # ##
    spottify(image_path, im_name, "small", dest_dir1)
    spottify(image_path, im_name, "large", dest_dir4)
    # counter += 4

    blur(image_path, im_name, dest_dir3)
    # counter += 1

    elastic(image_path, im_name, dest_dir1)
    # counter += 4

    fatify(image_path, im_name, 0.5,dest_dir1)
    fatify(image_path, im_name, 1, dest_dir2)
    fatify(image_path, im_name, 1.5, dest_dir3)
    fatify(image_path, im_name, 2, dest_dir4)
    # counter += 1


    flipper(image_path, im_name, 1, dest_dir1)
    flipper(image_path, im_name, 2, dest_dir2)
    flipper(image_path, im_name, 3, dest_dir3)
    flipper(image_path, im_name, 4, dest_dir4)
    # counter += 4


    compress1(image_path, im_name, dest_dir1)
    compress2(image_path, im_name, 5, dest_dir4)
    compress2(image_path, im_name, 10, dest_dir3)
    compress2(image_path, im_name, 50, dest_dir2)
    compress2(image_path, im_name, 70, dest_dir1)
    # counter += 1

    rotator(image_path, im_name, 1, dest_dir1)
    rotator(image_path, im_name, 2, dest_dir2)
    rotator(image_path, im_name, 3, dest_dir3)
    rotator(image_path, im_name, 4, dest_dir4)
    # counter += 1

    squamous(image_path, im_name, 1, dest_dir1)
    squamous(image_path, im_name, 2, dest_dir2)
    squamous(image_path, im_name, 3, dest_dir3)
    squamous(image_path, im_name, 4, dest_dir4)
    # counter += 8


    if im_name not in finished_ids_1:
        threader(image_path, im_name, "small", dest_dir1)
        threader(image_path, im_name, "large", dest_dir4)
        # counter += 4
