
import os
from tqdm import tqdm
from src._io.ioutils import read_image_any_format
import pandas as pd
valid_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.tif', '.svg']
base_folder = '/data/users/amolina/hmmkgv2/images/'
start_index = 21823 - 1

df = pd.read_csv('/data/users/amolina/hmmkgv2/images/downloaded_images.tsv', sep='\t')
# STEP 1: Recursively list all files under the folder if they have a valid extension
files = [os.path.join(base_folder, 'images', x)
         for x, z in
         zip(df['subpath'], df['missing'])
         if any(x.endswith(y) for y in valid_extensions) and z]

# STEP 3: Enumerate (and use tqdm) to try reading each file with `read_image_any_format`
for i, file in tqdm(enumerate(files[start_index:], start=start_index), total=len(files)-start_index):
    try:
        read_image_any_format(file)
    except Exception as e:
        print(f"Error reading file: {' / '.join(file.split('/'))} ,\n Exception: {e}, Index: {i}")
        exit()