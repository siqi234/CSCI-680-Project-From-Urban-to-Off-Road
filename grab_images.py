import os
import pandas as pd

img_dir = 'OFRD/train/image_data'
csv_file = 'Unlabeled/unlabeled_pool.csv'

exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

files = sorted(os.listdir(img_dir))
img_files = [f for f in files if os.path.splitext(f)[1].lower() in exts] 

df = pd.DataFrame(img_files, columns=['image_path'])

df_selected = df.iloc[::2].copy()
df_selected['image_path'] = df_selected['image_path'].apply(lambda x: os.path.join(img_dir, x).replace('\\', '/'))

df_selected.to_csv(csv_file, index=False)

print(f"Saved {len(df_selected)} image paths to {csv_file}")