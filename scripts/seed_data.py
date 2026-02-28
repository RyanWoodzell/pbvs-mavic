import numpy as np
from PIL import Image
import os

def create_dummy_image(path, mode='RGB', size=(256, 256)):
    if mode == 'RGB':
        data = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    else:
        data = np.random.randint(0, 256, (size[1], size[0]), dtype=np.uint8)
    img = Image.fromarray(data)
    img.save(path)

# Create train images
for cls in ['class_1', 'class_2']:
    for i in range(5):
        create_dummy_image(f'datasets/train/sar/{cls}/img_{i}.png', mode='L')
        create_dummy_image(f'datasets/train/eo/{cls}/img_{i}.png', mode='RGB')

# Create val images
for i in range(10):
    create_dummy_image(f'datasets/val/img_{i}.png', mode='L')

# Create val metadata
import pandas as pd
val_data = {
    'image_id': [f'img_{i}' for i in range(10)],
    'class': ['class_1']*5 + ['class_2']*3 + ['unknown']*2,
    'OOD_flag': [0]*8 + [1]*2
}
pd.DataFrame(val_data).to_csv('datasets/val_metadata.csv', index=False)
