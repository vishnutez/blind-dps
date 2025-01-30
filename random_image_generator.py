# Generate random images of size 256x256

import os
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm


def random_image_generator(root: str, num_images: int):
    os.makedirs(root, exist_ok=True)
    for i in tqdm(range(num_images)):
        img = np.random.rand(256, 256, 3) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img.save(os.path.join(root, f'{i}.png'))


if __name__ == '__main__':
    random_image_generator('./data/guidance_samples', 10)
    print('Random images are generated.')

