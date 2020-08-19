import numpy as np
from PIL import Image

class VisImage(object):
    def __init__(self, n_classes, label_color_map):

        self.n_classes = n_classes
        self.label_color_map = label_color_map

    # Define the helper function
    def decode_segmap(image):
    
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        
        for l in range(0, self.n_classes):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
            
        rgb = np.stack([r, g, b], axis=2)

        return Image.fromarray(rgb)

    def save_img(self, img, path):
        img.save(path)