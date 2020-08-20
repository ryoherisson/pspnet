import numpy as np
from PIL import Image

class VisImage(object):
    def __init__(self, n_classes, label_color_map):

        self.n_classes = n_classes
        self.label_color_map = label_color_map

    # Define the helper function
    def decode_segmap(self, image):
        image = np.array(image)
    
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        
        for l in range(0, self.n_classes):
            idx = image == l
            
            r[idx] = self.label_color_map[l][0]
            g[idx] = self.label_color_map[l][1]
            b[idx] = self.label_color_map[l][2]
            
        rgb = np.stack([r, g, b], axis=2)

        return Image.fromarray(rgb)

    def save_img(self, img, path):
        img.save(path)