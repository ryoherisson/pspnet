import numpy as np

class VisImage(object):
    def __init__(self, n_classes, label_color_map):

        self.n_classes = n_classes
        self.label_color_map = label_color_map

    # Define the helper function
    def decode_segmap(image, nc=21):
    
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        
        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
            
        rgb = np.stack([r, g, b], axis=2)
        return rgb