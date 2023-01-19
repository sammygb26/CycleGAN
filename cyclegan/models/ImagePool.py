import random
import torch


class ImagePool:
    def __init__(self, pool_size=0):
        self.pool_size = pool_size
        self.num_images = 0
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            if self.num_images < self.pool_size:
                self.num_images += 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0.0, 1.0)
                if p > 0.5:
                    rand_i = random.randint(0, len(self.images) - 1)
                    return_images.append(self.images[rand_i])
                    self.images[rand_i] = image
                else:
                    return_images.append(image)

        return torch.cat(return_images)
