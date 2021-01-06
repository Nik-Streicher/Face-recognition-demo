import os

from torch.utils.data import DataLoader
from torchvision import datasets


class Loader:
    def __init__(self, path):

        self.workers = 0 if os.name == 'nt' else 4
        self.dataset = datasets.ImageFolder(path)

    def load(self, mtcnn, resnet, users):

        def collate_fn(x):
            return x[0]

        self.dataset.idx_to_class = {i: c for c, i in self.dataset.class_to_idx.items()}
        loader = DataLoader(self.dataset, collate_fn=collate_fn, num_workers=self.workers)

        for x, y in loader:
            x_aligned = mtcnn(x)

            if x_aligned is not None:
                users.append([self.dataset.idx_to_class[y], resnet(x_aligned)])
        return users
