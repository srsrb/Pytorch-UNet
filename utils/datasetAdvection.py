import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms

def read_flow(file_path):
    """Lit un fichier .flo et retourne un tenseur (u, v)."""
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:  # Vérification du fichier .flo
            raise ValueError("Magic number incorrect in .flo file")
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
        data = np.resize(data, (h, w, 2))
    return torch.from_numpy(data).permute(2, 0, 1)  # Retourne (2, H, W)

class FlyingChairsDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        """
        Args:
            dataset_dir (str): Chemin vers le répertoire contenant les fichiers FlyingChairs.
            transform (callable, optional): Transformations à appliquer sur les images.
        """
        self.dataset_dir = dataset_dir
        self.transform = transform

        # Récupère tous les fichiers disponibles et trie pour assurer l'ordre
        self.image1_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith("_img1.ppm")])
        self.image2_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith("_img2.ppm")])
        self.flow_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith("_flow.flo")])

    def __len__(self):
        return len(self.flow_files)

    def __getitem__(self, idx):
        # Chemins des fichiers
        img1_path = os.path.join(self.dataset_dir, self.image1_files[idx])
        img2_path = os.path.join(self.dataset_dir, self.image2_files[idx])
        flow_path = os.path.join(self.dataset_dir, self.flow_files[idx])

        # Chargement des données
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        flow = read_flow(flow_path)

        # Transformations
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {'I1': img1, 'I2': img2, 'flow': flow}
