from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os


def get_datasets(data_folder: str, batch_size: int = 64):
    dataloader_a = create_dataloader(os.path.join(data_folder, "testA"), batch_size=batch_size)
    dataloader_b = create_dataloader(os.path.join(data_folder, "testB"), batch_size=batch_size)

    return dataloader_a, dataloader_b


def create_dataloader(root: str, img_size: int = 128, batch_size: int = 64) -> DataLoader:
    dataset = ImageFolder(
        root,
        transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))

    return DataLoader(dataset, batch_size, True)
