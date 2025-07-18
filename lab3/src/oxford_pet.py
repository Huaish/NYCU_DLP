import os
import torch
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
import albumentations as A

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))
        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)


        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        sample = dict(image=image, mask=mask, trimap=trimap)
        
        
        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = sample["image"]
        mask = sample["mask"]
        # image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        # mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        # trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = mask
        # sample["mask"] = np.expand_dims(mask, 0)
        # sample["trimap"] = np.expand_dims(trimap, 0)

        return {
            "image": torch.tensor(sample["image"], dtype=torch.float32),
            "mask": torch.tensor(sample["mask"], dtype=torch.long),
            # "trimap": torch.tensor(sample["trimap"], dtype=torch.float32),
        }


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def load_dataset(data_path, mode):
    # Define the data transforms
    data_transforms = {
        'train': A.Compose([
            A.RandomResizedCrop(256, 256, scale=(0.8, 1.0)),
            A.HorizontalFlip(),
            A.RandomRotate90(),
            # A.RandomBrightnessContrast(),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            A.GaussianBlur(),
            A.Normalize(),
        ]),
        'valid': A.Compose([
            A.Resize(256, 256),
            A.Normalize(),
        ]),
        'test': A.Compose([
            A.Resize(256, 256),
            A.Normalize(),
        ])
    }
    
    # Download the dataset if it doesn't exist or it is empty
    if not os.path.exists(data_path) or len(os.listdir(data_path)) == 0:
        print(f"Downloading the Oxford-IIIT Pet Dataset to {data_path}")
        OxfordPetDataset.download(data_path)
        
    # Load the dataset
    dataset = SimpleOxfordPetDataset(data_path, mode=mode, transform=data_transforms[mode])
    
    # Create a PyTorch DataLoader
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    return dataset

if __name__ == "__main__":
    data_path = "../dataset/oxford-iiit-pet"

    # Load the dataset
    train_dataset = load_dataset(data_path, "train")
    val_dataset = load_dataset(data_path, "valid")
    test_dataset = load_dataset(data_path, "test")
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")
    print(f"Test dataset: {len(test_dataset)} images")
    
    dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    for batch in dataloader:
        print(batch["image"].shape, batch["mask"].shape)
        break
    
    # Show the first image and mask
    import matplotlib.pyplot as plt
    plt.imshow(batch["image"][0].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    plt.savefig("image.png")
    
    plt.imshow(batch["mask"][0])
    plt.savefig("mask.png")