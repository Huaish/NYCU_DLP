import torch
from torch.utils.data import Dataset as torchData
from torchvision import transforms
import os
import json
from torchvision.datasets.folder import default_loader as imgloader

class LoadTrainData(torchData):
    """Training Dataset Loader
    
    Args:
        root: Dataset Path
        train_json: JSON file containing image paths and labels
        object_json: JSON file containing object labels
    """
    def __init__(self, root, train_json, object_json):
        super().__init__()
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        with open(train_json, 'r') as f:
            data = json.load(f)
            self.image_path, self.labels = zip(*data.items())
            
        with open(object_json, 'r') as f:
            self.label_map = json.load(f)
            
    def __len__(self):
        return len(self.labels)
    
    @property
    def info(self):
        return f"\nNumber of Training Data: {len(self.labels)}"
    
    def __getitem__(self, index):
        # Load and transform image
        img_path = os.path.join(self.root, self.image_path[index])
        img = self.transform(imgloader(img_path))
            
        # Convert labels to a multi-hot tensor
        label_tensor = torch.zeros(len(self.label_map))
        for label in self.labels[index]:
            label_tensor[self.label_map[label]] = 1
            
        return img, label_tensor
    
class LoadTestData(torchData):
    """Test Dataset Loader
    
    Args:
        root: Dataset Path
        test_json: JSON file containing labels
        object_json: JSON file containing object labels
    """
    def __init__(self, root, test_json, object_json):
        super().__init__()
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        with open(test_json, 'r') as f:
            self.labels = json.load(f)
        
        with open(object_json, 'r') as f:
            self.label_map = json.load(f)
            
    def __len__(self):
        return len(self.labels)
    
    @property
    def info(self):
        return f"\nNumber of Test Data: {len(self.labels)}"
    
    def __getitem__(self, index):
        # Convert labels to a multi-hot tensor
        label_tensor = torch.zeros(len(self.label_map))
        for label in self.labels[index]:
            label_tensor[self.label_map[label]] = 1
            
        return label_tensor
    
if __name__ == '__main__':
    # Test the train dataset loader
    train_dataset = LoadTrainData('../iclevr', 'train.json', 'objects.json')
    print(train_dataset.info)
    img, label = train_dataset[0]
    print(img.shape, label)
    
    # Test the test dataset loader
    test_dataset = LoadTestData('../iclevr', 'test.json', 'objects.json')
    print(test_dataset.info)
    label = test_dataset[0]
    print(label)
        