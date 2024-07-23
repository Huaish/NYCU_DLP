import torch
import numpy as np
import os

class MIBCI2aDataset(torch.utils.data.Dataset):
    def _getFeatures(self, filePath):
        # implement the getFeatures method
        """
        read all the preprocessed data from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        features = []
        for file in sorted(os.listdir(filePath)):
            feature = np.load(os.path.join(filePath, file))
            feature = np.expand_dims(feature, axis=1)
            features.append(feature)
        return np.concatenate(features)

    def _getLabels(self, filePath):
        # implement the getLabels method
        """
        read all the preprocessed labels from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        labels = []
        for file in os.listdir(filePath):
            label = np.load(os.path.join(filePath,file))
            labels.append(label)
            
        return np.concatenate(labels)

    def __init__(self, mode, method="LOSO"):
        # remember to change the file path according to different experiments
        assert mode in ['train', 'test', 'finetune']
        assert method in ['LOSO', 'SD', 'LOSOFT']
        
        if method == "LOSO":
            if mode == 'train':
                print("====================== Load LOSO Dataset (train) ======================")
                # subject dependent: ./dataset/SD_train/features/ and ./dataset/SD_train/labels/
                # leave-one-subject-out: ./dataset/LOSO_train/features/ and ./dataset/LOSO_train/labels/
                self.features = self._getFeatures(filePath='./dataset/LOSO_train/features/')
                self.labels = self._getLabels(filePath='./dataset/LOSO_train/labels/')
            if mode == 'test':
                print("====================== Load LOSO Dataset (test) ======================")
                # subject dependent: ./dataset/SD_test/features/ and ./dataset/SD_test/labels/
                # leave-one-subject-out and finetune: ./dataset/LOSO_test/features/ and ./dataset/LOSO_test/labels/
                self.features = self._getFeatures(filePath='./dataset/LOSO_test/features/')
                self.labels = self._getLabels(filePath='./dataset/LOSO_test/labels/')

        elif method == "SD":
            if mode == 'train':
                print("====================== Load SD Dataset (train) ======================")
                # subject dependent: ./dataset/SD_train/features/ and ./dataset/SD_train/labels/
                # leave-one-subject-out: ./dataset/LOSO_train/features/ and ./dataset/LOSO_train/labels/
                self.features = self._getFeatures(filePath='./dataset/SD_train/features/')
                self.labels = self._getLabels(filePath='./dataset/SD_train/labels/')
            if mode == 'test':
                print("====================== Load SD Dataset (test) ======================")
                # subject dependent: ./dataset/SD_test/features/ and ./dataset/SD_test/labels/
                # leave-one-subject-out and finetune: ./dataset/LOSO_test/features/ and ./dataset/LOSO_test/labels/
                self.features = self._getFeatures(filePath='./dataset/SD_test/features/')
                self.labels = self._getLabels(filePath='./dataset/SD_test/labels/')
        
        elif method == "LOSOFT":
            if mode == 'train':
                print("====================== Load LOSO+FT Dataset (train) ======================")
                # subject dependent: ./dataset/SD_train/features/ and ./dataset/SD_train/labels/
                # leave-one-subject-out: ./dataset/LOSO_train/features/ and ./dataset/LOSO_train/labels/
                self.features = self._getFeatures(filePath='./dataset/LOSO_train/features/')
                self.labels = self._getLabels(filePath='./dataset/LOSO_train/labels/')
            if mode == 'finetune':
                print("====================== Load LOSO+FT Dataset (finetune) ======================")
                # finetune: ./dataset/FT/features/ and ./dataset/FT/labels/
                self.features = self._getFeatures(filePath='./dataset/FT/features/')
                self.labels = self._getLabels(filePath='./dataset/FT/labels/')
            if mode == 'test':
                print("====================== Load LOSO+FT Dataset (test) ======================")
                # subject dependent: ./dataset/SD_test/features/ and ./dataset/SD_test/labels/
                # leave-one-subject-out and finetune: ./dataset/LOSO_test/features/ and ./dataset/LOSO_test/labels/
                self.features = self._getFeatures(filePath='./dataset/LOSO_test/features/')
                self.labels = self._getLabels(filePath='./dataset/LOSO_test/labels/')
        
    def __len__(self):
        # implement the len method
        return len(self.features)

    def __getitem__(self, idx):
        # implement the getitem method
        feature = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)
    
if __name__ == "__main__":
    dataset = MIBCI2aDataset(mode='test', method='SD')
    print(f"Number of samples: {len(dataset)}")
    for i in range(5):
        feature, label = dataset[i]
        print(f"Sample {i} - Feature shape: {feature.shape}, Label: {label}")