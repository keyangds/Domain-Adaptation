from torch.utils.data import Dataset, DataLoader

class ImputationDataset(Dataset):
    def __init__(self, data, features, mask):
        """
        data: Tensor containing the transformed training data
        features: Tensor containing the transformed features
        mask: Tensor indicating missing values (1 for missing, 0 for present)
        """
        self.data = data.float()
        self.features = features.float()
        self.mask = mask

    def __len__(self):
        return len(self.data) - 24 + 1  # Adjust length to account for sequence length

    def __getitem__(self, idx):
        return {
            'x': self.data[idx:idx+24].float(),  # Extract sequence of length 24
            'y': self.features[idx:idx+24].float(),  # Target is the feature at the end of the sequence
            'mask': self.mask[idx:idx+24]  # Mask for the entire sequence
        }