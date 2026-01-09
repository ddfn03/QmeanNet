from torch.utils.data import Dataset
import torch


class QmeanDataset(Dataset):
    def __init__(self, df):
        self.sequences = df['sequence'].tolist()
        q = df["Avg_Local_Score"].values.astype("float32")
        self.qmin = q.min()
        self.qmax = q.max()
        self.targets = 2 * (q - self.qmin) / (self.qmax - self.qmin) - 1.0

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self , idx):
        return self.sequences[idx] , torch.tensor(self.targets[idx], dtype=torch.float32)