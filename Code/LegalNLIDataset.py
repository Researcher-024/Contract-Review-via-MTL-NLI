import torch
from torch.utils.data import Dataset

class LegalNLIDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]

        return {
            "input_ids": torch.tensor(x["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(x["attention_mask"], dtype=torch.long),
            "start_positions": torch.tensor(x["start_positions"], dtype=torch.long),
            "end_positions": torch.tensor(x["end_positions"], dtype=torch.long),
            "labels": torch.tensor(x["labels"], dtype=torch.long),
            "target_evidence_representation": x["target_evidence_representation"].clone().requires_grad_(True)
        }
