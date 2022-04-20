from torch.utils.data import DataLoader
from datasets import Dataset

train_dataset = Dataset.from_file("data/coca_train.arrow")
valid_dataset = Dataset.from_file("data/coca_valid.arrow")

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, collate_fn=collator)  # type: ignore
valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, collate_fn=collator)  # type: ignore
    