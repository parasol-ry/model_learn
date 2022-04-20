from dataclasses import dataclass, field
from pathlib import Path

from torch.utils.data import DataLoader
from torch.optim import Adam
from datasets import load_from_disk, Dataset, load_dataset
from transformers import AutoTokenizer
from simpletrainer import Trainer, TrainerConfig

from coca.data import CocaCollator
from coca.caption_model import CaptionModel


@dataclass
class CocaConfig:
    coca_data: Path
    batch_size: int = 32
    num_prefix: int = 8
    frozen: bool = True
    lr: float = 1e-3
    trainer: dict = field(default_factory=dict)
    cuda: int = 0


def main(config: CocaConfig):

    # Dataset
    # train_dataset = load_dataset("./train_coco/coca_train.arrow")
    train_dataset = Dataset.from_file("data/coca_train.arrow")
    valid_dataset = Dataset.from_file("data/coca_valid.arrow")
    # valid_dataset = load_from_disk(str(config.coca_data)).values()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    collator = CocaCollator(tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, collate_fn=collator)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, collate_fn=collator)
    
    for x in train_dataloader:
        print(x['input_ids'].shape)
        print(x['attention_mask'].shape)
        print("++++++++++++++++++")
        break

    # Model
    # model = CaptionModel(config.num_prefix, config.frozen)

    # # Optimizer
    # optimizer = Adam(model.parameters(), lr=config.lr)

    # # Trainer
    # trainer_config = TrainerConfig(**config.trainer)
    # trainer = Trainer(
    #     trainer_config,
    #     model,
    #     optimizer,
    #     train_dataloader,
    #     valid_dataloader
    # )
    # trainer.train()


if __name__ == "__main__":
    config = CocaConfig("data/coca_demo_arrow", trainer={"experiment_name": "coca", "terminal": "rich:0"}, cuda=1)
    import torch
    torch.cuda.set_device(torch.cuda.device(config.cuda))
    main(config)
