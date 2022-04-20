from datasets import Dataset
from simpletrainer import Trainer, TrainerConfig, Callback
from transformers import AutoTokenizer
from model import ClipGPT2Model
from torch.optim import Adam
import model_config
from torch.utils.data import DataLoader
from simpletrainer.transformers_lr_scheduler import TransformersLRScheduler
import torch
from dataset import CocaCollator

config = model_config.config
model = ClipGPT2Model(config['prefix_length'], config['clip_length'])

# weights_path = config['weights_path']
# CPU = torch.device("cpu")
# model.load_state_dict(torch.load(weights_path, map_location=CPU))
# Dataset
train_dataset = Dataset.from_file("/home/renyi/workplace/ai-arcane/CLIP+GPT2/data/coca_train.arrow")
# valid_dataset = Dataset.from_file("./data/coca_valid.arrow")
# valid_dataset = load_from_disk(str(config.coca_data)).values()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
collator = CocaCollator(tokenizer)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config["batch_size"], collate_fn=collator)
# valid_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"], collate_fn=collator)

# id = 0
# for x in train_dataloader:
#     # if id == 3 : break
#     # id += 1
#     print(x['input_ids'].shape)
#     print(x['attention_mask'].shape)
#     print("++++++++++++++++++")
#     break

optimizer = Adam(params = model.parameters(), lr = config['lr'])

trainer_config = TrainerConfig(
    experiment_name = 'clip+gpt2_end',
    core_metric = '-loss',
    epochs = config['epoch'],
    terminal = 'rich:0',
    logger = "tensorboard:2",
)
lr_scheduler_config = TransformersLRScheduler(name="linear", num_warmup_steps=5000)

trainer = Trainer(
    trainer_config,
    model,
    optimizer,
    train_dataloader,
    # callbacks = [Feedback()],
    lr_scheduler = lr_scheduler_config,
)

trainer.train()