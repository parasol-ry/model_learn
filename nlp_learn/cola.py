import copy
from datasets import load_dataset, load_metric
from datasets.dataset_dict import DatasetDict
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import torch

from torch.optim import Adam

from simpletrainer import Trainer, TrainerConfig
from simpletrainer.transformers_lr_scheduler import TransformersLRScheduler
from simpletrainer import Callback

act_task = "cola"
model_checkpoint = "bert-base-uncased"
batch = 16

dataset: DatasetDict = load_dataset("glue", act_task)
metric = load_metric("glue", act_task)

train_dataset = dataset['train']
val_dataset = dataset['validation']

# print(dataset["train"][0])
# {'sentence': "Our friends won't buy this analysis, let alone the next one we propose.", 'label': 1, 'idx': 0}

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def collate_fn(records):

    batch_encoding = tokenizer(
        [record['sentence'] for record in records],
        padding = True,
        truncation = True,
        return_tensors = 'pt',
    )
    labels = torch.tensor([record['label'] for record in records])
    batch_encoding['labels'] = labels
    return batch_encoding

train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size = batch, collate_fn = collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size = batch, collate_fn = collate_fn)

config = AutoConfig.from_pretrained(model_checkpoint, num_labels = 2)
classifier = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config = config)

optimizer = Adam(classifier.parameters(), lr = 2e-5, weight_decay=0.01)
lr_scheduler_config = TransformersLRScheduler(name = "linear", num_warmup_steps = 0.1)

class Feedback(Callback):
    def on_stage_end(self, trainer: "Trainer") -> None:
        if trainer.training: return

        records = copy.deepcopy(trainer.validation_dataloader)
        predict = torch.tensor([]).to(trainer.device)
        labels = torch.tensor([]).to(trainer.device)
        for record in records:
            record = record.to(trainer.device)
            output = trainer.model(**record)
            output_label = torch.argmax(output["logits"], dim = 1)
            predict = torch.cat((predict, output_label), 0)
            labels = torch.cat((labels, record["labels"]), 0)
        metric_scores = metric.compute(predictions = predict, references = labels)
        trainer.stage.metrics.update(metric_scores)

trainer_config = TrainerConfig(
    experiment_name = "cola_experiments",
    core_metric = '-loss',
    epochs = 5,
    terminal="rich",
    logger="tensorboard:2",
)

trainer = Trainer(
    trainer_config,
    classifier,
    optimizer,
    train_dataloader,
    val_dataloader,
    lr_scheduler = lr_scheduler_config,
    callbacks = [Feedback()],
)

trainer.train()
