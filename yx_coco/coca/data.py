import torch


class CocaCollator:

    def __init__(self, tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        self.max_length = max_length
    
    def __call__(self, records):
        texts = [record["caption"] for record in records]
        image_feature = torch.tensor([record["clip"] for record in records], dtype=torch.float)
        encodes = self.tokenizer(texts, padding=True, max_length=self.max_length, truncation=True, return_tensors="pt")
        encodes["image_feature"] = image_feature
        return encodes
