#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import clip
from model import ClipGPT2Model
from transformers import AutoTokenizer
from model_config import config


# In[2]:


device = torch.device('cuda:0')
# _, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
weights_path = config['weights_path']
CPU = torch.device("cpu")

model = ClipGPT2Model(config['prefix_length'], config['clip_length'])
model.load_state_dict(torch.load(weights_path, map_location=CPU))
model = model.eval()
model = model.to(device)


# In[3]:


def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


# In[4]:


from datasets import Dataset

valid_dataset = Dataset.from_file("./data/coca_valid.arrow")


# In[5]:


# for x in valid_dataset:
#     print(x)
#     break


# In[6]:


import json

with open("/data/qiaowei/coco2014/annotations/captions_val2014.json") as f:
    data = json.load(f)


# In[7]:


# data


# In[8]:


import numpy as np

df = valid_dataset.to_pandas()


# In[9]:


# df


# In[10]:


for image in df['image']:
    print(image['path'])
    print(image['path'].split('/')[-1])
    break


# In[11]:


import re

def str_to_id(image_name):
    return int(re.findall("COCO_val2014_0*([1-9]\d*).jpg", image_name)[0])


# In[12]:


id_clip_map = {}
for image, clip in zip(df["image"], df["clip"]):
    id = str_to_id(image["path"].split("/")[-1])
    id_clip_map[id] = clip
    


# In[14]:


from tqdm import tqdm

id_caption_map = {}
for id_, clip in tqdm(list(id_clip_map.items())):
    with torch.no_grad():
        prefix = torch.tensor(clip, dtype=torch.float32, device = device).unsqueeze(0)
        # print(prefix.shape)
        prefix_embed = model.clip_project(prefix).reshape(1, config['prefix_length'], -1)
    generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
    # print(generated_text_prefix)
    id_caption_map[id_] = generated_text_prefix


# In[ ]:


with open("predict_end.json", "w") as f:
    json.dump(id_caption_map, f)

