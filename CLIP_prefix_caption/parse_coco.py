import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse


def main(clip_model_type: str):
    device = torch.device('cuda:1')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"/data/usr/renyi/image-caption/oscar_split_{clip_model_name}_train.pkl"
    out_path_test = f"/data/usr/renyi/image-caption/oscar_split_{clip_model_name}_test.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open('/data/usr/renyi/image-caption/train_caption.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    all_embeddings_test = []
    all_captions_test = []
    idx = 0
    for i in tqdm(range(len(data))):
        sign = 0
        d = data[i]
        img_id = d["image_id"]
        filename = f"/data/qiaowei/coco2014/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            filename = f"/data/qiaowei/coco2014/val2014/COCO_val2014_{int(img_id):012d}.jpg"
            sign = 1
        if sign == 0:
            continue
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        d["clip_embedding"] = idx
        idx = idx + 1
        if sign == 0:
            all_embeddings.append(prefix)
            all_captions.append(d)
        else :
            all_embeddings_test.append(prefix)
            all_captions_test.append(d)
        
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    # with open(out_path_test, 'wb') as f:
        # pickle.dump({"clip_embedding": torch.cat(all_embeddings_test, dim=0), "captions": all_captions_test}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
