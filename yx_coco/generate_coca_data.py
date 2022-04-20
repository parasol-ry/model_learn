from itertools import islice
from typing import Iterable

import torch
import clip
from tqdm import tqdm
from datasets import load_from_disk, Sequence, Value, Features
from datasets.arrow_writer import ArrowWriter

model, preprocess = clip.load("ViT-B/32", device="cuda:1")


def get_batch(dataset, batch_size=32):
    dataset = iter(dataset)
    while batch := list(islice(dataset, batch_size)):
        yield batch


def generate_coca_record_from_coco_dataset(coco_dataset, batch_size) -> Iterable[dict]:
    for records in tqdm(get_batch(coco_dataset, batch_size), total=len(coco_dataset) // batch_size):
        try:
            images = [i["image"] for i in records]
            processed_image = [preprocess(image).unsqueeze(0).to("cuda:1") for image in images]
            batch = torch.cat(processed_image, dim=0)
            clips = model.encode_image(batch).squeeze().tolist()
            for record, clip in zip(records, clips):
                record["clip"] = clip
            yield from records
        except:
            pass


def write_records_to_file(records, file_path, features):
    writer = ArrowWriter(path=file_path, features=features)
    for record in records:
        writer.write(record)
    writer.finalize()


def main():
    coco_dataset_dict = load_from_disk("/data/qiaowei/coco2014/coco_caption_arrow/")
    coco_train_dataset, coco_valid_dataset = coco_dataset_dict.values()
    #coco_train_dataset = coco_train_dataset.select(range(10000))
    #coco_valid_dataset = coco_valid_dataset.select(range(1000))

    coca_features = Features(**coco_train_dataset.features, clip=Sequence(Value("float64")))

    write_records_to_file(
        generate_coca_record_from_coco_dataset(coco_valid_dataset, 32),
        "data/coca_valid.arrow",
        coca_features
    )

    write_records_to_file(
        generate_coca_record_from_coco_dataset(coco_train_dataset, 32),
        "data/coca_train.arrow",
        coca_features
    )


if __name__ == "__main__":
    main()
