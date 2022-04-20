import json
import pickle
import time

import ray
import pandas as pd
from tqdm import tqdm


from latain.data import Block, BlockHandcraftRecord


ray.init(num_cpus=20)


def diff_filter(idx, other_idx, tag, other_tag):
    if tag or other_tag:
        return True
    return False


@ray.remote
class GBM:
    def __init__(self, model_path, dtype_path) -> None:

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        with open(dtype_path, "rb") as f:
            dtypes = pickle.load(f)

        self.model = model
        self.dtypes = dtypes

    def is_diff_file(self, xuen_dict):
        try:
            block = Block.from_dict(xuen_dict)
            flat_records, ids = BlockHandcraftRecord.from_block(block).to_flat_records(
                diff_filter
            )
            records = []
            records_span_idx = []
            for record, ((line_id1, span_idx1, _), (line_id2, span_idx2, _)) in zip(
                flat_records, ids
            ):
                if line_id1 == line_id2:
                    continue
                records.append(record)
                records_span_idx.append(((line_id1, span_idx1), (line_id2, span_idx2)))

            texts = []
            other_texts = []
            for span_id, other_span_id in records_span_idx:
                texts.append(block[int(span_id[0])][int(span_id[1])].text)
                other_texts.append(
                    block[int(other_span_id[0])][int(other_span_id[1])].text
                )
            text_df = pd.DataFrame({"text": texts, "other_text": other_texts})
            text_df["ids"] = records_span_idx

            input_df = pd.DataFrame(records)
            y = input_df.loc[:, "label"]

            text_df["label"] = y

            X = input_df.iloc[:, :-1].astype(self.dtypes)
            probs = self.model.predict_proba(X)[:, 1]

            preds = (probs > 0.5).astype("int")
            text_df["pred"] = preds
            text_df["prob"] = probs
            diff_df = text_df[text_df["pred"] != text_df["label"]]
            if len(set(diff_df["other_text"])) >= 5:
                return xuen_dict["file"]
        except Exception:
            return


def read_xuen_dicts(file_path):
    with open(file_path) as f:
        for line in f:
            yield json.loads(line)


num_workers = 20
num_prepared = 1000
num_running = 500

prepared_tasks = []
running_tasks = []
filenames = []
actor_pool = ray.util.ActorPool(
    [GBM.remote("model/lgbm.pkl", "model/dtypes.pkl") for _ in range(num_workers)]
)

start = time.time()
pbar = tqdm(total=6784)
for xuen_dict in read_xuen_dicts("data/xuen_dicts.jsonl"):
    if len(prepared_tasks) <= num_prepared:
        prepared_tasks.append(xuen_dict)
    else:
        temp_tasks = prepared_tasks[:500]
        prepared_tasks = prepared_tasks[500:]
        for i in actor_pool.map_unordered(lambda a, v: a.is_diff_file.remote(v), temp_tasks):
            filenames.append(i)
            pbar.update(1)
# filenames = ray.get(completed_tasks)
filenames = [i for i in filenames if i is not None]
end = time.time()
print(filenames[:10])
print(end - start)