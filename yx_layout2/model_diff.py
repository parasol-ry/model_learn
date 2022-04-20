import json
import pandas as pd

from latain.data import Block, BlockHandcraftRecord


def diff_filter(idx, other_idx, tag, other_tag):
    if tag or other_tag:
        return True
    return False


def load_model_and_dtypes(save_dir):
    import pickle

    with open(f"{save_dir}/lgbm.pkl", "rb") as f:
        model = pickle.load(f)

    with open(f"{save_dir}/dtypes.pkl", "rb") as f:
        dtypes = pickle.load(f)
    return model, dtypes


def read_xuen_dicts(file_path):
    with open(file_path) as f:
        for line in f:
            yield (json.loads(line),)


def get_diff_dataframe(xuen_dict):
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
            other_texts.append(block[int(other_span_id[0])][int(other_span_id[1])].text)
        text_df = pd.DataFrame({"text": texts, "other_text": other_texts})
        text_df["ids"] = records_span_idx

        input_df = pd.DataFrame(records)
        y = input_df.loc[:, "label"]
        text_df["label"] = y
        return input_df, text_df, xuen_dict["file"]
    except Exception:
        return None, None, None


def worker_init(worker_state):
    model, dtypes = load_model_and_dtypes("model")
    # df = pd.read_csv("/data/medivh_data/las.sample.part-1.meta.v2.tsv", sep="\t")
    # white_list = [name for name, source in zip(df["file"], df["source"]) if isinstance(source, str)]
    worker_state["model"] = model
    worker_state["dtypes"] = dtypes
    # worker_state["white_list"] = set(white_list)


def mpire_func(worker_state, xuen_dict):
    try:
        # if xuen_dict["file"] in worker_state["white_list"]:
        #     return
        model, dtypes = worker_state["model"], worker_state["dtypes"]
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
            other_texts.append(block[int(other_span_id[0])][int(other_span_id[1])].text)
        text_df = pd.DataFrame({"text": texts, "other_text": other_texts})
        text_df["ids"] = records_span_idx

        input_df = pd.DataFrame(records)
        y = input_df.loc[:, "label"]

        text_df["label"] = y

        probs = predict(input_df, model, dtypes)
        preds = (probs > 0.5).astype("int")
        text_df["pred"] = preds
        text_df["prob"] = probs
        diff_df = text_df[text_df["pred"] != text_df["label"]]
        if len(set(diff_df["other_text"])) >= 5:
            return xuen_dict["file"]
    except KeyError as e:
        return None
    except Exception:
        return None
    return None


def predict(input_dataframe, model, dtypes):
    X = input_dataframe.iloc[:, :-1].astype(dtypes)
    probs = model.predict_proba(X)[:, 1]
    return probs


# if __name__ == "__main__":
from mpire import WorkerPool
from tqdm import tqdm

xuen_file_path = "/data/medivh_data/workv2_sample.tagged.fixed.jsonl"
# xuen_file_path = "data/xuen_dicts.jsonl"
xuen_dicts = tqdm(read_xuen_dicts(xuen_file_path))

print("start...")
filenames = []
with WorkerPool(4, use_worker_state=True, keep_alive=True) as pool:
    async_results = pool.imap_unordered(
        mpire_func,
        xuen_dicts,
        iterable_len=10000,
        # iterable_len=5000,
        progress_bar=True,
        worker_init=worker_init,
        max_tasks_active=1000,
    )
    for filename in async_results:
        if filename is not None:
            filenames.append(filename)

with open("data/diff_xuen_dicts.txt", "w") as f:
    for filename in filenames:
        f.write(f"{filename}\n")
