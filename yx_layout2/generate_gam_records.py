import json
from pathlib import Path
from typing import Iterable

import typer
import pandas as pd
from tqdm import tqdm
from datasets.arrow_writer import ArrowWriter

from latain.core import Block
from latain.data import BlockHandcraftRecord


def base_filter(idx, other_idx, tag, other_tag):

    if tag or other_tag:
        return True

    if other_idx - idx <= 2:
        return True

    return False


def save_gbm_dataframe(df: pd.DataFrame, file_path: str):
    col_types = df.infer_objects().dtypes.to_dict()
    new_col_types = {}
    for k, v in col_types.items():
        if v.name == "object":
            new_col_types[k] = pd.CategoricalDtype(["same", "different"])
        else:
            new_col_types[k] = v
    df = df.astype(new_col_types)
    df.to_pickle(file_path)


def gbm_dataframe_from_records(records):
    _df = pd.DataFrame(records[:1000])
    col_types = _df.infer_objects().dtypes.to_dict()
    new_col_types = {}
    for k, v in col_types.items():
        if v.name == "object":
            new_col_types[k] = pd.CategoricalDtype(["same", "different"])
        else:
            new_col_types[k] = v
    df = pd.DataFrame(records).astype(new_col_types)
    return df


def infer_types(df: pd.DataFrame):
    col_types = df.infer_objects().dtypes.to_dict()
    new_col_types = {}
    for k, v in col_types.items():
        if v.name == "object":
            new_col_types[k] = pd.CategoricalDtype(["same", "different"])
        else:
            new_col_types[k] = v
    return new_col_types


# def gbm_dataframe_from_xuen_dicts(xuen_dicts):
#     _first_block_records = BlockHandcraftRecord.from_block(Block.from_dict(xuen_dicts[0])).to_flat_records()
#     df = pd.DataFrame(_first_block_records)
#     new_col_types = infer_types(df)
#     df = df.astype(new_col_types)

#     for raw_dict in tqdm(xuen_dicts[1:]):
#         try:
#             block = Block.from_dict(raw_dict)
#             flat_records = BlockHandcraftRecord.from_block(block).to_flat_records()
#             new_df = pd.DataFrame(flat_records).astype(new_col_types)
#             df = pd.concat([df, new_df])
#         except:
#             pass
#     df = df.reset_index(drop=True)
#     return df


def generate_gbm_record_from_xuen_dicts(
    xuen_dicts: Iterable[dict], filter_=None
) -> Iterable[dict]:
    err_count = 0
    for raw_dict in tqdm(xuen_dicts):
        try:
            block = Block.from_dict(raw_dict)
            flat_records, _ = BlockHandcraftRecord.from_block(block).to_flat_records(
                filter_
            )
            yield from flat_records
        except Exception:
            err_count += 1
            pass
    print(f"{err_count=}")


def generate_xuen_dict_from_jsonl_file(jsonl_file: Path) -> Iterable[dict]:
    with open(jsonl_file) as f:
        for line in f:
            yield json.loads(line)


def save_records_to_arrow(records: Iterable[dict], save_file: Path):
    writer = ArrowWriter(path=str(save_file))
    for record in records:
        writer.write(record)
    writer.finalize()


def main(
    input_file: Path = typer.Argument(None, exists=True, dir_okay=False, readable=True),
    output_file: Path = typer.Argument(
        None, exists=False, dir_okay=False, writable=True
    ),
):
    # df = gbm_dataframe_from_xuen_dicts(xuen_dicts)
    # df.to_feather(str(output_file))
    xuen_dicts = generate_xuen_dict_from_jsonl_file(input_file)
    records = generate_gbm_record_from_xuen_dicts(xuen_dicts, base_filter)
    save_records_to_arrow(records, output_file)


if __name__ == "__main__":
    typer.run(main)
