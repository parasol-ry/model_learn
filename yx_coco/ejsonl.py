import os
import json
from pathlib import Path
from multiprocessing import Pool
from typing import Any, Callable, Iterable, List, Optional, Union, Tuple

from functools import partial

from mpire import WorkerPool
from tqdm import tqdm
import time


ItemsFunc = Callable[[List[dict]], Any]
RawStrsFunc = Callable[[List[str]], Any]


def single_task(jsonl_file: Union[Path, str], items_func: ItemsFunc, start: int, size: int):
    with open(jsonl_file, 'rb') as f:
        f.seek(start)
        f.read1(size)
        # items = [json.loads(line) for line in f.read(size).splitlines()]
        result = items_func(items)
    return result


def single_task_with_raw_str(jsonl_file: Union[Path, str], raw_str_func: RawStrsFunc, start: int, size: int):
    f = open(jsonl_file, 'rb')
    with open(jsonl_file, 'rb') as f:
        f.seek(start)
        raw_strs = list(i.decode("utf-8") for i in f.read(size).splitlines())
        # result = raw_str_func(raw_strs)
    return None


def update_pbar(pbar, *args):
    pbar.update(1)


class JsonlProcessor:

    def __init__(self, num_workers: Optional[int] = None, chunk_size: Union[str, int]="100M") -> None:
        if isinstance(chunk_size, str):
            if chunk_size.endswith("M"):
                _chunk_size = int(chunk_size[:-1]) * 1024 * 1024
            elif chunk_size.endswith("G"):
                _chunk_size = int(chunk_size[:-1]) * 1024 * 1024 * 1024
            else:
                _chunk_size = int(chunk_size)
        else:
            _chunk_size = chunk_size

        self._pool = Pool(num_workers)
        self.num_workers = num_workers
        self.chunk_size = _chunk_size

    def run(self, jsonl_file: Union[Path, str], items_func: ItemsFunc):
        results = []
        chunks = list(self.chunkify(jsonl_file))
        # pbar = tqdm(total=len(chunks), unit="chunk", desc=items_func.__name__)
        with WorkerPool(self.num_workers, keep_alive=True) as pool:
            results = pool.map(single_task, ((jsonl_file, items_func, chunk[0], chunk[1]) for chunk in chunks), iterable_len=len(chunks), progress_bar=True)

        # for start, size in chunks:
        #     task_result = self._pool.apply_async(
        #         single_task,
        #         (str(jsonl_file), items_func, start, size),
        #         callback=partial(update_pbar, pbar),
        #     )
        #     results.append(task_result)
        # self._pool.close()
        # self._pool.join()
        results = [result.get() for result in results]
        #  pbar.close()
        return results

    def run_raw_strs(self, jsonl_file: Union[Path, str], raw_strs_func: RawStrsFunc):
        results = []
        chunks = list(self.chunkify(jsonl_file))
        # with WorkerPool(self.num_workers, start_method="spawn") as pool:
        #     results = pool.map_unordered(single_task_with_raw_str, ((str(jsonl_file), raw_strs_func, chunk[0], chunk[1]) for chunk in chunks), iterable_len=len(chunks), progress_bar=True)

        pbar = tqdm(total=len(chunks), unit="chunk", desc=raw_strs_func.__name__)
        for start, size in chunks:
            task_result = self._pool.apply_async(
                single_task_with_raw_str,
                (str(jsonl_file), raw_strs_func, start, size),
                callback=partial(update_pbar, pbar),
            )
            results.append(task_result)
        self._pool.close()
        self._pool.join()
        results = [result.get() for result in results]
        pbar.close()
        return results

    def chunkify(self, jsonl_file: Union[Path, str]) -> Iterable[Tuple[int, int]]:
        file_end = os.path.getsize(jsonl_file)
        with open(jsonl_file, 'rb') as f:
            chunk_end = f.tell()
            while True:
                chunkStart = chunk_end
                f.seek(self.chunk_size, 1)
                f.readline()
                chunk_end = f.tell()
                yield chunkStart, chunk_end - chunkStart
                if chunk_end > file_end:
                    break


def count_lines(lines):
    return len(lines)


if __name__ == "__main__":
    processor = JsonlProcessor(10, "10M")
    processor.run_raw_strs("/data/medivh_data/workv2_sample.tagged.fixed.jsonl", count_lines)
