import os
import json
from multiprocessing import Pool

import fire

def chunkify(fname, size=1024 * 1024 * 100):
    file_end = os.path.getsize(fname)
    with open(fname, 'rb') as f:
        chunk_end = f.tell()
        while True:
            chunkStart = chunk_end
            f.seek(size, 1)
            f.readline()
            chunk_end = f.tell()
            yield chunkStart, chunk_end - chunkStart
            if chunk_end > file_end:
                break
            
def _process(chunk_start, chunk_size, file_path, target_path):
    with open(file_path, 'rb') as f1, open(target_path, 'a+') as f2:
        f1.seek(chunk_start)
        print(chunk_size)
        # lines = f1.read(chunk_size).splitlines()
        # print(len(lines))
        # for idx, line in enumerate(lines, 1):
        #     item = json.loads(line.decode().strip())
        #     pass

            
def process(file_path, target_file_path):
    if os.path.exists(target_file_path):
        os.remove(target_file_path)
    pool = Pool()
    jobs = []
    for chunk_start, chunk_size in chunkify(file_path):
        # print(chunk_start)
        jobs.append(
            pool.apply_async(
                _process, (chunk_start, chunk_size, file_path, target_file_path)))
    pool.close()
    pool.join()
    # 需要结果
    # for job in jobs:
    #     result =  job.get()

if __name__ == "__main__":
    fire.Fire(process)