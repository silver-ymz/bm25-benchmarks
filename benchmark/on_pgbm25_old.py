import json
import os
from pathlib import Path
import time
from typing import Optional
import warnings
from functools import partial
import subprocess
import json
from pathlib import Path
import sys
from typing import List, Dict
import multiprocessing as mp

from tqdm.auto import tqdm
import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

import psycopg2

from utils.beir import merge_cqa_dupstack

def format_beir_result_keys(beir_results):
    return {
        k.split("@")[-1]: v for k, v in beir_results.items()
    }

db_params = {
    'host': '127.0.0.1',
    'database': 'testdb',
    'user': 'silver',
    'port': '28817',
    'password': '',
}
conn = psycopg2.connect(**db_params)

def pgbm25_create(corpus_records):
    # cursor = conn.cursor()
    # cursor.execute("CREATE TABLE bench (id TEXT, contents TEXT);")
    # cursor.execute("ALTER TABLE bench ADD COLUMN embedding bm25vector;")
    # cursor.execute("CREATE INDEX bench_embedding_bm25 ON bench USING bm25 (embedding bm25_ops);")
    # cursor.executemany("INSERT INTO bench (id, contents) VALUES (%(id)s, %(contents)s);", corpus_records)
    # cursor.execute("UPDATE bench SET embedding = tokenize(contents);")
    # cursor.execute("DELETE FROM bench;")
    # cursor.executemany("INSERT INTO bench (id, contents) VALUES (%(id)s, %(contents)s);", corpus_records)
    # cursor.execute("UPDATE bench SET embedding = tokenize(contents);")
    # conn.commit()

    cursor = conn.cursor()

    # cursor.execute("CREATE TABLE bench (id TEXT, contents TEXT, embedding bm25vector);")
    # cursor.executemany("INSERT INTO bench (id, contents) VALUES (%(id)s, %(contents)s);", corpus_records)
    # # cursor.execute("SELECT create_unicode_tokenizer_and_trigger('bench_tokenizer', 'bench', 'contents', 'embedding');")
    # cursor.execute("UPDATE bench SET embedding = tokenize(contents, 'Bert');")

    # cursor.execute("SET LOCAL bm25_catalog.segment_growing_max_page_size = 10;")
    cursor.execute("CREATE TABLE bench (id TEXT, contents TEXT);")
    cursor.execute("ALTER TABLE bench ADD COLUMN embedding bm25vector;")
    cursor.execute("""
CREATE INDEX bench_embedding_bm25 ON bench USING bm25 (embedding bm25_ops)
""")
    cursor.execute("""
SELECT create_tokenizer('bench_tokenizer', $$
tokenizer = 'Unicode'
table = 'bench'
column = 'contents'
$$);
""")
    cursor.executemany("INSERT INTO bench (id, contents) VALUES (%(id)s, %(contents)s);", corpus_records)
    
    cursor.execute("UPDATE bench SET embedding = tokenize(contents, 'bench_tokenizer');")
#     cursor.execute("""
# CREATE INDEX bench_embedding_bm25 ON bench USING bm25 (embedding bm25_ops);
# """)
#     cursor.execute("""
# CREATE INDEX bench_embedding_bm25 ON bench USING bm25 (embedding bm25_ops) with (options = "
# [encode.elias_fano]
# ")
# """)
#     cursor.execute("""
# CREATE INDEX bench_embedding_bm25 ON bench USING bm25 (embedding bm25_ops) with (options = "
# [partition.variable]
# lambda = 12.0
# ");
# """)
    
    
    conn.commit()


def pgbm25_build_index():
    cursor = conn.cursor()
    cursor.execute("DROP INDEX IF EXISTS bench_embedding_bm25;")
    conn.commit()
    start = time.time()
    cursor = conn.cursor()
    cursor.execute("""
CREATE INDEX bench_embedding_bm25 ON bench USING bm25 (embedding bm25_ops)
""")
    conn.commit()
    return time.time() - start

def pgbm25_search(qids, queries_lst, top_k):
    cursor = conn.cursor()
    time_search = 0
    cursor.execute("SET LOCAL bm25_catalog.bm25_limit = {};".format(top_k))
    # cursor.execute("SET LOCAL bm25_catalog.enable_index = off;")
    hits = {}
    for qid, query in tqdm(zip(qids, queries_lst)):
        # print(query)
        t0 = time.time()
        cursor.execute(
            "SELECT id, embedding <&> to_bm25query('bench_embedding_bm25', %s, 'bench_tokenizer') AS score FROM bench ORDER BY score LIMIT %s;",
            (query, top_k)
        )
        # print(query)
        time_search += time.time() - t0
        hits[qid] = {
            str(row[0]): -row[1] for row in cursor.fetchall()
        }
        # print(hits)
    return hits, time_search

def main(dataset, save_dir="datasets", result_dir="results", n_threads=1, top_k=1000, k1=1.2, b=0.75):
    warnings.filterwarnings("ignore", category=UserWarning)

    #### Download dataset and unzip the dataset
    base_url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

    url = base_url.format(dataset)
    data_dir = Path(save_dir)
    data_path = beir.util.download_and_unzip(url, str(data_dir))
    if dataset == "cqadupstack":
            merge_cqa_dupstack(data_path)
    
    if dataset == "msmarco":
        split = "dev"
    else:
        split = "test"
    
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    
    num_docs = len(corpus)
    corpus_records = [
        {'id': key, 'contents': val['title'] + " " + val['text']} for key, val in corpus.items()
    ]
    del corpus

    queries_lst = []
    qids = []
    for key, val in queries.items():
        queries_lst.append(val)
        qids.append(key)

    # t0 = time.time()
    # pgbm25_create(corpus_records)
    # time_create = time.time() - t0

    # time_index = pgbm25_build_index()

    # print('='*50)
    # print(f"[pg_bm25.rs] Insert: {time_create:.4f}s ({num_docs / time_create:.2f}/s)")
    # print(f"[pg_bm25.rs] Index: {time_index:.4f}s ({num_docs / time_index:.2f}/s)")

    del corpus_records

    # now, run query and get score as well
    
    k_values = [1,10,100,1000]

    # pgbm25_search(qids, queries_lst, top_k)
    results, time_search = pgbm25_search(qids, queries_lst, top_k)
    print(f"[pg_bm25.rs] Query: {time_search:.4f}s ({len(queries_lst) / time_search:.2f}/s)")
    print('-'*50)

    # results, time_search = pgbm25_search(qids, queries_lst, top_k)
    # print(f"[pg_bm25.rs] Query: {time_search:.4f}s ({len(queries_lst) / time_search:.2f}/s)")
    # print('-'*50)


    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values)
    print(ndcg)
    print(recall)
    print(precision)

    # Save everything to json
    save_dict = {
        "model": "pgbm25.rs",
        "dataset": dataset,
        "k1": k1,
        "b": b,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_threads": n_threads,
        "top_k": top_k,
        "stats": {
            "num_docs": num_docs,
            "num_queries": len(queries_lst),
        },
        "timing": {
            # "insert": {"elapsed": round(time_create, 4)},
            # "index": {"elapsed": round(time_index, 4)},
            "query": {"elapsed": round(time_search, 4)},
        },
        "ndcg": format_beir_result_keys(ndcg),
        "map": format_beir_result_keys(_map),
        "recall": format_beir_result_keys(recall),
        "precision": format_beir_result_keys(precision),
    }

    result_dir = Path(result_dir) / "pg_bm25"
    result_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(result_dir) / f"{dataset}-{os.urandom(8).hex()}.json"
    with open(save_path, "w") as f:
        json.dump(save_dict, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark pgbm25.rs on a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="fiqa",
        help="Dataset to benchmark on.",
    )

    parser.add_argument(
        "-n", "--n_threads",
        type=int,
        default=1,
        help="Number of jobs to run in parallel.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=1000,
        help="Number of top-k documents to retrieve.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Directory to save results.",
    )

    parser.add_argument(
        "--k1",
        type=float,
        default=1.2,
        help="BM25 k1 parameter.",
    )

    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b parameter.",
    )

    kwargs = vars(parser.parse_args())
    main(**kwargs)
