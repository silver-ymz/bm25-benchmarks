import argparse
import time
from pathlib import Path
import os
import json
import gc

import psycopg
from tqdm import tqdm

import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from utils.benchmark import get_max_memory_usage, Timer
from utils.beir import merge_cqa_dupstack, clean_results_keys

BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


class PgClient:
    def __init__(self, url, k1, b, tokenizer) -> None:
        assert tokenizer == "UNICODE"

        self.client = psycopg.connect(url, autocommit=True)
        self.k1 = k1
        self.b = b

    def create(self, corpus_ids, corpus_lst, query_ids, query_lst):
        with self.client.cursor() as cursor:
            # cursor.execute("CREATE TABLE corpus (id TEXT, text TEXT)")
            cursor.execute("CREATE TABLE corpus (id TEXT, text TEXT, embedding bm25vector)")

            cursor.execute("CREATE TABLE queries (id TEXT, text TEXT)")
            with cursor.copy("COPY corpus (id, text) FROM STDIN WITH (FORMAT BINARY)") as copy:
                copy.set_types(["text", "text"])
                for i, text in tqdm(zip(corpus_ids, corpus_lst), desc="copy corpus", leave=False):
                    copy.write_row((i, text))
            with cursor.copy("COPY queries (id, text) FROM STDIN WITH (FORMAT BINARY)") as copy:
                copy.set_types(["text", "text"])
                for i, text in tqdm(zip(query_ids, query_lst), desc="copy queries", leave=False):
                    copy.write_row((i, text))
            cursor.execute("""
SELECT create_tokenizer('test_token', $$
tokenizer = 'unicode'
table = 'corpus'
column = 'text'
stopwords = 'nltk'
$$);
""")
            cursor.execute("UPDATE corpus SET embedding = tokenize(text, 'test_token');")

    def index(self):
        with self.client.cursor() as cursor:
            cursor.execute("CREATE INDEX corpus_embedding_bm25 ON corpus USING bm25 (embedding bm25_ops)")


    def query(self, topk):
        with self.client.cursor() as cursor:
            cursor.execute(f"SET LOCAL bm25_catalog.bm25_limit = {topk};")
            cursor.execute(
                f"""
                select q.id as qid, c.id, c.score from queries q, lateral (
                    select id, corpus.embedding <&> to_bm25query('corpus_embedding_bm25', q.text, 'test_token') as score from corpus order by score limit {topk}
                ) c;
                """
            )
            return cursor.fetchall()

    def remove(self):
        with self.client.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS corpus")
            cursor.execute("DROP TABLE IF EXISTS queries")
            cursor.execute("SELECT drop_tokenizer('test_token')")

CLIENT_URL="postgresql://silver:silvervectorchordbm25@localhost:5432/testdb"

def main(
    dataset,
    top_k,
    result_dir,
    save_dir,
    k1,
    b,
    tokenizer,
    remove,
    skip_index,
):
    if remove:
        client = PgClient(CLIENT_URL, k1, b, tokenizer)
        client.remove()
        return

    data_path = beir.util.download_and_unzip(BASE_URL.format(dataset), save_dir)

    if dataset == "cqadupstack":
        merge_cqa_dupstack(data_path)

    if dataset == "msmarco":
        split = "dev"
    else:
        split = "test"

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    num_docs = len(corpus)
    num_queries = len(queries)

    corpus_ids, corpus_lst = [], []
    for key, val in corpus.items():
        corpus_ids.append(key)
        corpus_item = val["title"] + " " + val["text"]
        corpus_item = corpus_item.replace("\u0000", "")
        corpus_lst.append(corpus_item)

    del corpus

    qids, queries_lst = [], []
    for key, val in queries.items():
        qids.append(key)
        queries_lst.append(val)

    print("=" * 50)
    print("Dataset: ", dataset)
    print(f"Corpus Size: {num_docs:,}")
    print(f"Queries Size: {num_queries:,}")

    client = PgClient(CLIENT_URL, k1, b, tokenizer)

    timer = Timer("[pgbm25.rs]")
    if not skip_index:
        t_insert = timer.start("Insert")
        client.create(corpus_ids, corpus_lst, qids, queries_lst)
        timer.stop(t_insert, show=True, n_total=num_docs)

        del corpus_lst
        del corpus_ids
        del qids
        del queries_lst
        gc.collect()

        t_index = timer.start("Index")
        client.index()
        timer.stop(t_index, show=True, n_total=num_docs)
    else:
        del corpus_lst
        del corpus_ids
        del qids
        del queries_lst
        gc.collect()

    t_query = timer.start("Query")
    results = client.query(top_k)
    timer.stop(t_query, show=True, n_total=num_queries)

    format_results = {}
    for qid, cid, score in results:
        key = str(qid)
        if key not in format_results:
            format_results[key] = {}
        # use negative score since it's the negative dot production
        format_results[key][str(cid)] = -float(score)

    with open("results/pgvectors-without-index.json", "w") as f:
        json.dump(format_results, f, indent=2)

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, format_results, [1, 10, 100, 1000]
    )
    max_mem_gb = get_max_memory_usage("GB")

    print("=" * 50)
    print(f"Max Memory Usage: {max_mem_gb:.4f} GB")
    print("-" * 50)
    print(ndcg)
    print(recall)
    print("=" * 50)

    save_dict = {
        "model": "pgbm25.rs",
        "dataset": dataset,
        "k1": k1,
        "b": b,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "top_k": top_k,
        "max_mem_gb": max_mem_gb,
        "stats": {
            "num_docs": num_docs,
            "num_queries": num_queries,
        },
        "timing": timer.to_dict(underscore=True, lowercase=True),
        "scores": {
            "ndcg": clean_results_keys(ndcg),
            "map": clean_results_keys(_map),
            "recall": clean_results_keys(recall),
            "precision": clean_results_keys(precision),
        },
    }

    result_dir = Path(result_dir) / save_dict["model"]
    result_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(result_dir) / f"{dataset}-{hex(int(time.time()))}-{os.urandom(4).hex()}.json"
    with open(save_path, "w") as f:
        json.dump(save_dict, f, indent=2)


def build_argument():
    parser = argparse.ArgumentParser(
        description="Benchmark with pgbm25.rs.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="fiqa",
    )
    parser.add_argument(
        "-n",
        "--num_runs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--profile",
        action="store_true",
    )
    parser.add_argument(
        "--result_dir",
        default="results",
    )
    parser.add_argument(
        "--save_dir",
        default="datasets",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        default="UNICODE",
        choices=["BERT", "WORD", "TOCKEN", "UNICODE"],
    )
    parser.add_argument(
        "-r",
        "--remove",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--skip_index",
        action="store_true",
    )
    return parser

if __name__ == "__main__":
    parser = build_argument()
    kwargs = vars(parser.parse_args())
    print(kwargs)
    profile = kwargs.pop("profile")
    num_runs = kwargs.pop("num_runs")

    if profile:
        import cProfile
        import pstats

        if num_runs > 1:
            raise ValueError("Cannot profile with multiple runs.")

        cProfile.run("main(**kwargs)", filename="pg.prof")
        p = pstats.Stats("pg.prof")
        p.sort_stats("time").print_stats(50)
    else:
        for _ in range(num_runs):
            main(**kwargs)
