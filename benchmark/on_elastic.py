import json
import os
from pathlib import Path
import time

import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm.auto import tqdm

from elasticsearch8 import Elasticsearch
from elasticsearch8.helpers import streaming_bulk

from typing import Dict

from utils.benchmark import get_max_memory_usage, Timer
from utils.beir import merge_cqa_dupstack, clean_results_keys


class ElasticSearchClient:
    def __init__(
        self,
        index_name,
        host="http://localhost:9200",
        request_timeout=100,
        retry_on_timeout=True,
        connections_per_node=24,
        language="english",
    ):
        self.es = Elasticsearch(
            host,
            request_timeout=request_timeout,
            retry_on_timeout=retry_on_timeout,
            connections_per_node=connections_per_node,
        )
        self.index_name = index_name
        self.language = language

    def create_index(self, k1=1.2, b=0.75):
        mappings = {
            "properties": {
                "text": {"type": "text", "analyzer": self.language},
            }
        }
        settings = {
            "index": {
                "similarity": {
                    "default": {
                        "type": "BM25",
                        "k1": k1,
                        "b": b,
                    }
                },
                "refresh_interval": "-1",
            },
            "analysis": {
                "analyzer": {
                    "custom_analyzer": {
                        "type": "standard",
                        "max_token_length": 1_000_000,
                        "stopwords": "_english_",
                        "filter": ["lowercase", "custom_snowball"],
                    }
                },
                "filter": {
                    "custom_snowball": {"type": "snowball", "language": "English"}
                },
            },
        }
        self.es.indices.create(
            index=self.index_name, mappings=mappings, settings=settings
        )

    def remove_index(self):
        self.es.indices.delete(index=self.index_name, ignore_unavailable=True)

    def index(self, corpus: Dict[str, Dict[str, str]]):
        def gendata():
            for idx, body in corpus.items():
                text = body["title"] + " " + body["text"]
                yield {"_index": self.index_name, "_id": idx, "text": text}

        progress = tqdm(unit="docs", total=len(corpus))
        for ok, result in streaming_bulk(self.es, gendata(), index=self.index_name):
            progress.update(1)
        progress.close()

        self.es.indices.refresh(index=self.index_name)

    def query(self, queries: Dict[str, str], top_k=1000) -> Dict[str, Dict[str, float]]:
        results = {}
        for query_id, query in tqdm(queries.items(), desc="Query"):
            body = {
                "query": {
                    "match": {"text": query},
                },
                "_source": False,
                "size": top_k,
            }
            response = self.es.search(index=self.index_name, body=body)
            hits = response["hits"]["hits"]
            results[query_id] = {hit["_id"]: hit["_score"] for hit in hits}
        return results


def main(
    dataset,
    n_threads=1,
    top_k=1000,
    save_dir="datasets",
    result_dir="results",
    host="http://localhost:9200",
    k1=1.2,
    b=0.75,
    skip_index=False,
):
    #### Download dataset and unzip the dataset
    base_url = (
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
    )
    data_path = beir.util.download_and_unzip(base_url.format(dataset), save_dir)

    if dataset == "msmarco":
        split = "dev"
    else:
        split = "test"

    if dataset == "cqadupstack":
        merge_cqa_dupstack(data_path)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    num_docs = len(corpus)
    num_queries = len(queries)

    print("=" * 50)
    print("Dataset: ", dataset)
    print(f"Corpus Size: {num_docs:,}")
    print(f"Queries Size: {num_queries:,}")
    timer = Timer("[Elastic-BM25]")

    client = ElasticSearchClient(index_name=dataset, host=host)

    if not skip_index:
        client.remove_index()
        client.create_index(k1=k1, b=b)

        t = timer.start("Index")
        client.index(corpus)
        timer.stop(t, show=True, n_total=num_docs)
    del corpus

    t_query = timer.start("Query")
    results = client.query(queries, top_k)
    timer.stop(t_query, show=True, n_total=num_queries)
    del queries

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, [1, 10, 100, 1000]
    )

    max_mem_gb = get_max_memory_usage("GB")

    print("=" * 50)
    print(f"Max Memory Usage: {max_mem_gb:.4f} GB")
    print("-" * 50)
    print(ndcg)
    print(recall)
    print("=" * 50)

    # Save everything to json
    save_dict = {
        "model": "elastic-bm25",
        "dataset": dataset,
        "stemmer": "snowball",
        "tokenizer": "elastic",
        "k1": k1,
        "b": b,
        "method": "bm25",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_threads": n_threads,
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
    save_path = Path(result_dir) / f"{dataset}-{os.urandom(8).hex()}.json"
    with open(save_path, "w") as f:
        json.dump(save_dict, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark rank-bm25 on a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="fiqa",
        help="Dataset to benchmark on.",
    )

    parser.add_argument(
        "-t",
        "--n_threads",
        type=int,
        default=1,
        help="Number of threads to run in parallel.",
    )

    parser.add_argument(
        "-n", "--num_runs", type=int, default=1, help="Number of runs to repeat main."
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=1000,
        help="Number of top-k documents to retrieve.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="datasets",
        help="Directory to save datasets.",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:9200",
        help="Host of the ElasticSearch server.",
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

    parser.add_argument(
        "-s",
        "--skip_index",
        action="store_true",
        help="Skip indexing the corpus.",
    )

    kwargs = vars(parser.parse_args())
    profile = kwargs.pop("profile")
    num_runs = kwargs.pop("num_runs")

    if profile:
        import cProfile
        import pstats

        if num_runs > 1:
            raise ValueError("Cannot profile with multiple runs.")

        cProfile.run("main(**kwargs)", filename="rankbm25.prof")
        p = pstats.Stats("rankbm25.prof")
        p.sort_stats("time").print_stats(50)
    else:
        for _ in range(num_runs):
            main(**kwargs)
