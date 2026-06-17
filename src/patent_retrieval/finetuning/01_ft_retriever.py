#import multiprocessing as mp
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from pyrootutils import setup_root
import pandas as pd
from patent_retrieval import utils as utils, dataset as dataset, encoder as encoder



root = setup_root(__file__)
logger = utils.get_logger(__name__)
test_topics_path=Path(os.environ["CLEF_IP_LOCATION"])/ "02_topics"/ "test-pac"/ "relass_clef-ip-2011-PAC.txt"
train_topics_path=Path(os.environ["CLEF_IP_LOCATION"])/ "02_topics"/ "training-pac"/"clef-ip-2011_PACTraining"/"qrels.txt"

def load_topics(topics_path: Path) -> list[str]:
    df = pd.read_csv(
        topics_path, sep="\t", header=None, names=["topic", "candidate", "n"]
    )
    return df
train_topics = load_topics(train_topics_path)
test_topics = load_topics(test_topics_path)
patent_encoder = encoder.get_encoder(type="dense",backend="openai",store_type="faiss",model_name="Qwen/Qwen3-Embedding-4B",index_dir="/home/alm3rng/scratch/clef_ip_2011/qwen3_emb_4b_v3_title-abstract-claims")

patent_encoder.load_index(path="/home/alm3rng/scratch/clef_ip_2011/qwen3_emb_4b_v3_title-abstract-claims",store_type="faiss")
index_ids = patent_encoder.get_indices()

index_id_set = set(index_ids)
train_candidates = set(train_topics["candidate"].unique())

in_index = train_candidates & index_id_set
not_in_index = train_candidates - index_id_set

print(f"total train candidates: {len(train_candidates)}")
print(f"present in index: {len(in_index)}")
print(f"missing from index: {len(not_in_index)}")
train_candidate = train_topics[train_topics["candidate"].isin(index_id_set)]

def prepend_instruct(task_description: str="", query: str="") -> str:
    if not task_description:
            task_description = ""
    return f'Instruct: Given a patent, perform a prior art search and identify relevant existing patents. \nQuery:{query}'

results = []
for topic in tqdm(train_candidate["topic"].unique()):
    topic_file = next(
            train_topics_path.parent.glob(f"files/{topic}.xml")
    )

    topic_patent = dataset.parse_patent([topic_file])[0]
    query = prepend_instruct(query=
                dataset.extract_query_text(
                topic_patent,
                ["title", "abstract","claims"],
            ))
    search_results = patent_encoder.search(query, k=500,fetch_k=500)
    results.extend(
                {"topic": topic, "number": match_num, "score": score}
                for match_num, score in search_results
            )

results = pd.DataFrame.from_records(results)
results_file = "data/results.csv"
results.to_csv(results_file, index=False)

