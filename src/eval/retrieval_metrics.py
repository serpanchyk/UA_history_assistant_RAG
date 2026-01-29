import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Callable, Any


def calculate_hit(expected: str, found: list[str]) -> int:
    return 1 if expected in found else 0


def calculate_reciprocal_rank(expected: str, found: list[str]) -> float:
    if expected in found:
        return 1 / (found.index(expected) + 1)
    return 0.0


def parse_results(results: list[Any], is_image: bool) -> list[str]:
    parsed = []
    for point in results:
        payload = point.payload
        if is_image:
            path_str = payload.get('path', '')
            parsed.append(Path(path_str).name)
        else:
            parsed.append(payload.get('text', ''))
    return parsed


def get_expected_value(row: pd.Series, is_image: bool) -> str:
    if is_image:
        return Path(row['expected_image_path']).name
    return row['ground_truth_text']


def process_single_query(
        row: pd.Series,
        search_func: Callable,
        k: int,
        is_image: bool
) -> tuple[int, float]:
    try:
        results = search_func(row['query'], k)
        found_vals = parse_results(results, is_image)
        expected_val = get_expected_value(row, is_image)

        hit = calculate_hit(expected_val, found_vals)
        mrr = calculate_reciprocal_rank(expected_val, found_vals)
        return hit, mrr
    except Exception as e:
        return 0, 0.0


def run_retrieval_experiment(
        variant_name: str,
        search_func: Callable[[str, int], Any],
        dataset: pd.DataFrame,
        k: int = 5,
        is_image_search: bool = False
) -> dict:
    hits = []
    ranks = []

    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        hit, rank = process_single_query(row, search_func, k, is_image_search)
        hits.append(hit)
        ranks.append(rank)

    return {
        "Variant": variant_name,
        "Hit Rate": sum(hits) / len(hits) if hits else 0,
        "MRR": sum(ranks) / len(ranks) if ranks else 0,
        "k": k
    }