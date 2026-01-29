import pandas as pd
from langchain_openai import AzureChatOpenAI
from tqdm import tqdm

from src.index.embedder import EmbeddingService
from src.eval.generation_metrics import (
    evaluate_faithfulness,
    evaluate_citation_correctness,
    evaluate_answer_relevance,
    evaluate_semantic_similarity
)
from src.rag.llm_service import LLMService

def run_generation_eval(
        llm_service: LLMService,
        judge_llm: AzureChatOpenAI,
        embedder: EmbeddingService,
        dataset: pd.DataFrame
) -> pd.DataFrame:
    results = []
    print("Running extended generation evaluation...")

    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        query = row['query']
        ground_truth = row.get('ground_truth_text', '')

        context_obj = llm_service.retrieve(query)
        response_generator = llm_service.generate_response(query, context_obj)
        full_response = "".join(list(response_generator))

        faith_score = evaluate_faithfulness(context_obj.text_context, full_response, judge_llm)
        cit_score = evaluate_citation_correctness(context_obj.text_context, full_response, judge_llm)

        rel_score = evaluate_answer_relevance(query, full_response, judge_llm)

        sem_score = 0.0
        if ground_truth:
            sem_score = evaluate_semantic_similarity(full_response, ground_truth, embedder)

        results.append({
            "query": query,
            "response": full_response,
            "faithfulness": faith_score,
            "citation_correctness": cit_score,
            "answer_relevance": rel_score,
            "semantic_similarity": sem_score
        })

    return pd.DataFrame(results)