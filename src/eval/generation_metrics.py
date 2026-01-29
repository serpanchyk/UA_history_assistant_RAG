import numpy as np
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate

from src import PROMPTS_DIR_PATH
from src.fs_io.filesystem import read_text_file
from src.index.embedder import EmbeddingService


def evaluate_faithfulness(context: str, answer: str, judge_llm: AzureChatOpenAI) -> int:
    """
    Check if model uses only information from context and not hallucinate.
    """
    prompt_template = PromptTemplate.from_template(read_text_file(PROMPTS_DIR_PATH / "evaluate_faithfulness.txt"))
    prompt = prompt_template.format(
        context=context,
        answer=answer
    )

    try:
        res = judge_llm.invoke([HumanMessage(content=prompt)])
        return 1 if "1" in res.content else 0
    except:
        return 0


def evaluate_citation_correctness(context: str, answer: str, judge_llm: AzureChatOpenAI) -> int:
    """
    Check if the citations correct and supported by the context.
    """
    prompt_template = PromptTemplate.from_template(read_text_file(PROMPTS_DIR_PATH / "evaluate_citation_correctness.txt"))
    prompt = prompt_template.format(
        context=context,
        answer=answer
    )
    try:
        res = judge_llm.invoke([HumanMessage(content=prompt)])
        return 1 if "1" in res.content else 0
    except:
        return 0


def calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def evaluate_semantic_similarity(
        generated_answer: str,
        ground_truth_text: str,
        embedder: EmbeddingService
) -> float:
    """
    Calculates cosine similarity between the generated answer and the source chunk.
    High score = The answer covers the same semantic ground as the source.
    """
    # Use 'query' mode or 'text' mode depending on your embedder,
    # but strictly we just want dense vector comparison here.
    vec_gen = embedder.embed_text(generated_answer)['dense']
    vec_gt = embedder.embed_text(ground_truth_text)['dense']

    return calculate_cosine_similarity(vec_gen, vec_gt)


def evaluate_answer_relevance(query: str, answer: str, judge_llm: AzureChatOpenAI) -> int:
    """
    Checks if the answer actually addresses the specific question asked.
    """
    prompt_template = PromptTemplate.from_template(
        read_text_file(PROMPTS_DIR_PATH / "evaluate_answer_relevance.txt"))
    prompt = prompt_template.format(
        query=query,
        answer=answer
    )
    try:
        res = judge_llm.invoke([HumanMessage(content=prompt)])
        return 1 if "1" in res.content else 0
    except:
        return 0
