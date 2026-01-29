import os
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

from src.config import settings
from src.fs_io.dataframes import read_parquet

load_dotenv()

llm_gen = AzureChatOpenAI(
    azure_deployment="gpt-5-mini",
    azure_endpoint="https://case-synthesizer.cognitiveservices.azure.com/",
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version="2024-05-01-preview"
)

TARGET_COUNT = 50


def generate_text_eval_dataset():
    print(f"\n--- Generating {TARGET_COUNT} Text Questions ---")

    if not settings.CHUNKS_DF_PATH.exists():
        print(f"Error: Chunks file not found at {settings.CHUNKS_DF_PATH}")
        return

    chunks_df = read_parquet(settings.CHUNKS_DF_PATH)
    eval_data = []

    while len(eval_data) < TARGET_COUNT:
        try:
            row = chunks_df.sample(1).iloc[0]
            context = row['text'][:1000]

            prompt = f"""
            Excerpt from a history textbook of Ukraine:
            "{context}"

            Your task: Formulate an exam question (NMT style) based on this excerpt.

            Instructions:
            1. **Focus on the Core:** Ask about the key event, person, date, or consequence.
            2. **Style:** The question should be clear and specific.
            3. **Output Language:** Ukrainian.
            4. **Use keyword** mentions the names if historical figures. 

            Your Question (in Ukrainian):
            """

            response = llm_gen.invoke(prompt)
            q = response.content.strip()

            if len(q) < 10:
                continue

            eval_data.append({
                "query": q,
                "expected_id": row['id'],
                "ground_truth_text": row['text']
            })
            print(f"[{len(eval_data)}/{TARGET_COUNT}] Text: {q}")

        except Exception as e:
            print(f"Error: {e}. Retrying in 2s...")
            time.sleep(2)

    df_eval = pd.DataFrame(eval_data)
    save_path = settings.EVAL_TEXTS_DF_PATH
    df_eval.to_parquet(save_path)
    print(f"Text evaluation data saved to: {save_path}")


if __name__ == "__main__":
    generate_text_eval_dataset()