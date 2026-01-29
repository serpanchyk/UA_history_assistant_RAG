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


def generate_image_eval_dataset():
    print(f"\n--- Generating {TARGET_COUNT} Image Queries ---")

    if not settings.IMAGES_DF_PATH.exists():
        print(f"Error: Images file not found at {settings.IMAGES_DF_PATH}")
        return

    images_df = read_parquet(settings.IMAGES_DF_PATH)

    # Filter valid images
    valid_images = images_df[
        (images_df['caption'].str.len() > 20) &
        (images_df['caption'] != 'Зображення без опису')
        ].copy()

    if valid_images.empty:
        print("Error: No valid images found (all captions are too short or missing).")
        return

    image_eval_data = []

    # Retry loop
    while len(image_eval_data) < TARGET_COUNT:
        try:
            # Sample 1 random image
            row = valid_images.sample(1).iloc[0]
            caption = row['caption']

            # Check for duplicates in current session
            if any(d['expected_image_path'] == row['path'] for d in image_eval_data):
                continue

            prompt = f"""
            Excerpt from a caption of image of history textbook of Ukraine:
            "{caption}"

            Your task: Formulate an exam question (NMT style) based on this excerpt.

            Instructions:
            1. **Focus on the Core:** Ask about the key event, person, date, or consequence.
            2. **Style:** The question should be clear and specific.
            3. **Output Language:** Ukrainian.
            4. **Use keyword** mentions the names if historical figures, events, buildings, arts. 

            Your Question (in Ukrainian):
            """

            response = llm_gen.invoke(prompt)
            q = response.content.strip()

            if len(q) < 5:
                continue

            image_eval_data.append({
                "query": q,
                "expected_image_path": row['path'],
                "expected_doc_id": row['doc_id'],
                "ground_truth_caption": caption
            })
            print(f"[{len(image_eval_data)}/{TARGET_COUNT}] Image: {q}")

        except Exception as e:
            print(f"Error: {e}. Retrying in 2s...")
            time.sleep(2)

    df_image_eval = pd.DataFrame(image_eval_data)
    save_path = settings.EVAL_IMAGES_DF_PATH
    df_image_eval.to_parquet(save_path)
    print(f"Image evaluation data saved to: {save_path}")


if __name__ == "__main__":
    generate_image_eval_dataset()