from llama_cpp import Llama

llm = Llama(
    model_path="models/MamayLM-Gemma-3-4B-IT-v1.0.Q4_K_S.gguf",
    n_ctx=4096,
    n_threads=8,
    penalize_nl=False,
    verbose=True
)

def get_response(content):
    messages = [{"role": "user", "content": content}]

    response_stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=2048,
        temperature=0.1,
        top_p=0.9,
        repeat_penalty=1.0,
        stop=["<eos>", "<end_of_turn>"],
        stream=True
    )

    for chunk in response_stream:
        delta = chunk['choices'][0]['delta']

        if 'content' in delta:
            print(delta['content'], end='', flush=True)

    print("\n\n--- Done ---")



