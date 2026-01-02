from rag.llm import get_response

def question_llm():
    with open('rag/prompt.txt', 'r') as f:
        prompt = f.read()
    get_response(prompt)


if __name__ == '__main__':
    question_llm()

