from transformers import pipeline

"""
使用来自给定上下文的信息回答问题
"""
question_answerer = pipeline("question-answering")
answer = question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
print(answer)
"""
{'score': 0.6385916471481323, 'start': 33, 'end': 45, 'answer': 'Hugging Face'}
"""
