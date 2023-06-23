from transformers import pipeline

# ------------------一次只输入一个数据------------------
classifier = pipeline("sentiment-analysis")  # 设置需要完成的任务
answer = classifier("I've been waiting for a HuggingFace course my whole life.")
print(answer)
"""
[{'label': 'POSITIVE', 'score': 0.9598047137260437}]
"""


# ------------------一次只输入多个数据------------------
answer = classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)
print(answer)
"""
[{'label': 'POSITIVE', 'score': 0.9598047137260437},
 {'label': 'NEGATIVE', 'score': 0.9994558095932007}]
"""