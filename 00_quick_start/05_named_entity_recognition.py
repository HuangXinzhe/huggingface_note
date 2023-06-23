from transformers import pipeline
"""
命名实体识别
"""
ner = pipeline("ner", grouped_entities=True, model="StanfordAIMI/stanford-deidentifier-base")  # grouped_entities=True是对属于同一个实体的重新组合
answer = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(answer)
"""
[{'entity_group': 'PER', 'score': 0.99816, 'word': 'Sylvain', 'start': 11, 'end': 18}, 
 {'entity_group': 'ORG', 'score': 0.97960, 'word': 'Hugging Face', 'start': 33, 'end': 45}, 
 {'entity_group': 'LOC', 'score': 0.99321, 'word': 'Brooklyn', 'start': 49, 'end': 57}
]
"""
