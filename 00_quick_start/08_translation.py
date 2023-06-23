from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")  # 法语翻译英语
# max_length或min_length控制翻译长短
answer = translator("Ce cours est produit par Hugging Face.")
print(answer)
"""
[{'translation_text': 'This course is produced by Hugging Face.'}]
"""
