from transformers import pipeline
"""
对尚未标记的文本进行分类
实际项目中的常见场景，因为注释文本通常很耗时并且需要领域专业知识。
直接指定用于分类的标签，因此不必依赖预训练模型的标签。
下面的模型展示了如何使用这两个标签将句子分类为正面或负面——但也可以使用您喜欢的任何其他标签集对文本进行分类。
"""
classifier = pipeline("zero-shot-classification")
answer = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
print(answer)
"""
{'sequence': 'This is a course about the Transformers library',
 'labels': ['education', 'business', 'politics'],
 'scores': [0.8445963859558105, 0.111976258456707, 0.043427448719739914]}
"""