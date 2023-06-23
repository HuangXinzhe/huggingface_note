from transformers import pipeline

"""
使用的原始模型的时候，很容易生成性别歧视、种族主义或恐同内容。这种固有偏见不会随着微调模型而使消失。
"""
unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("This man works as a [MASK].")
print([r["token_str"] for r in result])

result = unmasker("This woman works as a [MASK].")
print([r["token_str"] for r in result])
