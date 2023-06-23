"""
第一部分通过pipline直接完成情感分析任务
第二至第四部分将任务拆解为几个部分
第二部分对原始数据进行预处理，获得句子中的每个词的标记
第三和第四部分展示的是不同的模型及第四部分展示的模型结果
"""

# ==============================第一部分_整体模型完成==============================
"""
将一些文本传递到pipeline时涉及三个主要步骤：
    1、文本被预处理为模型可以理解的格式。
    2、预处理的输入被传递给模型。
    3、模型处理后输出最终人类可以理解的结果。

现在可用的pipeline
    feature-extraction (get the vector representation of a text)
    fill-mask
    ner (named entity recognition)
    question-answering
    sentiment-analysis
    summarization
    text-generation
    translation
    zero-shot-classification
"""
# from transformers import pipeline
# 情感分析，积极还是消极

# classifier = pipeline("sentiment-analysis")
# classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 单句判断
# answer = classifier("I've been waiting for a HuggingFace course my whole life.")

# 多句判断
# answer = classifier(
#     ["I've been waiting for a HuggingFace course my whole life.",
#      "I hate this so much!"]
# )
#
# print(answer)

# ==============================第二部分_使用分词器进行预处理==============================
"""
tokenizer
    将输入拆分为单词、子单词或符号（如标点符号），称为标记(token)
    将每个标记(token)映射到一个整数
    添加可能对模型有用的其他输入
"""
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"  # 使用此模型的权重
tokenizer = AutoTokenizer.from_pretrained(checkpoint)  # 使用该模型的标记器

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")  # 获得数据中每个单词的标记
# print(inputs)

# ==============================第三部分_浏览模型==============================
# from transformers import AutoModel
#
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# model = AutoModel.from_pretrained(checkpoint)
#
# outputs = model(**inputs)
# print(outputs.last_hidden_state.shape)  # torch.Size([2, 16, 768])，表示两个数据，每个数据16个单词，每个单词由768维向量表示


# ==============================第四部分_浏览模型==============================
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

"""
获得两个句子对应的标签概率值，较大数值的标签即为分类结果的标签
"""
print(outputs.logits.shape)  # torch.Size([2, 2])，表示两个句子两个标签

print(outputs.logits)  # tensor([[-1.5607,  1.6123], [ 4.1692, -3.3464]], grad_fn=<AddmmBackward0>)

# 通过softmax将结果转换为概率
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)  # tensor([[4.0195e-02, 9.5981e-01], [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward0>)

print(model.config.id2label)  # {0: 'NEGATIVE', 1: 'POSITIVE'}， 模型结果对应的标签
