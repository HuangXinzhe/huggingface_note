import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)  # 模型默认输入是多个句子

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)


model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)  # tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward0>)
print(model(torch.tensor(sequence2_ids)).logits)  # tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)
print(model(torch.tensor(batched_ids)).logits)  # tensor([[ 1.5694, -1.3895], [ 1.3374, -1.2163]], grad_fn=<AddmmBackward0>)
"""
sequence2与batched中结果不一致原因在于padding
使用attention_mask完成去除padding影响
"""

batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

# 1表示注意相对应的标记，0表示不应注意相对应的标记
attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)


"""
长序列
对于Transformers模型，可以通过模型的序列长度是有限的。
大多数模型处理多达512或1024个令牌的序列，当要求处理更长的序列时，会崩溃。
此问题有两种解决方案：
    1、使用支持的序列长度较长的模型。
    2、截断序列。

通过指定max_sequence_length参数
sequence = sequence[:max_sequence_length]
"""