"""
MRPC（微软研究释义语料库）数据集
由5801对句子组成，每个句子对带有一个标签，指示它们是否为同义（即，如果两个句子的意思相同）
"""

# import torch
# from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
#
# # Same as before
# checkpoint = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
# sequences = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "This course is amazing!",
# ]
# batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
#
# # This is new
# batch["labels"] = torch.tensor([1, 1])
#
# optimizer = AdamW(model.parameters())
# loss = model(**batch).loss
# loss.backward()
# optimizer.step()


from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
# print(raw_datasets)
"""
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
"""

# raw_train_dataset = raw_datasets["train"]
# print(raw_train_dataset[0])
"""
{'idx': 0,
 'label': 1,
 'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
 'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .'}
"""

# print(raw_train_dataset.features)
"""
{'sentence1': Value(dtype='string', id=None),
 'sentence2': Value(dtype='string', id=None),
 'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
 'idx': Value(dtype='int32', id=None)}
 0代表not_equivalent，1代表equivalent
"""

from transformers import AutoTokenizer

# 预处理数据集
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

inputs = tokenizer("This is the first sentence.", "This is the second one.")
# print(inputs)
"""
{ 
  'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
"""

# id转换回文字
# print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))
"""
模型需要输入的形式是 [CLS] sentence1 [SEP] sentence2 [SEP]
输入中 [CLS] sentence1 [SEP] 它们的类型标记ID均为0，而其他部分，对应于sentence2 [SEP]，所有的类型标记ID均为1
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
[      0,      0,    0,     0,       0,          0,   0,       0,      1,    1,     1,        1,     1,   1,       1]
"""

tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)  # 可以通过传递num_proc参数使用并行处理
# print(tokenized_datasets)
"""
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 1725
    })
})
"""

"""
动态填充语料长度
每个batch中填充至该batch中最长的语料
"""
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
print(samples)
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print(samples)
print([len(x) for x in samples["input_ids"]])  # [50, 59, 47, 67, 59, 50, 62, 32]

batch = data_collator(samples)  # 根据一批中最长的数据进行填充
print({k: v.shape for k, v in batch.items()})
"""
{'attention_mask': torch.Size([8, 67]),
 'input_ids': torch.Size([8, 67]),
 'token_type_ids': torch.Size([8, 67]),
 'labels': torch.Size([8])}
"""