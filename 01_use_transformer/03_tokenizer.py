"""
标记器
将数据转换为模型可以处理的数据
"""

"""
基于词的标记
缺点是词库大，需要标记的词汇多，未登录词多
有多种标记方式，以下例子为以空格分割标记
"""
# tokenized_text = "Jim Henson was a puppeteer".split()
# print(tokenized_text)

"""
基于字符的标记
字符有限，较少会出现未登录词（英文）
缺点：存在空格和标点符号的问题
"""

"""
子词标记化
不应将常用词拆分为更小的子词，而应将稀有词分解为有意义的子词
"""

# from transformers import BertTokenizer, AutoTokenizer

# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# AutoTokenizer 类将根据检查点名称在库中获取正确的标记器(tokenizer)类
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# answer = tokenizer("Using a Transformer network is simple")
# print(answer)  # {'input_ids': [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}

# 标记器保存的文件：special_tokens_map.json，tokenizer.json，tokenizer_config.json，vocab.txt
# tokenizer.save_pretrained("directory_on_my_computer")

"""
编码
1、标记化：将文本拆分为标记
2、将标记转换为输入ID
"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)  # 获得文本的标记

print(tokens)  # ['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']

ids = tokenizer.convert_tokens_to_ids(tokens)  # 从词符到输入ID

print(ids)  # [7993, 170, 13809, 23763, 2443, 1110, 3014]

"""
解码
从词汇索引中，想要得到一个字符串可以通过decode()方法实现
decode方法不仅将索引转换回标记(token)，还将属于相同单词的标记(token)组合在一起以生成可读的句子
"""
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)  # 'Using a Transformer network is simple'

"""
特殊字符
"""
sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])  # [101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102]

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)  # [1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]

print(tokenizer.decode(model_inputs["input_ids"]))  # "[CLS] i've been waiting for a huggingface course my whole life. [SEP]"
print(tokenizer.decode(ids))  # "i've been waiting for a huggingface course my whole life."

"""
以下为较为完整版代码
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
print(output)
