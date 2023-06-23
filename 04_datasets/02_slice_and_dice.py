"""
切片
"""

# TSV只是使用制表符而不是逗号作为分隔符的CSV变体，我们可以使用加载csv文件的load_dataset()函数并指定分隔符
from datasets import load_dataset

data_files = {"train": "./data/drugsCom_raw/drugsComTrain_raw.tsv", "test": "./data/drugsCom_raw/drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
# print(drug_dataset)
# print(len(drug_dataset))
"""
DatasetDict({
    train: Dataset({
        features: ['Unnamed: 0', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 161297
    })
    test: Dataset({
        features: ['Unnamed: 0', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 53766
    })
})
"""

# drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))  # 将训练数据打乱，取其前1000个数据
# Peek at the first few examples
# print(drug_sample[:3])
"""
{'Unnamed: 0': [87571, 178045, 80482],
 'drugName': ['Naproxen', 'Duloxetine', 'Mobic'],
 'condition': ['Gout, Acute', 'ibromyalgia', 'Inflammatory Conditions'],
 'review': ['"like the previous person mention, I&#039;m a strong believer of aleve, it works faster for my gout than the prescription meds I take. No more going to the doctor for refills.....Aleve works!"',
  '"I have taken Cymbalta for about a year and a half for fibromyalgia pain. It is great\r\nas a pain reducer and an anti-depressant, however, the side effects outweighed \r\nany benefit I got from it. I had trouble with restlessness, being tired constantly,\r\ndizziness, dry mouth, numbness and tingling in my feet, and horrible sweating. I am\r\nbeing weaned off of it now. Went from 60 mg to 30mg and now to 15 mg. I will be\r\noff completely in about a week. The fibro pain is coming back, but I would rather deal with it than the side effects."',
  '"I have been taking Mobic for over a year with no side effects other than an elevated blood pressure.  I had severe knee and ankle pain which completely went away after taking Mobic.  I attempted to stop the medication however pain returned after a few days."'],
 'rating': [9.0, 3.0, 10.0],
 'date': ['September 2, 2015', 'November 7, 2011', 'June 5, 2013'],
 'usefulCount': [36, 13, 128]}
 
 Unnamed: 0，每个患者的匿名ID
 condition，包含有描述健康状况的标签
 """

# 验证Unnamed：0为患者的匿名ID
# for split in drug_dataset.keys():
#     assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))  # assert（断言）用于判断一个表达式，在表达式条件为false的时候触发异常

# DatasetDict.rename_column()函数一次性重命名DatasetDict中共有的列
drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
# print(drug_dataset)
"""
DatasetDict({
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 161297
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 53766
    })
})
"""

# condition列存在None, 不能转换为小写
# 第一种写法
# def filter_nones(x):
#     return x["condition"] is not None
# drug_dataset.filter(filter_nones)

# 第二种写法
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)


# 小写函数
def lowercase_condition(example):
    return {"condition": example["condition"].lower()}


drug_dataset = drug_dataset.map(lowercase_condition)
# Check that lowercasing worked
# print(drug_dataset["train"]["condition"][:3])
"""
['left ventricular dysfunction', 'adhd', 'birth control']
"""

"""
创建新的数据列
    Dataset.map()
    Dataset.add_column() 
计算每条评论的长度
"""
# def compute_review_length(example):
#     return {"review_length": len(example["review"].split())}
#
# drug_dataset = drug_dataset.map(compute_review_length)

drug_dataset = drug_dataset.map(lambda example: {"review_length": len(example["review"].split())})

# Inspect the first training example
print(drug_dataset["train"][0])
"""
{'patient_id': 206461,
 'drugName': 'Valsartan',
 'condition': 'left ventricular dysfunction',
 'review': '"It has no side effect, I take it in combination of Bystolic 5 Mg and Fish Oil"',
 'rating': 9.0,
 'date': 'May 20, 2012',
 'usefulCount': 27,
 'review_length': 17}
"""

# 使用Dataset.sort()对这个新列进行排序
print(drug_dataset["train"].sort("review_length")[:3])

# Dataset.filter()功能来删除包含少于30个单词的评论
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
# print(drug_dataset.num_rows)  # {'train': 138514, 'test': 46108}

# 评论中是否存在HTML字符代码，使用Python的html模块取消这些字符的转义
import html

# text = "I&#039;m a transformer called BERT"
# print(html.unescape(text))  # "I'm a transformer called BERT"

# drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})

"""
Dataset.map()方法有一个batched参数，如果设置为True, map函数将会分批执行所需要进行的操作（批量大小是可配置的，但默认为1,000）
"""
new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True
)


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True)

# 计时
# notebook中
# %time tokenized_dataset = drug_dataset.map(tokenize_function, batched=True)
import time

start_time = time.time()
tokenized_dataset = drug_dataset.map(tokenize_function, batched=True)
end_time = time.time()
print(end_time-start_time)



# num_proc 参数并指定要在调用中使用的进程数
slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)


def slow_tokenize_function(examples):
    return slow_tokenizer(examples["review"], truncation=True)


tokenized_dataset = drug_dataset.map(slow_tokenize_function, batched=True, num_proc=8)
"""
DatasetDict({
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 138514
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 46108
    })
})
"""



def tokenize_and_split(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,  # 标记器返回全部文本
    )

result = tokenize_and_split(drug_dataset["train"][0])
# print(result)

# print([len(inp) for inp in result["input_ids"]])  # [128, 49]    由于设置了最大长度，在128个字符出截断，分成了一个128和一个49

# 对数据集中所有的数据进行字符转换为词向量
# tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)  # 同一数据会生成不同的特征，导致数据扩增，使得列长度不同产生错误

# 解决方法一
# tokenized_dataset = drug_dataset.map(
#     tokenize_and_split, batched=True, remove_columns=drug_dataset["train"].column_names
# )  # 保留remove_columns中的列，其余均删除

# 解决方法二
# 生成的新特征与原始数据进行映射
def tokenize_and_split(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # Extract mapping between new and old indices
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result

tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
# print(tokenized_dataset)
"""
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'condition', 'date', 'drugName', 'input_ids', 'patient_id', 'rating', 'review', 'review_length', 'token_type_ids', 'usefulCount'],
        num_rows: 206772
    })
    test: Dataset({
        features: ['attention_mask', 'condition', 'date', 'drugName', 'input_ids', 'patient_id', 'rating', 'review', 'review_length', 'token_type_ids', 'usefulCount'],
        num_rows: 68876
    })
})
"""


# 各种第三方库之间的转换
drug_dataset.set_format("pandas")
# print(drug_dataset["train"][:3])

train_df = drug_dataset["train"][:]

# 计算"condition"类之间的分布
frequencies = (
    train_df["condition"]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={"index": "condition", "condition": "frequency"})
)
# print(frequencies.head())

# 完成了Pandas分析，通过使用对象Dataset.from_pandas()方法创建一个新的Dataset

from datasets import Dataset

freq_dataset = Dataset.from_pandas(frequencies)
# print(freq_dataset)
"""
Dataset({
    features: ['condition', 'frequency'],
    num_rows: 819
})
"""

# 将输出格式drug_dataset从pandas重置到arrow
drug_dataset.reset_format()

# 创建验证集
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# Rename the default "test" split to "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# Add the "test" set to our `DatasetDict`
drug_dataset_clean["test"] = drug_dataset["test"]
# print(drug_dataset_clean)
"""
DatasetDict({
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'review_clean'],
        num_rows: 110811
    })
    validation: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'review_clean'],
        num_rows: 27703
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'review_clean'],
        num_rows: 46108
    })
})
"""


"""
保存数据集
三种方式
    Arrow	Dataset.save_to_disk()
    CSV	    Dataset.to_csv()
    JSON	Dataset.to_json()
"""
drug_dataset_clean.save_to_disk("drug-reviews")
"""
这将创建一个具有以下结构的目录：
drug-reviews/
├── dataset_dict.json
├── test
│   ├── dataset.arrow
│   ├── dataset_info.json
│   └── state.json
├── train
│   ├── dataset.arrow
│   ├── dataset_info.json
│   ├── indices.arrow
│   └── state.json
└── validation
    ├── dataset.arrow
    ├── dataset_info.json
    ├── indices.arrow
    └── state.json
"""

# 读取数据，load_from_disk()
from datasets import load_from_disk

drug_dataset_reloaded = load_from_disk("drug-reviews")
# print(drug_dataset_reloaded)
"""
DatasetDict({
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length'],
        num_rows: 110811
    })
    validation: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length'],
        num_rows: 27703
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length'],
        num_rows: 46108
    })
})
"""

# 对于CSV和JSON格式，必须将每个部分存储为单独的文件。一种方法是迭代DatasetDict中的键和值
for split, dataset in drug_dataset_clean.items():
    dataset.to_json(f"drug-reviews-{split}.jsonl")

# 加载文件
data_files = {
    "train": "drug-reviews-train.jsonl",
    "validation": "drug-reviews-validation.jsonl",
    "test": "drug-reviews-test.jsonl",
}
drug_dataset_reloaded = load_dataset("json", data_files=data_files)