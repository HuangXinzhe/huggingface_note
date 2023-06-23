"""
加载数据两种方式：
    本地文件
    远程文件

支持的几种常见的数据格式
    load_dataset("csv", data_files="my_file.csv")
    load_dataset("text", data_files="my_file.txt")
    load_dataset("json", data_files="my_file.jsonl")
    load_dataset("pandas", data_files="my_dataframe.pkl")
"""

# ================================================加载本地数据=============================================
"""
加载本地数据集
以意大利语问答数据为例
"""
from datasets import load_dataset

# 所有数据都存储在data中，构建自己的数据集需要注意
# squad_it_dataset = load_dataset("json", data_files="./data/SQuAD_it/SQuAD_it-train.json", field="data")
# print(squad_it_dataset)
"""
默认情况下, 加载本地文件会创建一个带有train的DatasetDict对象
DatasetDict({
    train: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 442
    })
})
"""

"""
构建train和test的DatasetDict对象，使用Dataset.map()函数同时处理训练集和测试集
"""
# data_files = {"train": "./data/SQuAD_it/SQuAD_it-train.json", "test": "./data/SQuAD_it/SQuAD_it-test.json"}
# squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
# print(squad_it_dataset)

"""
Datasets实际上支持输入文件的自动解压,所以我们可以跳过使用gzip,直接设置data_files参数传递压缩文件
"""
# data_files = {"train": "./data/SQuAD_it/SQuAD_it-train.json.gz", "test": "./data/SQuAD_it/SQuAD_it-test.json.gz"}
# squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
# print(squad_it_dataset)

# ================================================加载远程数据=============================================
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
print(squad_it_dataset)