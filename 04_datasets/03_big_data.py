from datasets import load_dataset

# This takes a few minutes to run, so go grab a tea or coffee while you wait :)
data_files = "https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
pubmed_dataset = load_dataset("json", data_files=data_files, split="train")
# print(pubmed_dataset)
"""
Dataset({
    features: ['meta', 'text'],
    num_rows: 15518009
})
"""

# 数据集第一个元素
# print(pubmed_dataset[0])
"""
{'meta': {'pmid': 11409574, 'language': 'eng'},
 'text': 'Epidemiology of hypoxaemia in children with acute lower respiratory infection.\nTo determine the prevalence of hypoxaemia in children aged under 5 years suffering acute lower respiratory infections (ALRI), the risk factors for hypoxaemia in children under 5 years of age with ALRI, and the association of hypoxaemia with an increased risk of dying in children of the same age ...'}
"""

# 在Python中测量内存使用情况
# Process类允许检查当前进程的内存使用情况
import psutil

# Process.memory_info is expressed in bytes, so convert to megabytes
# print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")  # RAM used: 5678.33 MB

# rss属性是指常驻集的大小, 它是进程在RAM中占用的内存比例。这个测量结果也包括了Python编译器和我们加载的库所使用的内存,
# 所以实际上用于加载数据集的内存会更小一些。为了比较, 让我们使用dataset_size属性看看数据集在磁盘上有多大。由于结果像之前一样用字节表示, 我们需要手动将其转换为GB
# print(f"Number of files in dataset : {pubmed_dataset.dataset_size}")
size_gb = pubmed_dataset.dataset_size / (1024**3)
# print(f"Dataset size (cache file) : {size、_gb:.2f} GB")
"""
Number of files in dataset : 20979437051
Dataset size (cache file) : 19.54 GB
"""

# Datasets将每一个数据集看作一个内存映射文件, 它提供了RAM和文件系统存储之间的映射, 该映射允许库访问和操作数据集的元素, 而且无需将其完全加载到内存中。




import timeit

code_snippet = """batch_size = 1000

for idx in range(0, len(pubmed_dataset), batch_size):
    _ = pubmed_dataset[idx:idx + batch_size]
"""

time = timeit.timeit(stmt=code_snippet, number=1, globals=globals())
print(
    f"Iterated over {len(pubmed_dataset)} examples (about {size_gb:.1f} GB) in "
    f"{time:.1f}s, i.e. {size_gb/time:.3f} GB/s"
)
"""
'Iterated over 15518009 examples (about 19.5 GB) in 64.2s, i.e. 0.304 GB/s'
"""


# 流式数据集
# streaming=True
pubmed_dataset_streamed = load_dataset(
    "json", data_files=data_files, split="train", streaming=True
)

# print(next(iter(pubmed_dataset_streamed)))
"""
{'meta': {'pmid': 11409574, 'language': 'eng'},
 'text': 'Epidemiology of hypoxaemia in children with acute lower respiratory infection.\nTo determine the prevalence of hypoxaemia in children aged under 5 years suffering acute lower respiratory infections (ALRI), the risk factors for hypoxaemia in children under 5 years of age with ALRI, and the association of hypoxaemia with an increased risk of dying in children of the same age ...'}
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_dataset = pubmed_dataset_streamed.map(lambda x: tokenizer(x["text"]))
# print(next(iter(tokenized_dataset)))  # 输出是逐个的返回
"""
{'input_ids': [101, 4958, 5178, 4328, 6779, ...], 'attention_mask': [1, 1, 1, 1, 1, ...]}
"""

# 此处打乱数据仅是打乱缓冲区中的数据
shuffled_dataset = pubmed_dataset_streamed.shuffle(buffer_size=10_000, seed=42)
# print(next(iter(shuffled_dataset)))
"""
{'meta': {'pmid': 11410799, 'language': 'eng'},
 'text': 'Randomized study of dose or schedule modification of granulocyte colony-stimulating factor in platinum-based chemotherapy for elderly patients with lung cancer ...'}
"""
# 在这个示例中从缓冲区的前10,000个示例中随机选择了一个示例。一旦访问了一个示例, 它在缓冲区中的位置就会被语料库中的下一个示例填充 (即上述案例中的第10,001个示例)。



# IterableDataset.take() 和 IterableDataset.skip() 函数从流式数据集中选择元素, 它的作用类似于 Dataset.select()
dataset_head = pubmed_dataset_streamed.take(5)  # 选择前5个
# print(list(dataset_head))
"""
[{'meta': {'pmid': 11409574, 'language': 'eng'},
  'text': 'Epidemiology of hypoxaemia in children with acute lower respiratory infection ...'},
 {'meta': {'pmid': 11409575, 'language': 'eng'},
  'text': 'Clinical signs of hypoxaemia in children with acute lower respiratory infection: indicators of oxygen therapy ...'},
 {'meta': {'pmid': 11409576, 'language': 'eng'},
  'text': "Hypoxaemia in children with severe pneumonia in Papua New Guinea ..."},
 {'meta': {'pmid': 11409577, 'language': 'eng'},
  'text': 'Oxygen concentrators and cylinders ...'},
 {'meta': {'pmid': 11409578, 'language': 'eng'},
  'text': 'Oxygen supply in rural africa: a personal experience ...'}]
"""


# IterableDataset.skip() 函数将打乱的数据集拆分为训练集和验证集
# Skip the first 1,000 examples and include the rest in the training set
train_dataset = shuffled_dataset.skip(1000)
# Take the first 1,000 examples for the validation set
validation_dataset = shuffled_dataset.take(1000)





# interleave_datasets() 函数, 它将一个 IterableDataset 对象列表组合为单个的 IterableDataset, 其中新数据集的元素是通过在列表中的对象交替获得的
law_dataset_streamed = load_dataset(
    "json",
    data_files="https://the-eye.eu/public/AI/pile_preliminary_components/FreeLaw_Opinions.jsonl.zst",
    split="train",
    streaming=True,
)
# print(next(iter(law_dataset_streamed)))
"""
{'meta': {'case_ID': '110921.json',
  'case_jurisdiction': 'scotus.tar.gz',
  'date_created': '2010-04-28T17:12:49Z'},
 'text': '\n461 U.S. 238 (1983)\nOLIM ET AL.\nv.\nWAKINEKONA\nNo. 81-1581.\nSupreme Court of United States.\nArgued January 19, 1983.\nDecided April 26, 1983.\nCERTIORARI TO THE UNITED STATES COURT OF APPEALS FOR THE NINTH CIRCUIT\n*239 Michael A. Lilly, First Deputy Attorney General of Hawaii, argued the cause for petitioners. With him on the brief was James H. Dannenberg, Deputy Attorney General...'}
"""


from itertools import islice
from datasets import interleave_datasets

combined_dataset = interleave_datasets([pubmed_dataset_streamed, law_dataset_streamed])
# print(list(islice(combined_dataset, 2)))
"""
[{'meta': {'pmid': 11409574, 'language': 'eng'},
  'text': 'Epidemiology of hypoxaemia in children with acute lower respiratory infection ...'},
 {'meta': {'case_ID': '110921.json',
   'case_jurisdiction': 'scotus.tar.gz',
   'date_created': '2010-04-28T17:12:49Z'},
  'text': '\n461 U.S. 238 (1983)\nOLIM ET AL.\nv.\nWAKINEKONA\nNo. 81-1581.\nSupreme Court of United States.\nArgued January 19, 1983.\nDecided April 26, 1983.\nCERTIORARI TO THE UNITED STATES COURT OF APPEALS FOR THE NINTH CIRCUIT\n*239 Michael A. Lilly, First Deputy Attorney General of Hawaii, argued the cause for petitioners. With him on the brief was James H. Dannenberg, Deputy Attorney General...'}]
"""
#  islice() 函数从合并的数据集中选择前两个示例, 并且我们可以看到它们实际上就是两个源数据集中的前两个示例拼在一起形成的



# 流式传输整个825GB的 Pile, 你可以按照如下方式获取所有准备好的文件
base_url = "https://the-eye.eu/public/AI/pile/"
data_files = {
    "train": [base_url + "train/" + f"{idx:02d}.jsonl.zst" for idx in range(30)],
    "validation": base_url + "val.jsonl.zst",
    "test": base_url + "test.jsonl.zst",
}
pile_dataset = load_dataset("json", data_files=data_files, streaming=True)
# print(next(iter(pile_dataset["train"])))
"""
{'meta': {'pile_set_name': 'Pile-CC'},
 'text': 'It is done, and submitted. You can play “Survival of the Tastiest” on Android, and on the web...'}
"""