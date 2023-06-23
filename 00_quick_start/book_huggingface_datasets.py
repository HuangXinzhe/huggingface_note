from pprint import pprint
from datasets import list_datasets, load_dataset
datasets_list = list_datasets()  # 全部数据集列表
# print(len(datasets_list))

# 加载SST数据集（训练数据部分）
dataset = load_dataset('sst', split='train')
# print(len(dataset))

# 打印以字典对象存储的样本，字典中存储标签、原始句子、标记序列、句法分析树
# pprint(dataset[0])

# datasets提供的评价方法
from datasets import list_metrics, load_metric
metrics_list = list_metrics()  # 全部评价方法
# print(metrics_list)

# 加载准确率评价方法
accuracy_metric = load_metric('accuracy')
# references真是结果与predictions预测结果计算准确率
results = accuracy_metric.compute(references=[0, 1, 0], predictions=[1, 1, 0])
print(results)