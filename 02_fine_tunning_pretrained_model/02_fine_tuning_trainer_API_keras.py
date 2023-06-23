from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

"""
数据预处理
"""
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

"""
训练
定义一个TrainingArguments类，它将包含Trainer用于训练和评估的所有超参数。
您唯一必须提供的参数是保存训练模型的目录，以及训练过程中的checkpoint。
对于其余的参数，您可以保留默认值，这对于基本微调应该非常有效。
"""
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")  # push_to_hub=True，训练期间自动将模型上传至hub

"""
定义模型
"""
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

"""
有了模型就可以定义一个Trainer 
通过将之前构造的所有对象传递给它——我们的model，training_args，训练和验证数据集，data_collator和tokenizer 
"""
from transformers import Trainer

# trainer = Trainer(
#     model,
#     training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
# )
#
# trainer.train()  # 模型训练

"""
评估
Trainer.predict()命令来使用我们的模型进行预测
"""
# predictions = trainer.predict(tokenized_datasets["validation"])
# print(predictions.predictions.shape, predictions.label_ids.shape)  # 输出结果是具有三个字段的命名元组：predictions，label_ids和metrics

import numpy as np

# preds = np.argmax(predictions.predictions, axis=-1)


import evaluate

# metric = evaluate.load("glue", "mrpc")
# metric.compute(predictions=preds, references=predictions.label_ids)

"""
评估相关打包
"""


def compute_metrics(eval_preds):  # 评价模型的优劣
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")  # 每个epoch后进行验证
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
