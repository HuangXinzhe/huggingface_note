from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "snowfly/bert-finetuned-ner"
token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple"
)
result = token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(result)