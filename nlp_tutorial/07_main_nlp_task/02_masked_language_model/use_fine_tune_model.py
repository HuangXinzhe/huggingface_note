from transformers import pipeline

text = ""

mask_filler = pipeline("fill-mask", 
                       model="huggingface-course/distilbert-base-uncased-finetuned-imdb")
preds = mask_filler(text)

for pred in preds:
    print(f">>> {pred['sequence']}")
