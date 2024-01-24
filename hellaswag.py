from datasets import load_dataset
from transformers import AutoModelForMultipleChoice, AutoTokenizer
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMultipleChoice.from_pretrained(model_name)

dataset = load_dataset('hellaswag')

def encode(examples):
    inputs = [examples['ctx'] + ' ' + choice for choice in examples['endings']]
    return tokenizer(inputs, padding=True, truncation=True, return_tensors="pt", max_length=512)

encoded_dataset = dataset.map(encode, batched=True)

def compute_accuracy(model, dataset):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for example in dataset:
            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in example.items() if k in tokenizer.model_input_names}
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1)
            correct += (prediction == example['label']).sum().item()
            total += len(prediction)

    return correct / total

accuracy = compute_accuracy(model, encoded_dataset['validation'])
print(f"Accuracy: {accuracy}")