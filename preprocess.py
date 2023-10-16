from datasets import load_dataset
from transformers import AutoTokenizer

def preprocess(dataset, train_samples, test_samples):
    # load pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

    # prepare text for tokenization
    def preprocess_text(examples):
        return tokenizer(examples["text"], truncation=True)

    # load dataset
    dataset = load_dataset(dataset)
    train_dataset = dataset["train"].shuffle(seed=42).select(range(train_samples))
    test_dataset = dataset["test"].shuffle(seed=42).select(range(test_samples))

    # tokenize dataset
    train_dataset = train_dataset.map(preprocess_text, batched=True)
    test_dataset = test_dataset.map(preprocess_text, batched=True)

    return dataset