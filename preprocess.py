import torch, argparse
from datasets import load_dataset
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="imdb",
                    help="Path to repository of the dataset to use.")
parser.add_argument("--train_samples", type=int, default=3000,
                    help="Number of training samples to use.")
parser.add_argument("--test_samples", type=int, default=300,
                    help="Number of testing samples to use.")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True)