import torch, argparse
from datasets import load_dataset
from transformers import AutoTokenizer, DefaultDataCollator, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
from huggingface_hub import notebook_login





# define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="imdb",
                    help="Path to repository of the dataset to use.")
parser.add_argument("-t", "--train_samples", type=int, default=3000,
                    help="Number of training samples to use.")
parser.add_argument("-ts", "--test_samples", type=int, default=300,
                    help="Number of testing samples to use.")
parser.add_argument("-tn", "--text_name", type=str, default="text",
                    help="Name of the text column in the dataset.")
parser.add_argument("-o", "--output_dir", type=str, default="default",
                    help="Path to output directory.")
args = parser.parse_args()





# load dataset
dataset = load_dataset(args.dataset)
small_train_dataset = dataset["train"].shuffle(seed=42).select(range(args.train_samples))
small_test_dataset = dataset["test"].shuffle(seed=42).select(range(args.test_samples))

# loading pretrained DistilBERT tokenizer
# tokenization - breaking text into smaller units (tokens)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# prepare text inputs for model using map method
def preprocess(examples): 
    return tokenizer(examples[args.text_name], truncation=True, padding=True)

tokenized_train = small_train_dataset.map(preprocess, batched=True)
tokenized_test = small_test_dataset.map(preprocess, batched=True)

# convert training samples to PyTorch tensors and concatenate them
data_collator = DefaultDataCollator()





model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# define metrics to evaluate model, accuracy and F1 score
def compute_metrics(eval_pred):
    load_accuracy =  evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}





training_args = TrainingArguments(
    output_dir = args.output_dir,
    learning_rate = 2e-5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    num_train_epochs = 2,
    weight_decay = 0.01,
    save_strategy = "epoch",
    push_to_hub = True,
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_train,
    eval_dataset = tokenized_test,
    tokenizer = tokenizer,
    data_collator = data_collator,
    compute_metrics = compute_metrics,
)



trainer.train()
trainer.evaluate()