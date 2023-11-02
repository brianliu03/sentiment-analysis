import torch, evaluate, wandb
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer




# start a new wandb run to track this script
wandb.init(

    # set the wandb project where this run will be logged
    project="sentiment-analysis",
    name="bach1.0",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 2e-5,
    "architecture": "Transformer",
    "dataset": "sst2",
    "epochs": 2,
    }
)

wandb.define_metric("accuracy")




# python3 train.py -d sst2 -t 10 -ts 5 -tn sentence -o sst2
dataset_name = "sst2"
text_name = "sentence"
output_dir = "sst2"






# load dataset
dataset = load_dataset(dataset_name)
small_train_dataset = dataset["train"].shuffle(seed=45)
small_test_dataset = dataset["test"].shuffle(seed=45)

# loading pretrained DistilBERT tokenizer
# tokenization - breaking text into smaller units (tokens)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# prepare text inputs for model using map method
def preprocess(examples): 
    return tokenizer(examples[text_name], truncation=True, padding=True)

tokenized_train = small_train_dataset.map(preprocess, batched=True)
tokenized_test = small_test_dataset.map(preprocess, batched=True)

# convert training samples to PyTorch tensors and concatenate them
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)





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
    output_dir = output_dir,
    learning_rate = 2e-5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    num_train_epochs = 2,
    weight_decay = 0.01,
    save_strategy = "epoch",
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

# use weights and biases
# print log file, nicer way is weights and biases, or tensorboard
# logging errors
