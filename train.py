import torch, argparse
from preprocess import preprocess

# define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="imdb",
                    help="Path to repository of the dataset to use.")
parser.add_argument("--train_samples", type=int, default=3000,
                    help="Number of training samples to use.")
parser.add_argument("--test_samples", type=int, default=300,
                    help="Number of testing samples to use.")
args = parser.parse_args()

dataset = preprocess(args.dataset, args.train_samples, args.test_samples)
