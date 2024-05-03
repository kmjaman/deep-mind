import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append('/path/to/project/')

from model.base_model import BaseTransformer
from model.config import TransformerConfig
from training.train import train_model
from evaluation.evaluate import evaluate_model
from data.text_dataset import TextDataset
from utils.vocabulary import Vocabulary
from utils.tokenizer import Tokenizer
from data.preprocess_custom_data import load_custom_data

# Objective function for Optuna
def objective(trial):
    # Define hyperparameters to tune
    num_layers = trial.suggest_int("num_layers", 2, 6)
    num_heads = trial.suggest_int("num_heads", 2, 2)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 512, step=64)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # Prepare dataset
    file_path = "custom_data.txt"
    custom_texts = load_custom_data(file_path)

    vocab = Vocabulary()
    for text in custom_texts:
        for token in text.split():
            vocab.add_token(token)

    tokenizer = Tokenizer(vocab)
    dataset = TextDataset(custom_texts, tokenizer, max_length=20)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Define model
    config = TransformerConfig(
        vocab_size=len(vocab),
        num_layers=num_layers,
        num_heads=num_heads,
        ff_hidden_size=hidden_dim,
        dropout=dropout,
    )
    model = BaseTransformer(config)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.get_index("[PAD]"))

    # Train and evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, dataloader, optimizer, loss_fn, device, epochs=5)
    eval_loss, eval_accuracy = evaluate_model(model, dataloader, loss_fn, device)

    # Return evaluation loss as the optimization metric
    return eval_loss

# Run Optuna tuning
def run_hyperparameter_tuning():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)
    return study.best_params

if __name__ == "__main__":
    best_params = run_hyperparameter_tuning()
