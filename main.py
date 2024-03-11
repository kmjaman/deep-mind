import torch
import os

def main():
    print("Welcome to Deep Mind!")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

if __name__ == "__main__":
    main()
