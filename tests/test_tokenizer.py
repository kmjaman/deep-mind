import sys
sys.path.append('/path/to/project/')

from utils.vocabulary import Vocabulary
from utils.tokenizer import Tokenizer

def test_tokenizer():
    vocab = Vocabulary()
    for token in ["hello", "world", "this", "is", "a", "test"]:
        vocab.add_token(token)

    tokenizer = Tokenizer(vocab)

    text = "hello world! this is a test."
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print(f"Original text: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    assert decoded.startswith("[CLS]") and decoded.endswith("[SEP]")
    print("Tokenizer test passed!")

if __name__ == "__main__":
    test_tokenizer()
