import re

class Tokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def tokenize(self, text):
        # Basic tokenization: split by spaces and punctuation
        tokens = re.findall(r"\w+|[^\s\w]", text, re.UNICODE)
        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        token_ids = [self.vocabulary.get_index(token) for token in tokens]
        return [self.vocabulary.get_index(self.vocabulary.start_token)] + token_ids + [self.vocabulary.get_index(self.vocabulary.end_token)]

    def decode(self, token_ids):
        tokens = [self.vocabulary.get_token(token_id) for token_id in token_ids]
        return " ".join(tokens)
    