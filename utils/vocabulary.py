class Vocabulary:
    def __init__(self):
        self.token_to_index = {}
        self.index_to_token = []
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.start_token = "[CLS]"
        self.end_token = "[SEP]"

        self.add_token(self.pad_token)  # Padding token
        self.add_token(self.unk_token)  # Unknown token
        self.add_token(self.start_token)  # Start of sequence
        self.add_token(self.end_token)  # End of sequence

    def add_token(self, token):
        if token not in self.token_to_index:
            self.index_to_token.append(token)
            self.token_to_index[token] = len(self.index_to_token) - 1

    def get_index(self, token):
        return self.token_to_index.get(token, self.token_to_index[self.unk_token])

    def get_token(self, index):
        return self.index_to_token[index] if index < len(self.index_to_token) else self.unk_token

    def __len__(self):
        return len(self.index_to_token)
    