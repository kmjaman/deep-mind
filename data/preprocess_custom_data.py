def load_custom_data(file_path):
    """
    Load custom text data from a file.
    Each line is treated as a separate training sample.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        texts = f.readlines()
    return [line.strip() for line in texts if line.strip()]
