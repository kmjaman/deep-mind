import torch

def generate_text(model, tokenizer, input_text, max_length, device, temperature=1.0, top_k=None):
    model.eval()
    model.to(device)

    # Tokenize input text
    input_ids = tokenizer.encode(input_text)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    generated_ids = input_ids  # Start with input tokens

    for _ in range(max_length - len(input_ids)):
        with torch.no_grad():
            logits = model(input_tensor)  # Get model predictions
            next_token_logits = logits[:, -1, :]  # Focus on the last token

            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature

            # Optionally apply top-k filtering
            if top_k is not None:
                top_k_logits, _ = torch.topk(next_token_logits, top_k)
                min_value = top_k_logits[:, -1].unsqueeze(-1)
                next_token_logits[next_token_logits < min_value] = -float("inf")

            # Compute probabilities and sample the next token
            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()

        # Break if end token is generated
        if next_token == tokenizer.vocabulary.get_index("[SEP]"):
            break

        # Append the next token and prepare for the next iteration
        generated_ids.append(next_token)
        input_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device)

    # Decode generated tokens into text
    return tokenizer.decode(generated_ids)
