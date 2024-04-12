import torch

def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    model.to(device)

    total_loss = 0
    correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            # Forward pass
            logits = model(input_ids)
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_ids.view(-1)

            # Calculate loss
            loss = loss_fn(logits, target_ids)
            total_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            mask = target_ids != loss_fn.ignore_index  # Exclude padding tokens
            correct += (predictions[mask] == target_ids[mask]).sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_tokens

    print(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
