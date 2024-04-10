def train_model(model, dataloader, optimizer, loss_fn, device, epochs=1):
    model.train()
    model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            # Forward pass
            logits = model(input_ids)
            logits = logits.view(-1, logits.size(-1))  # Flatten for CrossEntropyLoss
            target_ids = target_ids.view(-1)  # Flatten targets

            loss = loss_fn(logits, target_ids)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
