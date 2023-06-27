from tqdm.auto import tqdm
import torch

def train(model, train_dataloader, test_dataloader, criterion, optimizer, epochs, device, PATH):

  # Send model to device
  model = model.to(device)

  train_losses, test_losses = [], []
  train_accies, test_accies = [], []
  best_acc = 0

  for epoch in range(epochs):

    train_loss, test_loss = 0, 0
    train_acc, test_acc = 0, 0

    ### Train ###
    model.train()

    for X, y in tqdm(train_dataloader):

      # Send data to device
      X, y = X.to(device), y.to(device)

      # Forwad pass
      outputs = model(X)
      loss = criterion(outputs, y)
      optimizer.zero_grad() # Xóa cái optimizer ở vòng lặp trước

      # Calculate loss per batch
      train_loss += loss.item()
      train_acc += (outputs.argmax(dim=1)==y).sum().item() / len(y)

      # Optimizer & Backward
      loss.backward() # 
      optimizer.step() # update weight

    ### Evaluate ###
    model.eval()
    with torch.inference_mode():

      for X, y in tqdm(test_dataloader):

        # Send data to device
        X, y = X.to(device), y.to(device)

        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Calculate loss per batch
        test_loss += loss.item()
        test_acc += (outputs.argmax(dim=1)==y).sum().item() / len(y)

    # Save stuff
    train_loss, train_acc = train_loss/len(train_dataloader), train_acc/len(train_dataloader)
    test_loss, test_acc = test_loss/len(test_dataloader), test_acc/len(test_dataloader)

    train_losses.append(train_loss), train_accies.append(train_acc)
    test_losses.append(test_loss), test_accies.append(test_acc)

    # Tracking the model
    print(f'Epoch: {epoch}| Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}| Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}')

    # Save the model
    if best_acc < test_acc:
      torch.save(model, PATH)
      best_acc = test_acc

  return train_losses, train_accies, test_losses, test_accies