import torch
from tqdm.auto import tqdm

def evaluate_text_model(test_dataloader, model, device):
  y_pred = []

  model.eval()
  with torch.inference_mode():

    for X, _ in tqdm(test_dataloader):

      # Forward pass
      y_pred += model(X.to(device)).argmax(dim=1).tolist()

  return y_pred
