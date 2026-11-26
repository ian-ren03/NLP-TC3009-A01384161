import os, torch
def save_experiment(model, optimizer, history, model_name, save_dir):
  ck = {'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history}

  torch.save(ck, os.path.join(save_dir, f"{model_name}.pt"))
  print(f"Checkpoint guardado en Drive: {model_name}.pt")

def load_experimment(model, model_name, save_dir, device = 'cuda'):
  ck = torch.load(os.path.join(save_dir, f"{model_name}.pt"), map_location = device)
  model.load_state_dict(ck['model_state_dict'])
  model.to(device).eval()
  return model, ck['history']
