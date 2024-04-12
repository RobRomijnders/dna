"""Test functions in util_model.py"""
from dpgnn import constants, util_model
import numpy as np
import torch


def test_nonzero_gradient():
  device = "cpu"
  num_features = 3

  cfg = {
    'num_features': num_features,
    'num_layers': 2,
    'layerwidth': 16,
    'upper_spectral_norm': 1,
    'do_neural_augment': 1,
    'num_power_iterations': 1,
    'probab1': 0.03,
  }

  models = [
    util_model.SplitGraphNetwork(cfg).to(device),
    util_model.MaxPoolNetwork(cfg).to(device),
  ]

  X = 0.1 + 0.9*torch.rand((32, constants.CTC+14, num_features)).to(device)
  fn_pred = torch.rand((32, 1)).to(device).detach()
  y = torch.rand((32, 1)).to(device)

  for model in models:
    pred = model(X, fn_pred)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(pred, y)

    loss.backward()

    # Check that the gradients are nonzero
    for name, param in model.named_parameters():
      if param.grad is not None:
        print(name, param.grad.shape)
        assert np.sum(np.abs(param.grad.cpu().numpy())) > 1E-9
      else:
        print(f"Skipping {name} because it has no gradient")
