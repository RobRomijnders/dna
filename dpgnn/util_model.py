"""Model for DPGNN."""
from dpgnn import constants
import torch
from torch import nn


# Define model
class NeuralNetwork(nn.Module):
  """Model."""

  def __init__(self, cfg):
    super().__init__()

    num_features = cfg['num_features']
    layerwidth = cfg['layerwidth']
    self.upper_spectral_norm = torch.tensor(
      cfg['upper_spectral_norm'], dtype=torch.float32)

    def make_linear(arg_in, arg_out):
      if self.upper_spectral_norm > 0:
        return nn.utils.spectral_norm(
          nn.Linear(arg_in, arg_out),
          n_power_iterations=cfg['num_power_iterations'])
      return nn.Linear(arg_in, arg_out)

    layers = [
      make_linear(num_features, layerwidth),
      nn.Dropout(0.1),
      nn.GELU()]
    for _ in range(cfg['num_layers'] - 1):
      layers.append(make_linear(layerwidth, layerwidth))
      layers.append(nn.Dropout(0.1))
      layers.append(nn.GELU())
    layers.append(make_linear(layerwidth, 1))
    self.linear_relu_stack = nn.Sequential(*layers)

  def forward(self, x):
    """Forward pass."""
    logits = self.linear_relu_stack(x)
    return torch.squeeze(logits, 1) + x[:, 0]


class MaxPoolNetwork(nn.Module):
  """Model."""

  def __init__(self, cfg):
    super().__init__()

    num_features = cfg['num_features']
    self.layerwidth = layerwidth = cfg['layerwidth']
    self.upper_spectral_norm = torch.tensor(
      cfg['upper_spectral_norm'], dtype=torch.float32)
    self._do_neural_augment = cfg['do_neural_augment']
    self._probab1 = cfg['probab1']

    def make_linear(arg_in, arg_out):
      if self.upper_spectral_norm > 0:
        return nn.utils.spectral_norm(
          nn.Linear(arg_in, arg_out),
          n_power_iterations=cfg['num_power_iterations'])
      return nn.Linear(arg_in, arg_out)

    # Leaf network for contacts
    layers = [
      make_linear(num_features, layerwidth),
      nn.Dropout(0.1),
      nn.GELU()]
    for _ in range(cfg['num_layers'] - 1):
      layers.append(make_linear(layerwidth, layerwidth))
      layers.append(nn.Dropout(0.1))
      layers.append(nn.GELU())
    layers.append(make_linear(layerwidth, layerwidth))
    self.leaf_network_contact = nn.Sequential(*layers)
    del layers

    # Leaf network for contacts
    layers = [
      make_linear(num_features, layerwidth),
      nn.Dropout(0.1),
      nn.GELU()]
    for _ in range(cfg['num_layers'] - 1):
      layers.append(make_linear(layerwidth, layerwidth))
      layers.append(nn.Dropout(0.1))
      layers.append(nn.GELU())
    layers.append(make_linear(layerwidth, layerwidth))
    self.leaf_network_obs = nn.Sequential(*layers)
    del layers

    # Backbone network
    layers = []
    for _ in range(cfg['num_layers']):
      layers.append(make_linear(layerwidth, layerwidth))
      layers.append(nn.Dropout(0.1))
      layers.append(nn.GELU())
    layers.append(make_linear(layerwidth, 1))
    self.backbone_network = nn.Sequential(*layers)

  def forward(self, x, fn_pred):
    """Forward pass."""
    mask = x[:, :, 0] >= 0
    row_sum = torch.sum(mask, dim=1, keepdim=True) + 1E-9
    mask = (mask / row_sum).unsqueeze(-1).detach()

    # Apply leaf network
    node_features = torch.cat((
      self.leaf_network_contact(x[:, :constants.CTC, :]),
      self.leaf_network_obs(x[:, constants.CTC:, :]),
    ), dim=1)

    half = self.layerwidth // 2

    backbone_features_mean = torch.sum(
      node_features[:, :, :half] * mask, dim=1)
    backbone_features_max = torch.max(
      node_features[:, :, half:], dim=1).values

    backbone_features = torch.cat(
      (backbone_features_mean, backbone_features_max), dim=1)

    # Apply backbone network
    logits = torch.squeeze(self.backbone_network(backbone_features), 1)
    logits *= self._probab1
    if self._do_neural_augment:
      logits = logits + fn_pred
    return logits


class SplitGraphNetwork(nn.Module):
  """Model."""

  def __init__(self, cfg):
    super().__init__()

    num_features = cfg['num_features']
    layerwidth = cfg['layerwidth']
    self.upper_spectral_norm = torch.tensor(
      cfg['upper_spectral_norm'], dtype=torch.float32)
    self._do_neural_augment = cfg['do_neural_augment']
    self._probab1 = cfg['probab1']

    def make_linear(arg_in, arg_out):
      if self.upper_spectral_norm > 0:
        return nn.utils.spectral_norm(
          nn.Linear(arg_in, arg_out),
          n_power_iterations=cfg['num_power_iterations'])
      return nn.Linear(arg_in, arg_out)

    # Leaf network for contacts
    layers = [
      make_linear(num_features, layerwidth),
      nn.Dropout(0.1),
      nn.ReLU()]
    for _ in range(cfg['num_layers'] - 1):
      layers.append(make_linear(layerwidth, layerwidth))
      layers.append(nn.Dropout(0.1))
      layers.append(nn.ReLU())
    layers.append(make_linear(layerwidth, layerwidth))
    self.leaf_network_contact = nn.Sequential(*layers)
    del layers

    # Leaf network for contacts
    layers = [
      make_linear(num_features, layerwidth),
      nn.Dropout(0.1),
      nn.ReLU()]
    for _ in range(cfg['num_layers'] - 1):
      layers.append(make_linear(layerwidth, layerwidth))
      layers.append(nn.Dropout(0.1))
      layers.append(nn.ReLU())
    layers.append(make_linear(layerwidth, layerwidth))
    self.leaf_network_obs = nn.Sequential(*layers)
    del layers

    # Backbone network
    layers = []
    for _ in range(cfg['num_layers']):
      layers.append(make_linear(layerwidth, layerwidth))
      layers.append(nn.Dropout(0.1))
      layers.append(nn.ReLU())
    layers.append(make_linear(layerwidth, 1))
    self.backbone_network = nn.Sequential(*layers)

  def forward(self, x, fn_pred):
    """Forward pass."""
    mask = x[:, :, 0] >= 0
    row_sum = torch.sum(mask, dim=1, keepdim=True) + 1E-9
    mask = (mask / row_sum).unsqueeze(-1).detach()

    # Apply leaf network
    node_features = torch.cat((
      self.leaf_network_contact(x[:, :constants.CTC, :]),
      self.leaf_network_obs(x[:, constants.CTC:, :]),
    ), dim=1)
    backbone_features = torch.sum(node_features * mask, dim=1)

    # Apply backbone network
    logits = torch.squeeze(self.backbone_network(backbone_features), 1)
    logits *= self._probab1
    if self._do_neural_augment:
      logits = logits + fn_pred
    return logits
