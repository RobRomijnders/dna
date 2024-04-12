"""Main training script for neural nets with bounded Lipschitz on ABM data."""
import argparse
from dpgnn.config import config  # pytype: disable=import-error
from dpgnn import logger, LOGGER_FILENAME, util, util_wandb, util_dataset, util_model  # pylint: disable=line-too-long
import numpy as np
import os
import random
import socket
import threading
import time
import traceback
import wandb

import torch
from torch import nn
from torch.optim import lr_scheduler


def train(dataloader, model, loss_fn, optimizer):
  """Trains one epoch of the model."""
  num_samples = 0
  model.train()

  t_start = time.time()
  for num_batch, features in enumerate(dataloader):
    X, y = features['features'], features['outcome']
    fn_pred = features['fn_pred'].to(device).detach()
    X, y = X.to(device), y.to(device)

    # Compute prediction error
    pred = model(X, fn_pred)
    loss = loss_fn(pred, y)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    num_samples += len(X)
    if num_batch % 1000 == 0:
      t_elapsed = time.time() - t_start
      t_per_batch = t_elapsed / (num_batch + 1)

      loss = loss.item()
      logger.info(
        f"loss: {loss:>7f}  [{num_samples:10}] time: {t_per_batch:5.3f}")


def test(dataloader, model, loss_fn, max_num_batches=-1):
  """Tests the full dataset on the model."""
  num_samples = 0
  num_batches = 0
  model.eval()
  test_loss, correct = 0, 0
  with torch.no_grad():
    for num_batch, features in enumerate(dataloader):
      if num_batch >= max_num_batches > 0:
        break

      X, y = features['features'], features['outcome']
      fn_pred = features['fn_pred'].to(device).detach()
      X, y = X.to(device), y.to(device)
      pred = model(X, fn_pred)
      test_loss += loss_fn(pred, y).item()
      num_batches += 1

      correct += ((pred > 0.5) == y).type(torch.float).sum().item()
      num_samples += len(X)
  test_loss /= num_batches
  correct /= num_samples
  logger.info((
    f"Accuracy: {100*correct:5.1f},  Avg loss: {test_loss:>8f}, "
    f"predictions: {pred.mean():5.2f}({torch.std(pred.data):5.2f}) \n"))
  return test_loss


def train_normalized_nn(cfg, runner):
  """Train normalized neural net on ABM data."""
  dirname = cfg['dirname']
  model_type = cfg['model_type']
  batch_size = cfg['batch_size']
  num_epochs = cfg['num_epochs']
  learning_rate = cfg['learning_rate']
  learning_rate_decay = cfg['learning_rate_decay']
  lr_warmup = cfg['lr_warmup']
  upper_norm = cfg['upper_spectral_norm']
  spectral_norm_decay = cfg['spectral_norm_decay']
  num_features = cfg['num_features']

  assert int(spectral_norm_decay) == 0, "Not implemented yet"

  weight_decay = cfg['weight_decay']

  # log stats to Wandb
  # Start a daemon to log to wandb
  daemon_wandb = threading.Thread(
    target=util_wandb.log_to_wandb, args=(runner_global,), daemon=True)
  daemon_wandb.start()

  # Load data
  num_workers = 4  # Hardcoded for now, assumes that the data is in 4 files also
  fname_train = os.path.join(dirname, "train_split*.jl")
  fname_test = os.path.join(dirname, "test_split*.jl")

  transform_fn = util_dataset.make_features
  if model_type > 0:
    if num_features == 2:
      transform_fn = util_dataset.make_features_graph_2
    elif num_features == 3:
      transform_fn = util_dataset.make_features_graph_3
    else:
      raise ValueError((
        'Only implemented for 2 or 3 features. \nTwo features corresponds to '
        'time and pinf. \nThree features corresponds to time, pinf, and age.'))

  # Prepare dataset and dataloader
  ds_train = util_dataset.MyIterableDataset(
    fname=fname_train,
    transform_fn=transform_fn,
    num_workers=num_workers)

  dataloader_train = torch.utils.data.DataLoader(
    ds_train, num_workers=num_workers, batch_size=batch_size)

  ds_test = util_dataset.MyIterableDataset(
    fname=fname_test,
    transform_fn=transform_fn,
    num_workers=num_workers)

  dataloader_test = torch.utils.data.DataLoader(
    ds_test, num_workers=num_workers, batch_size=batch_size)

  if model_type == 0:
    assert num_features == 6, "num_features must be 6 for NeuralNetwork"
    model = util_model.NeuralNetwork(cfg).to(device)
  elif model_type == 1:
    model = util_model.MaxPoolNetwork(cfg).to(device)
  elif model_type == 2:
    model = util_model.SplitGraphNetwork(cfg).to(device)
  else:
    raise ValueError(f"Unknown model_type {model_type}")
  logger.info(model)

  for features in dataloader_test:
    X = features['features']
    y = features['outcome']
    logger.info(f"Shape of X [N, D]: {X.shape}")
    logger.info(f"Shape of y: {y.shape} {y.dtype}")

    assert X.shape[-1] == num_features
    break

  loss_fn = nn.MSELoss()
  optimizer = torch.optim.Adam(  # pytype: disable=module-attr
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay)  # pytype: disable=module-attr
  if learning_rate_decay > 0:
    scheduler = lr_scheduler.StepLR(
      optimizer, step_size=1, gamma=learning_rate_decay)
  else:
    scheduler = lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=num_epochs, eta_min=learning_rate/1000)

  if lr_warmup > 0:
    # In the first 'lr_warmup' iterations, the learning rate is linearly
    # increased from 10% to 100% of the initial learning rate.
    schedule_warmup = lr_scheduler.ConstantLR(
      optimizer, factor=0.1, total_iters=lr_warmup)
    scheduler = lr_scheduler.ChainedScheduler(
      [schedule_warmup, scheduler])

  for t in range(num_epochs):
    # Training and testing
    logger.info(f"Epoch {t+1}\n-------------------------------")
    train(dataloader_train, model, loss_fn, optimizer)

    logger.info('Training set:')
    loss_train = test(dataloader_train, model, loss_fn, max_num_batches=25)
    logger.info('Test set:')
    loss_test = test(dataloader_test, model, loss_fn, max_num_batches=25)

    # Update learning rate
    scheduler.step()

    runner.log({
      "loss_train": loss_train,
      "loss_test": loss_test})
  logger.info("Done!")

  # Final losses
  print("Final train:")
  loss_train = test(dataloader_train, model, loss_fn)
  print("Final test:")
  loss_test = test(dataloader_test, model, loss_fn)
  overfit_ratio = (loss_test - loss_train) / loss_train

  # Calculate AUROC and AvP on test predictions
  predictions_test, fn_input_test, outcome_test = [], [], []
  idx = 0
  with torch.no_grad():
    for features in dataloader_test:
      X, y = features['features'], features['outcome']
      fn_pred = features['fn_pred'].to(device).detach()
      X = X.to(device)
      pred = model(X, fn_pred)  # pylint: disable=not-callable
      pred = pred.cpu().numpy()

      predictions_test.append(pred)
      fn_input_test.append(features['fn_pred'].cpu().numpy())
      outcome_test.append(y.cpu().numpy())
      idx += pred.shape[0]

  predictions_test = np.concatenate(predictions_test)
  fn_input_test = np.concatenate(fn_input_test)
  outcome_test = np.concatenate(outcome_test).astype(np.int32)

  logger.info("Without prediction")
  util.bootstrap_sampling_ave_precision(
    predictions=fn_input_test, outcome=outcome_test)
  logger.info("\n")

  logger.info("After prediction")
  auroc_before, _ = util.bootstrap_sampling_ave_precision(
    predictions=predictions_test, outcome=outcome_test)

  # Calculate singular vectors exactly if upper_norm > 0
  singularvalue_max = -1.
  if upper_norm > 0:
    for module in model.modules():
      if isinstance(module, nn.Linear):
        fn = next(iter(module._forward_pre_hooks.values()))  # pylint: disable=protected-access
        weight = fn.compute_weight(module, False).detach()

        # Costly computation, scales cubically, only do after training
        U_tensor, S_tensor, Vh_tensor = torch.linalg.svd(weight)

        # Set the exact left and right singular vectors
        module.weight_u = U_tensor[:, 0]
        module.weight_v = Vh_tensor[0, :]

        singularvalues = S_tensor.cpu().numpy()

        # Maintain the largest singular value
        singularvalue_max = max((
          np.abs(singularvalues[0]), singularvalue_max))
    logger.info((
      f"Largest singular value found: {singularvalue_max}"))

  logger.info("After setting singular vectors")
  auroc, ave_p = util.bootstrap_sampling_ave_precision(
    predictions=predictions_test, outcome=outcome_test)

  fname_model = util.get_fname_model(runner.get_url())
  torch.save(model.state_dict(), "results/" + fname_model)
  logger.info(f"Saved PyTorch Model State to {fname_model}")

  # Final info to WandB
  runner.log({
    "auroc": auroc,
    "ave_p": ave_p,
    "loss_train": loss_train,
    "loss_test": loss_test,
    "singular_value_max": singularvalue_max,
    "overfit_ratio": overfit_ratio,
    "auroc_diff": auroc - auroc_before})


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description='Train Normalized Neural Net on ABM data.')
  parser.add_argument('--configname', type=str, default='default',
                      help='Name of the config file for the data')

  logger.info(f"SLURM env N_TASKS: {os.getenv('SLURM_NTASKS')}")

  args = parser.parse_args()

  # Set num threads
  torch.set_num_threads(util.get_cpu_count())

  configname = args.configname
  fname_config = f"dpgnn/config/{configname}.ini"

  # Set up locations to store results
  experiment_name = 'train_gnn'
  results_dir_global = f'results/{experiment_name}/{configname}/'
  util.maybe_make_dir(results_dir_global)

  config_obj = config.ConfigBase(fname_config)

  # Start WandB
  config_wandb = config_obj.to_dict()
  config_wandb["configname"] = configname
  config_wandb["cpu_count"] = util.get_cpu_count()

  # WandB tags
  tags = [configname]
  tags.append("local" if (os.getenv('SLURM_JOB_ID') is None) else "slurm")

  do_wandb = 'carbon' not in socket.gethostname()
  if do_wandb:
    runner_global = wandb.init(
      project="dpgnn",
      notes=" ",
      name=None,
      tags=tags,
      config=config_wandb,
    )

    config_wandb = dict(runner_global.config)
    logger.info(config_wandb)
  else:
    runner_global = util_wandb.WandbDummy()

  logger.info(f"Logger filename {LOGGER_FILENAME}")
  logger.info(f"Saving to results_dir_global {results_dir_global}")
  logger.info(f"sweep_id: {os.getenv('SWEEPID')}")
  logger.info(f"slurm_id: {os.getenv('SLURM_JOB_ID')}")
  logger.info(f"slurm_name: {os.getenv('SLURM_JOB_NAME')}")
  logger.info(f"slurm_ntasks: {os.getenv('SLURM_NTASKS')}")

  util_wandb.make_git_log()

  # Get cpu, gpu or mps device for training.
  device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu")
  logger.info(f"Using {device} device hardware")
  logger.info(f"CUDA visible devices: {os.getenv('CUDA_VISIBLE_DEVICES')}")

  if device == 'cuda':
    print(f'Initial GPU memory allocated: {torch.cuda.memory_allocated()}')
    print((
      f"CUDA device {torch.cuda.current_device()} "
      f"out of {torch.cuda.device_count()}"))

  # Set random seed
  seed_value = config_wandb.get("seed", -1)
  if seed_value > 0:
    random.seed(seed_value)
    np.random.seed(seed_value)
  else:
    seed_value = random.randint(0, 999)
  # Random number generator to pass as argument to some imported functions
  arg_rng = np.random.default_rng(seed=seed_value)

  try:
    train_normalized_nn(
      cfg=config_wandb,
      runner=runner_global)

  except Exception as e:
    # This exception sends an WandB alert with the traceback and sweepid
    logger.info(f'Error repr: {repr(e)}')
    traceback_report = traceback.format_exc()

    slurmid = os.getenv('SLURM_JOB_ID')
    if slurmid is not None:
      wandb.alert(
        title=f"Error {os.getenv('SWEEPID')}-{slurmid}",
        text=(
          f"'{configname}'\n"
          + traceback_report)
      )
    raise e

  runner_global.finish()
