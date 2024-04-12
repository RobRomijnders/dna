"""Utility functions for DPGNN."""
from dpgnn import logger
import numpy as np
import os
import time

from sklearn import metrics


def bootstrap_sampling_ave_precision(predictions, outcome, num_samples=12):
  """Bootstrap samples the performance metrics."""
  results_roc = np.zeros((num_samples))
  results_ap = np.zeros((num_samples))

  for i in range(num_samples):
    if num_samples == 1:
      ind = np.arange(predictions.shape[0])
    else:
      ind = np.random.choice(
        predictions.shape[0], predictions.shape[0], replace=True)

    results_roc[i] = metrics.roc_auc_score(
      y_true=outcome[ind], y_score=predictions[ind])
    results_ap[i] = metrics.average_precision_score(
      y_true=outcome[ind], y_score=predictions[ind])

  q20, q50_auroc, q80 = np.quantile(results_roc, [0.2, 0.5, 0.8])*100
  logger.info(f"ROC: {q50_auroc:5.1f} [{q20:5.1f}, {q80:5.1f}]")

  q20, q50_ap, q80 = np.quantile(results_ap, [0.2, 0.5, 0.8])*100
  logger.info(f"AvP: {q50_ap:5.1f} [{q20:5.1f}, {q80:5.1f}]")
  return q50_auroc, q50_ap


def maybe_make_dir(dirname: str):
  if not os.path.exists(dirname):
    logger.info(os.getcwd())
    logger.info(f"Making data_dir {dirname}")
    os.makedirs(dirname)


def get_cpu_count() -> int:
  # Divide cpu_count among tasks when running multiple tasks via SLURM
  # This is a heuristic for running on shared slurm nodes where os.cpu_count()
  # returns the total number of cpus on the node, not the shared part.
  num_tasks = 1
  slurm_ntasks = os.getenv("SLURM_NTASKS")
  if slurm_ntasks is not None:
    num_tasks = int(slurm_ntasks)
  return int(os.cpu_count() // num_tasks)


def cosine_decay(
    weight_start: float,
    weight_end: float,
    current_step: int,
    total_steps: int):
  """Cosine decay from weight_start to weight_end over total_steps."""
  assert current_step >= 0
  if current_step >= total_steps:
    return weight_end

  cos = (1. + np.cos(np.pi * current_step / total_steps)) / 2.
  return weight_end + (weight_start - weight_end) * cos


def get_fname_model(url):
  """Returns a filename specific for this day and experiment."""
  return time.strftime("%Y%m%d") + "_" + url.split("/")[-1] + "_model.pth"
