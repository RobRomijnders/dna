"""Test functions for util.py"""
from dpgnn import util, util_wandb
import numpy as np


def test_bootstrap_sampling_ave_precision():
  """Test bootstrap sampling for average precision."""
  # Test 1
  y_true = np.array([0, 0, 1, 1, 1, 1, 1, 0], dtype=np.int32)
  y_pred = np.array(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
  auroc, ave_precision = util.bootstrap_sampling_ave_precision(
    predictions=y_pred, outcome=y_true, num_samples=1)

  assert auroc > 0.5
  assert ave_precision > 0.5


def test_cosine_decay():

  weight_start = 5.
  weight_end = 1.

  result = util.cosine_decay(weight_start, weight_end, 0, 10)
  np.testing.assert_almost_equal(result, weight_start)

  result = util.cosine_decay(weight_start, weight_end, 10, 10)
  np.testing.assert_almost_equal(result, weight_end)

  result = util.cosine_decay(weight_start, weight_end, 5, 10)
  np.testing.assert_almost_equal(result, (5.+1.)/2.)


def test_get_fname_model():
  runner = util_wandb.WandbDummy()
  result = util.get_fname_model(runner.get_url())
  assert result.endswith("_model.pth")
  assert result.startswith("2")
