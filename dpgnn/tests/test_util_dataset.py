"""Test the util_dataset.py file"""
from dpgnn import util_dataset
import numpy as np
import torch


def test_load_data():
  fname = 'dpgnn/tests/test_data.txt'

  num_workers = 2
  ds = util_dataset.MyIterableDataset(
    fname=fname,
    transform_fn=util_dataset.make_features,
    num_workers=num_workers)

  dataloader = torch.utils.data.DataLoader(
    ds, num_workers=num_workers, batch_size=2)

  # Single-process loading
  for num_batch, element in enumerate(dataloader):
    # print(num_batch, element)
    np.testing.assert_array_almost_equal(
      element['features'].shape, [2, 6])

    outcome = element['outcome'].cpu().numpy()
    break

  np.testing.assert_array_almost_equal(outcome, [1, 0])
