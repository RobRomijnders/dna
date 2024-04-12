"""Utility functions to load and process data."""
from dpgnn import constants, logger
import json
import glob
import numpy as np
import torch


class NoContacts(Exception):
  """Custom exception for no contacts in a data row."""


def make_features(data):
  """Converts the JSON to a dictionary of features and outcomes."""
  contacts = np.array(data['contacts'], dtype=np.int64)

  # For now we will have six hardcoded features
  features = np.zeros([6], dtype=np.float32)
  features[0] = data['fn_pred']
  features[1] = data['user_age']

  if len(contacts) > 0:
    # Contact feature 2 is age
    features[2], features[3] = np.quantile(contacts[:, 2] / 10, [0.5, 0.8])
    # Contact feature 3 is pinf
    features[4], features[5] = np.quantile(contacts[:, 3] / 1024, [0.5, 0.8])

  return {
    'fn_pred': data['fn_pred'],
    'features': features,
    'outcome': np.float32(data['sim_state'] == 2 or data['sim_state'] == 1)
  }


def make_features_graph_3(data):
  """Converts the JSON to the graph features."""
  contacts = np.array(data['contacts'], dtype=np.int64)
  observations = np.array(data['observations'], dtype=np.int64)

  if len(contacts) == 0:
    contacts = -1 * torch.ones(size=(constants.CTC, 3), dtype=torch.float32)
  else:
    # Remove sender information
    contacts = np.concatenate((contacts[:, 0:1], contacts[:, 2:]), axis=1)
    contacts = torch.tensor(contacts, dtype=torch.float32)

    contacts = torch.nn.functional.pad(
      contacts, [0, 0, 0, constants.CTC-len(contacts)],
      mode='constant', value=-1.)

    # Column 0 is the timestep
    contacts[:, 1] /= 10  # Column 1 is the age
    contacts[:, 2] /= 1024  # Column 2 is the pinf according to FN

  if len(observations) == 0:
    observations = -1 * torch.ones(size=(14, 3), dtype=torch.float32)
  else:
    observations = torch.tensor(observations, dtype=torch.float32)

    observations = torch.nn.functional.pad(
      observations, [0, 1, 0, 14-len(observations)],
      mode='constant', value=-1.)
  observations[:, 2] = 2.0

  # Concatenate the contacts and observations
  contacts = torch.cat((contacts, observations), dim=0)

  return {
    'fn_pred': torch.tensor(data['fn_pred'], dtype=torch.float32),
    'features': contacts,
    'outcome': np.float32(data['sim_state'] == 2 or data['sim_state'] == 1)
  }


def make_features_graph_2(data):
  """Converts the JSON to the graph features."""
  contacts = np.array(data['contacts'], dtype=np.int64)
  observations = np.array(data['observations'], dtype=np.int64)

  if len(contacts) == 0:
    contacts = -1 * torch.ones(size=(constants.CTC, 2), dtype=torch.float32)
  else:
    # Remove sender information
    contacts = np.concatenate((contacts[:, 0:1], contacts[:, 3:4]), axis=1)
    contacts = torch.tensor(contacts, dtype=torch.float32)

    contacts = torch.nn.functional.pad(
      contacts, [0, 0, 0, constants.CTC-len(contacts)],
      mode='constant', value=-1.)

    contacts[:, 1] /= 1024  # Divide the pinf by 1024 to get a number in [0, 1]

  if len(observations) == 0:
    observations = -1 * torch.ones(size=(14, 2), dtype=torch.float32)
  else:
    observations = torch.tensor(observations, dtype=torch.float32)

    observations = torch.nn.functional.pad(
      observations, [0, 0, 0, 14-len(observations)],
      mode='constant', value=-1.)

  # Concatenate the contacts and observations
  contacts = torch.cat((contacts, observations), dim=0)

  return {
    'fn_pred': torch.tensor(data['fn_pred'], dtype=torch.float32),
    'features': contacts,
    'outcome': np.float32(data['sim_state'] == 2 or data['sim_state'] == 1)
  }


class MyIterableDataset(torch.utils.data.IterableDataset):
  """An iterable dataset that reads from a file."""

  def __init__(self, fname, transform_fn=None, num_workers=1, verbose=False):
    super(MyIterableDataset).__init__()

    # Temporary heuristic to speed up file reads. This assumes that the files
    # are split into 1, 2, or 4 parts.
    assert num_workers in [1, 2, 4], "Only 1, 2, or 4 workers supported"
    self.verbose = verbose

    self.fname = fname
    self.transform_fn = transform_fn

    self._num_samples = None
    self._num_workers = num_workers

  def _get_filenames(self):
    """Return the filenames to read from."""
    worker_id = torch.utils.data.get_worker_info().id

    fnames = glob.glob(self.fname)
    if worker_id is not None:
      fnames = fnames[worker_id::self._num_workers]

    return fnames

  def __len__(self):  # pylint: disable=invalid-length-returned
    """Return the number of samples in the dataset."""
    assert False, "Not implemented yet"

  def __getitem__(self, index):
    """Return the item at the given index."""
    del index
    raise NotImplementedError("Not implemented yet")

  def __iter__(self):
    """Return an iterator over the dataset."""
    fnames_worker = self._get_filenames()
    worker_id = torch.utils.data.get_worker_info().id

    if self.verbose:
      logger.info(f"Worker {worker_id} reading from {fnames_worker}")

    def iterable():
      for fname in fnames_worker:
        with open(fname) as f:
          for line in f:
            # TODO(rob): make json also just a transform function
            data = json.loads(line.rstrip('\n'))

            try:
              if self.transform_fn:
                data = self.transform_fn(data)
            except NoContacts:
              continue

            yield data

    return iterable()
