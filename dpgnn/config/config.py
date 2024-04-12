"""Config parser for entire project."""
from ast import literal_eval
import configparser
from dpgnn import logger


class ConfigBase(configparser.ConfigParser):
  """Configurations used for model and data."""

  def __init__(self, fname, **kwargs):
    super().__init__(**kwargs)
    self.read(fname)
    logger.info(f"Reading config: {fname}")

  def get_value(self, x, fallback=None):
    """Gets the value of a config property."""
    try:
      return literal_eval(self.get("DEFAULT", x))
    except configparser.NoOptionError:
      return fallback

  def to_dict(self):
    """Converts the config to a dictionary."""
    config_dict = dict(self.__getitem__("DEFAULT"))  # pylint: disable=unnecessary-dunder-call
    return {k: literal_eval(v) for k, v in config_dict.items()}
