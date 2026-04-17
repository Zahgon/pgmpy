import logging
import os
from pathlib import Path

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

logger = logging.getLogger("pgmpy")
logger.addHandler(logging.NullHandler())

PGMPY_DATA_HOME = os.path.join(Path.home(), ".pgmpy")


class Config:
    def __init__(self):
        """
        Default configuration initilization.
        """
        self.BACKEND = "numpy"
        self.DTYPE = "float64"
        self.DEVICE = None
        self.SHOW_PROGRESS = True

    def set_device(self, device=None):
        """
        Sets the device if using pytorch backend.

        Parameters
        ----------
        device: str (default: None)
            Either 'cuda': to create arrays on GPU, or 'cpu' to create arrays on CPU.
            If None, sets to cuda if GPU is available else uses CPU.
        """
        pass

    def get_device(self):
        """
        Returns the current backend device.
        """
        pass

    def set_backend(
        self,
        backend: str,
        device: str | None = None,
        dtype=None,
    ):
        """
        Setup the compute backend.

        Parameters
        ----------
        backend: str (numpy or torch)
            Sets the compute backend to `backend`.

        device: str (default: None)
            Sets the device for torch backend. For numpy backend, sets device=None.
            If None, sets device to the first gpu if available else cpu.

        dtype: Instance of numpy.dtype or torch.dtype (default: None)
            Sets the dtype for arrays. If None, sets to either numpy.float64 or
            torch.float64 depending on the backend.
        """
        pass

    def get_backend(self):
        """
        Returns the current backend.
        """
        pass

    def set_show_progress(self, show_progress: bool):
        """
        Sets a global variable to (not) show progress bars.

        Parameters
        ----------
        show_progress: boolean
            If True, shows progress bars, else doesn't.
        """
        pass

    def get_show_progress(self):
        """
        Returns boolean whether to show progress bar or not.
        """
        pass

    def set_dtype(self, dtype=None):
        """
        Sets the dtype for value matrices.

        Parameters
        ----------
        dtype: Instance of numpy.dtype or torch.dtype. (default: None)
            Sets the dtype to `dtype`. If None set to either numpy.float64 or torch.float64 depending on the backend.
        """
        pass

    def get_dtype(self):
        """
        Returns the dtype.
        """
        pass

    def get_compute_backend(self):
        pass


config = Config()
