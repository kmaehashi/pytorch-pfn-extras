from pytorch_pfn_extras.runtime._registry import _RuntimeRegistry  # NOQA
from pytorch_pfn_extras.runtime._runtime import BaseRuntime  # NOQA
from pytorch_pfn_extras.runtime._runtime import PyTorchRuntime  # NOQA

import pytorch_pfn_extras.runtime._runtime_xla  # NOQA


runtime_registry = _RuntimeRegistry(PyTorchRuntime)
