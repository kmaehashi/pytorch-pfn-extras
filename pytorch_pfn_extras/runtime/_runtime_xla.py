from pytorch_pfn_extras.runtime import runtime_registry
from pytorch_pfn_extras.runtime import PyTorchRuntime


class XLARuntime(PyTorchRuntime):
    def __init__(
            self,
            device_spec: DeviceLike,
            options: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Defer importing torch_xla.
        import torch_xla.core.xla_model as xm
        self._xm = xm

        if not isinstance(device_spec, torch.device):
            if options is None:
                devkind = 'TPU'
            else:
                devkind = options.get('devkind', 'TPU')
            if ':' in device_spec:
                dev_idx = int(device_spec.split(':')[1])
            else:
                dev_idx = 0
            device_spec = xm.xla_device(n=dev_idx, devkind=devkind)
        super().__init__(device_spec, options)

    def train_post_step(
            self,
            trainer: '_Trainer',
            module: torch.nn.Module,
            batch_idx: int,
            batch: Any,
            outs: Any,
    ) -> None:
        self._xm.optimizer_step(trainer.optimizers['main'], barrier=True)


runtime_registry.register('xla', XLARuntime)