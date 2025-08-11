import functools
from typing import Union, List, Dict, Any

from torch import nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict

class ModelWrapper(Stateful):
    """Wrapper for model state dict handling"""

    def __init__(self, model: Union[nn.Module, List[nn.Module]]):
        self.model = [model] if isinstance(model, nn.Module) else model

    def state_dict(self) -> Dict[str, Any]:
        return {k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))
