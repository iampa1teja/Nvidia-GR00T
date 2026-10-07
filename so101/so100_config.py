"""SO-101 modalities copied from the saved pick-place data contract.

The historical ``so100_config`` name is retained for GR00T's dataset tools.
These are fresh NEW_EMBODIMENT adapters trained from the official foundation;
the settings describe data and never select a fine-tuned model initialization.
Recorded arm/gripper units are preserved. Sixteen future absolute actions are
masked into the foundation model's action capacity.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


so100_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["room", "wrist"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["single_arm", "gripper"]),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in range(2)
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0], modality_keys=["annotation.human.task_description"]
    ),
}

register_modality_config(so100_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
