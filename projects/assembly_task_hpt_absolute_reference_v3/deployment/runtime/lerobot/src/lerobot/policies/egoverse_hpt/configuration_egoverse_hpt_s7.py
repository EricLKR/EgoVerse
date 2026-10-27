"""Config for EgoVerseHPTS7Policy -- identical to EgoVerseHPTConfig plus the fields needed for live S7
token computation. Separate registered policy type ("egoverse_hpt_s7") since lerobot's PreTrainedConfig
registry keys on this string; a plain subclass of EgoVerseHPTConfig without its own registration would
collide with the base "egoverse_hpt" type.
"""
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.egoverse_hpt.configuration_egoverse_hpt import EgoVerseHPTConfig


@PreTrainedConfig.register_subclass("egoverse_hpt_s7")
@dataclass
class EgoVerseHPTS7Config(EgoVerseHPTConfig):
    # Which raw camera key (BEFORE any resize the base policy's own pipeline does) S7 reads its
    # past/current frame pair from -- should be the SAME physical camera as front_camera_key, since S7 was
    # trained on the front camera view.
    s7_source_camera_key: str = "435if"

    # The validated S7 operating horizon (Phase B's Gate 4/5 results) -- S7 looks for a frame this many
    # seconds behind the current one. Never change this to try to cover a longer horizon: the whole S7
    # investigation found calibrated short-horizon prediction breaks down well before 0.5s.
    s7_delta_seconds: float = 0.1

    # How much frame history to retain (seconds) -- must be a bit more than s7_delta_seconds so a
    # same-or-later frame is always available; a small margin, not a large buffer.
    s7_history_buffer_seconds: float = 0.5

    # Max age (seconds) tolerance when the closest available past frame doesn't land near
    # s7_delta_seconds -- mirrors the timestamp-deviation gate used throughout Phase B (10%), loosened
    # slightly here since a live control loop's cadence is less uniform than a fixed-fps recording.
    s7_timestamp_tolerance_frac: float = 0.25

    def __post_init__(self):
        super().__post_init__()
        if self.s7_history_buffer_seconds <= self.s7_delta_seconds:
            raise ValueError("s7_history_buffer_seconds must exceed s7_delta_seconds.")
