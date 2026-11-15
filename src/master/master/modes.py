"""Driving modes and the per-mission settings used by the master's arbitration."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from geometry_msgs.msg import Twist


class Mode(Enum):
    NORMAL = 'normal'
    SLOW = 'slow'
    CROSSWALK = 'crosswalk'
    THREE_ROAD = '3road'
    CONE = 'cone'
    PARKING = 'parking'

    @classmethod
    def parse(cls, name):
        """Mode from its string name ('normal', '3road', ...); None if unknown."""
        name = str(name).strip().lower()
        for m in cls:
            if m.value == name or m.name.lower() == name:
                return m
        return None


@dataclass
class Mission:
    """A registered mission node, as seen by the master."""

    mode: Mode
    state_topic: str
    cmd_topic: str
    max_sec: float                          # safety limit on the time in this mode
    state: int = 0                          # MissionState.state, IDLE when unknown
    reason: str = ''
    phase: str = ''                         # step inside RUNNING, e.g. 'stop'
    speed_limit: float = 0.0                # m/s, normal-mode speed cap it asks for, 0 = none
    state_stamp: Optional[float] = None     # s, last MissionState received
    request_since: Optional[float] = None   # s, when the current REQUEST started (FIFO order)
    cmd: Optional[Twist] = None
    cmd_stamp: Optional[float] = None
    preempts: set = field(default_factory=set)  # modes this mission may interrupt
    # modes whose REQUEST is ignored while this mission is not IDLE (or caps the speed)
    blocks: set = field(default_factory=set)
