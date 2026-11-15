"""
Topic health monitor.

Each watched topic gets a raw (not deserialised) subscription, so watching image
topics costs almost nothing. The callback only stores the arrival time; status()
is evaluated from the master timer.

States: OK, SLOW (rate below min_hz), LOST (was received, now older than
timeout), NONE (never received).
"""
from collections import deque
from dataclasses import dataclass

from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_message

OK = 'OK'
SLOW = 'SLOW'
LOST = 'LOST'
NONE = 'NONE'


@dataclass
class TopicStatus:
    state: str
    hz: float     # measured rate over the window, 0 if unknown
    age: float    # s since the last message, inf if never


class TopicWatch:

    def __init__(self, node, name, topic, type_name, min_hz, timeout, window):
        self.name = name
        self.topic = topic
        self.min_hz = float(min_hz)
        self.timeout = float(timeout)
        self.window = float(window)
        self.stamps = deque()
        self.ever = False   # received at least once
        self.clock = node.get_clock()
        node.create_subscription(get_message(type_name), topic, self._callback,
                                 qos_profile_sensor_data, raw=True)

    def _callback(self, _msg):
        self.stamps.append(self.clock.now().nanoseconds * 1e-9)

    def status(self, now):
        while self.stamps and now - self.stamps[0] > self.window:
            self.stamps.popleft()
        if not self.stamps:
            return TopicStatus(LOST if self.ever else NONE, 0.0, float('inf'))
        self.ever = True
        age = now - self.stamps[-1]
        n = len(self.stamps)
        span = self.stamps[-1] - self.stamps[0]
        hz = (n - 1) / span if n >= 2 and span > 0 else 0.0
        if age > self.timeout:
            return TopicStatus(LOST, 0.0, age)
        if n >= 3 and hz < self.min_hz:
            return TopicStatus(SLOW, hz, age)
        return TopicStatus(OK, hz, age)


def build_watches(node, names, topics, types, min_hz, timeout, window):
    n = len(names)
    if not (len(topics) == len(types) == len(min_hz) == n):
        raise ValueError('monitor_names/topics/types/min_hz must have the same length')
    return [TopicWatch(node, names[i], topics[i], types[i], min_hz[i], timeout, window)
            for i in range(n)]
