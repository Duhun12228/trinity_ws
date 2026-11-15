"""
Common base of the mission nodes.

A mission node always runs. Its trigger part (cheap) watches for the start condition and
reports it to the master through /mission/<name>/state (trinity_msgs/MissionState,
published at state_rate_hz even when unchanged). Its control part only runs while the
master's mode (/master/mode, latched) is this mission.

State flow handled here
  any            --(mode becomes mine)-->      RUNNING   on_start()
  RUNNING        --finish(reason)-->            DONE      (keeps running until the master switches)
  RUNNING / DONE --(mode is no longer mine)-->  IDLE      on_stop(), cooldown starts
The subclass moves IDLE -> ARMED -> REQUEST (set_state), may report steps inside RUNNING
with set_phase(), and decides when to finish().
Leaving the mode for any reason (done, preempted, manual) ends the mission: the node
goes IDLE and does not request again during the cooldown, so the master falls back
to normal instead of resuming it.

suppress_modes: while the master is in one of these modes (e.g. slow during crosswalk) a
pending ARMED / REQUEST is dropped and in_cooldown() is true; after that mode ends,
in_cooldown() stays true for suppress_cooldown more seconds.
"""
from rclpy.exceptions import ParameterUninitializedException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from trinity_msgs.msg import MasterMode, MissionState

STATE_NAMES = {MissionState.IDLE: 'IDLE', MissionState.ARMED: 'ARMED',
               MissionState.REQUEST: 'REQUEST', MissionState.RUNNING: 'RUNNING',
               MissionState.DONE: 'DONE'}

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)


class MissionNode(Node):

    def __init__(self, node_name, mission, topic_name=None):
        super().__init__(node_name)
        self.mission = mission  # mode name, e.g. 'slow' or '3road'
        # ROS names cannot start with a digit: '3road' uses topic_name 'three_road'
        self.declare_parameter('mode_topic', '/master/mode')
        self.declare_parameter('state_topic', f'/mission/{topic_name or mission}/state')
        self.declare_parameter('state_rate_hz', 20.0)
        self.declare_parameter('cooldown', 3.0)
        self.declare_parameter('suppress_modes', Parameter.Type.STRING_ARRAY)
        self.declare_parameter('suppress_cooldown', 3.0)
        p = self.get_parameter
        self.cooldown = float(p('cooldown').value)
        try:
            self.suppress_modes = set(p('suppress_modes').value or [])
        except ParameterUninitializedException:  # not set in the yaml: no suppression
            self.suppress_modes = set()
        self.suppress_cooldown = float(p('suppress_cooldown').value)

        self.state = MissionState.IDLE
        self.reason = ''
        self.phase = ''
        self.speed_limit = 0.0      # m/s, normal-mode speed cap asked of the master, 0 = none
        self.active = False          # master mode == this mission
        self.active_since = None     # s
        self.suppressed_by = None    # master mode currently suppressing this mission
        self.cooldown_until = float('-inf')

        self.pub_state = self.create_publisher(MissionState, p('state_topic').value, 10)
        self.create_subscription(MasterMode, p('mode_topic').value, self.mode_callback, LATCHED)
        self.create_timer(1.0 / float(p('state_rate_hz').value), self._timer)

    # --------------------------------------------------------- for subclasses
    def on_start(self, now):
        """Called when the master switches to this mission."""

    def on_stop(self, now):
        """Called when the master leaves this mission (done, preempted or manual)."""

    def tick(self, now):
        """Called at state_rate_hz before the state is published."""

    def set_state(self, state, reason=''):
        if state == self.state:
            return
        why = f' ({reason})' if reason else ''
        self.get_logger().info(f'{STATE_NAMES[self.state]} -> {STATE_NAMES[state]}{why}')
        self.state = state
        self.reason = reason
        if state != MissionState.RUNNING:
            self.phase = ''

    def set_phase(self, phase, reason=''):
        """Step inside RUNNING, shown on the master dashboard."""
        if phase == self.phase:
            return
        why = f' ({reason})' if reason else ''
        self.get_logger().info(f'phase {self.phase or "-"} -> {phase}{why}')
        self.phase = phase
        self.reason = reason

    def finish(self, reason):
        if self.state == MissionState.RUNNING:
            self.set_state(MissionState.DONE, reason)

    def in_cooldown(self, now):
        return self.suppressed_by is not None or now < self.cooldown_until

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # --------------------------------------------------------------- internals
    def mode_callback(self, msg):
        now = self.now()
        mine = msg.mode == self.mission
        if mine and not self.active:
            self.active = True
            self.active_since = now
            how = 'granted' if self.state == MissionState.REQUEST else f'forced: {msg.reason}'
            self.set_state(MissionState.RUNNING, how)
            self.on_start(now)
        elif not mine and self.active:
            self.active = False
            done = self.state == MissionState.DONE
            self.on_stop(now)
            self.set_state(MissionState.IDLE, 'done' if done else
                           f'interrupted -> {msg.mode} ({msg.reason})')
            self.cooldown_until = now + self.cooldown

        if msg.mode in self.suppress_modes:
            if self.suppressed_by is None:
                self.suppressed_by = msg.mode
                if self.state in (MissionState.ARMED, MissionState.REQUEST):
                    self.set_state(MissionState.IDLE, f'suppressed by {msg.mode}')
        elif self.suppressed_by is not None:
            self.suppressed_by = None
            self.cooldown_until = max(self.cooldown_until, now + self.suppress_cooldown)

    def _timer(self):
        self.tick(self.now())
        msg = MissionState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.mission = self.mission
        msg.state = self.state
        msg.reason = self.reason
        msg.phase = self.phase
        msg.speed_limit = float(self.speed_limit)
        self.pub_state.publish(msg)
