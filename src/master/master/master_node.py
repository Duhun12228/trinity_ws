"""
Master node: arbitrates the driving mode and is the only publisher of /cmd_vel.

Inputs
  /stanley_cmd_vel (Twist)            lane follow command (normal mode, transition gap)
  /mission/<m>/state (MissionState)   per registered mission: IDLE/ARMED/REQUEST/RUNNING/DONE
  /<m>_cmd_vel (Twist)                per registered mission, used while its mode is active
  /master/set_mode (String)           manual switch, for testing ('normal' ends a mission)
  monitor_* topics                    pipeline health only (raw subscriptions, see monitor.py)
Outputs
  /cmd_vel (Twist)                    final command
  /master/mode (MasterMode)           latched; on every change + once a second

Arbitration (every cycle)
  normal : the mission that has been in REQUEST the longest gets the mode (FIFO).
  mission: ends on DONE, max time, lost state topic, or the node going IDLE (cancelled).
           A REQUEST from a mission listed in `preempt_rules` ("crosswalk>slow") takes over.
  block  : `block_rules` ("3road>cone"): while 3road is not IDLE or asks for a speed limit
           (inside its section), a REQUEST from cone is ignored.
           An interrupted mission goes back to IDLE on its own (mission_base.py), so the
           master returns to normal afterwards instead of resuming it.
Command
  normal : /stanley_cmd_vel.
  mission: its own cmd once one arrived after entering the mode; until then (transition
           gap) /stanley_cmd_vel, and a stop if nothing comes within gap_timeout.
  Any command older than cmd_timeout counts as missing -> stop.
"""
import math
import signal

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String
from trinity_msgs.msg import MasterMode, MissionState

from master.console import Console
from master.modes import Mission, Mode
from master.monitor import build_watches, OK

STATE_NAMES = {MissionState.IDLE: 'IDLE', MissionState.ARMED: 'ARMED',
               MissionState.REQUEST: 'REQUEST', MissionState.RUNNING: 'RUNNING',
               MissionState.DONE: 'DONE'}


def limit_speed(cmd, v_max):
    """Cap the forward speed keeping the steering angle (w scales with v)."""
    if cmd.linear.x <= v_max:
        return cmd
    out = Twist()
    out.linear.x = v_max
    out.angular.z = cmd.angular.z * v_max / cmd.linear.x
    return out


def apply_steer_trim(cmd, trim, wheelbase, max_steer):
    """Add a steering angle offset to the final command (wheel alignment): the angle of
    (v, w) for an Ackermann car, + trim, clipped to +-max_steer, back to w. v = 0 or
    trim = 0: unchanged (a stop stays a stop)."""
    v = cmd.linear.x
    if trim == 0.0 or abs(v) < 1e-3:
        return cmd
    delta = math.atan(cmd.angular.z * wheelbase / v) + trim
    delta = max(-max_steer, min(max_steer, delta))
    out = Twist()
    out.linear.x = v
    out.angular.z = v * math.tan(delta) / wheelbase
    return out


LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)


class MasterNode(Node):

    def __init__(self):
        super().__init__('master_node')
        self.init_params()
        self.init_state()
        self.init_ros()
        names = ', '.join(m.mode.value for m in self.missions.values()) or 'none'
        self.console.info(
            f'master ready: {self.stanley_cmd_topic} -> [mode] -> {self.cmd_vel_topic} '
            f'| missions: {names} (manual: {self.set_mode_topic})')
        self.console.info(
            f'SYSTEM CHECK: waiting for {len(self.watches)} topics '
            f'(max {self.startup_timeout:.0f} s), /cmd_vel held at zero ...')

    # ------------------------------------------------------------------ init
    def init_params(self):
        self.declare_parameter('stanley_cmd_topic', '/stanley_cmd_vel')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('mode_topic', '/master/mode')
        self.declare_parameter('set_mode_topic', '/master/set_mode')
        self.declare_parameter('run_rate_hz', 30.0)
        self.declare_parameter('cmd_timeout', 0.3)
        self.declare_parameter('steer_trim', 0.0)      # rad, added to every final command
        self.declare_parameter('wheelbase', 0.2)
        self.declare_parameter('max_steer', 0.983)
        # missions (parallel lists)
        self.declare_parameter('missions', ['slow'])
        self.declare_parameter('mission_state_topics', ['/mission/slow/state'])
        self.declare_parameter('mission_cmd_topics', ['/slow_cmd_vel'])
        self.declare_parameter('mission_max_sec', [10.0])
        self.declare_parameter('preempt_rules', ['crosswalk>slow'])
        self.declare_parameter('block_rules', ['3road>cone'])
        self.declare_parameter('gap_timeout', 0.5)
        self.declare_parameter('mission_state_timeout', 1.0)
        self.declare_parameter('mission_idle_grace', 1.0)
        # console / monitor
        self.declare_parameter('color', True)
        self.declare_parameter('quiet', False)  # set by the launch file, only for the hint text
        self.declare_parameter('startup_min_sec', 2.0)
        self.declare_parameter('startup_timeout_sec', 20.0)
        self.declare_parameter('status_period_sec', 10.0)
        self.declare_parameter('state_hold_sec', 0.5)
        self.declare_parameter('monitor_timeout', 1.0)
        self.declare_parameter('monitor_window', 2.0)
        self.declare_parameter('monitor_names', ['stanley'])
        self.declare_parameter('monitor_topics', ['/stanley_cmd_vel'])
        self.declare_parameter('monitor_types', ['geometry_msgs/msg/Twist'])
        self.declare_parameter('monitor_min_hz', [10.0])

        p = self.get_parameter
        self.stanley_cmd_topic = p('stanley_cmd_topic').value
        self.cmd_vel_topic = p('cmd_vel_topic').value
        self.mode_topic = p('mode_topic').value
        self.set_mode_topic = p('set_mode_topic').value
        self.run_rate_hz = float(p('run_rate_hz').value)
        self.cmd_timeout = float(p('cmd_timeout').value)
        self.steer_trim = float(p('steer_trim').value)
        self.wheelbase = float(p('wheelbase').value)
        self.max_steer = float(p('max_steer').value)
        self.gap_timeout = float(p('gap_timeout').value)
        self.state_timeout = float(p('mission_state_timeout').value)
        self.idle_grace = float(p('mission_idle_grace').value)
        self.quiet = bool(p('quiet').value)
        self.startup_min = float(p('startup_min_sec').value)
        self.startup_timeout = float(p('startup_timeout_sec').value)
        self.status_period = float(p('status_period_sec').value)
        self.state_hold = float(p('state_hold_sec').value)
        self.console = Console(color=bool(p('color').value))
        self.missions = self.parse_missions()

    def parse_missions(self):
        p = self.get_parameter
        names = list(p('missions').value)
        states = list(p('mission_state_topics').value)
        cmds = list(p('mission_cmd_topics').value)
        max_sec = list(p('mission_max_sec').value)
        if not (len(names) == len(states) == len(cmds) == len(max_sec)):
            raise ValueError('missions / mission_state_topics / mission_cmd_topics / '
                             'mission_max_sec must have the same length')
        missions = {}
        for name, st, cmd, mx in zip(names, states, cmds, max_sec):
            mode = Mode.parse(name)
            if mode is None or mode == Mode.NORMAL:
                raise ValueError(f'bad mission name {name!r}')
            missions[mode] = Mission(mode, st, cmd, float(mx))
        for rule in p('preempt_rules').value:
            hi, _, lo = (s.strip() for s in str(rule).partition('>'))
            hi_m, lo_m = Mode.parse(hi), Mode.parse(lo)
            if hi_m is None or lo_m is None:
                raise ValueError(f'bad preempt rule {rule!r} (use "crosswalk>slow")')
            if hi_m in missions:
                missions[hi_m].preempts.add(lo_m)
        for rule in p('block_rules').value:
            hi, _, lo = (s.strip() for s in str(rule).partition('>'))
            hi_m, lo_m = Mode.parse(hi), Mode.parse(lo)
            if hi_m is None or lo_m is None:
                raise ValueError(f'bad block rule {rule!r} (use "3road>cone")')
            if hi_m in missions:
                missions[hi_m].blocks.add(lo_m)
        return missions

    def init_state(self):
        self.mode = Mode.NORMAL
        self.mode_seq = 0
        self.mode_since = self.now()
        self.mode_reason = 'start'
        self.manual_request = None   # Mode from /master/set_mode, applied in run()
        self.gap_warned = False      # "no cmd from <m>" printed for this entry
        self.stanley_cmd = None
        self.stanley_stamp = None
        self.last_cmd = Twist()
        self.started = False
        self.t_start = self.now()
        self.t_last_wait_log = self.t_start
        self.t_last_summary = None
        self.t_last_mode_pub = None
        self.reported = []
        self.pending = []
        self.reported_mission = {m: (MissionState.IDLE, '') for m in self.missions}
        self.reported_cap = None     # (speed, mission) last printed
        self.reported_block = set()  # blocked modes whose ignored REQUEST was printed

    def init_ros(self):
        p = self.get_parameter
        names = list(p('monitor_names').value)
        topics = list(p('monitor_topics').value)
        types = list(p('monitor_types').value)
        min_hz = list(p('monitor_min_hz').value)
        for m in self.missions.values():  # mission nodes show up in SYSTEM CHECK too
            names.append(f'mission_{m.mode.value}')
            topics.append(m.state_topic)
            types.append('trinity_msgs/msg/MissionState')
            min_hz.append(2.0)
        self.watches = build_watches(self, names, topics, types, min_hz,
                                     p('monitor_timeout').value, p('monitor_window').value)

        self.create_subscription(Twist, self.stanley_cmd_topic, self.stanley_cmd_callback, 10)
        self.create_subscription(String, self.set_mode_topic, self.set_mode_callback, 10)
        for m in self.missions.values():
            self.create_subscription(MissionState, m.state_topic,
                                     lambda msg, m=m: self.mission_state_callback(m, msg), 10)
            self.create_subscription(Twist, m.cmd_topic,
                                     lambda msg, m=m: self.mission_cmd_callback(m, msg), 10)
        self.pub_cmd = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.pub_mode = self.create_publisher(MasterMode, self.mode_topic, LATCHED)
        self.create_timer(1.0 / self.run_rate_hz, self.run)

    # ------------------------------------------------------------- callbacks
    def stanley_cmd_callback(self, msg):
        self.stanley_cmd = msg
        self.stanley_stamp = self.now()

    def mission_state_callback(self, m, msg):
        now = self.now()
        if msg.state == MissionState.REQUEST and m.state != MissionState.REQUEST:
            m.request_since = now
        elif msg.state != MissionState.REQUEST:
            m.request_since = None
        m.state, m.reason, m.phase, m.state_stamp = msg.state, msg.reason, msg.phase, now
        m.speed_limit = float(msg.speed_limit)

    def mission_cmd_callback(self, m, msg):
        m.cmd, m.cmd_stamp = msg, self.now()

    def set_mode_callback(self, msg):
        mode = Mode.parse(msg.data)
        if mode is None:
            self.console.warn(f'unknown mode {msg.data!r}; one of {[x.value for x in Mode]}')
            return
        if mode != Mode.NORMAL and mode not in self.missions:
            self.console.warn(f'mode {mode.value} has no mission node registered -> ignored')
            return
        self.manual_request = mode

    # ------------------------------------------------------------------- run
    def run(self):
        now = self.now()
        statuses = [w.status(now) for w in self.watches]
        if not self.started:
            self.startup_check(now, statuses)
            cmd = Twist()  # hold still until the check is done
        else:
            self.report_topic_changes(now, statuses)
            self.report_mission_changes(now)
            self.arbitrate(now)
            cmd = self.select_cmd(now)
            self.periodic_summary(now, statuses)
        cmd = apply_steer_trim(cmd, self.steer_trim, self.wheelbase, self.max_steer)
        self.last_cmd = cmd
        self.pub_cmd.publish(cmd)
        if self.t_last_mode_pub is None or now - self.t_last_mode_pub >= 1.0:
            self.publish_mode(now)

    # ----------------------------------------------------------- arbitration
    def mission_alive(self, m, now):
        return m.state_stamp is not None and now - m.state_stamp <= self.state_timeout

    def arbitrate(self, now):
        if self.manual_request is not None:
            mode, self.manual_request = self.manual_request, None
            if mode != self.mode:
                self.set_mode(mode, 'manual', now)
            return

        blocked = self.blocked_modes(now)
        requests = sorted((m for m in self.missions.values()
                           if m.state == MissionState.REQUEST and self.mission_alive(m, now)
                           and m.mode not in blocked),
                          key=lambda m: m.request_since)

        if self.mode == Mode.NORMAL:
            if requests:
                self.set_mode(requests[0].mode, f'{requests[0].mode.value} requested '
                              f'({requests[0].reason})', now)
            return

        active = self.missions[self.mode]
        elapsed = now - self.mode_since
        for r in requests:
            if self.mode in r.preempts:
                self.set_mode(r.mode, f'{r.mode.value} preempts {self.mode.value}', now)
                return
        if not self.mission_alive(active, now):
            self.set_mode(Mode.NORMAL, f'{self.mode.value} state lost', now)
        elif active.state == MissionState.DONE:
            why = f' ({active.reason})' if active.reason else ''
            self.set_mode(Mode.NORMAL, f'{self.mode.value} done{why}', now)
        elif active.max_sec > 0 and elapsed > active.max_sec:
            self.set_mode(Mode.NORMAL, f'{self.mode.value} max time {active.max_sec:.0f} s', now)
        elif active.state == MissionState.IDLE and elapsed > self.idle_grace:
            self.set_mode(Mode.NORMAL, f'{self.mode.value} went idle', now)

    def blocked_modes(self, now):
        """
        {blocked mode: blocking mission} from block_rules. A mission blocks while it is alive
        and not IDLE, or asks for a speed limit (inside its section even between steps).
        Ignored REQUESTs are printed once per blocking episode.
        """
        blocked = {}
        for m in self.missions.values():
            if m.blocks and self.mission_alive(m, now) and (
                    m.state != MissionState.IDLE or m.speed_limit > 0.0):
                for b in m.blocks:
                    blocked.setdefault(b, m.mode.value)
        for b, by in blocked.items():
            req = self.missions.get(b)
            if req is not None and req.state == MissionState.REQUEST \
                    and b not in self.reported_block:
                self.console.warn(f'{b.value} request ignored: {by} is active ({req.reason})')
                self.reported_block.add(b)
        self.reported_block &= set(blocked)
        return blocked

    def set_mode(self, new, reason, now):
        self.console.mode_change(self.mode, new, reason)
        self.mode = new
        self.mode_seq += 1
        self.mode_since = now
        self.mode_reason = reason
        self.gap_warned = False
        self.publish_mode(now)

    def publish_mode(self, now):
        msg = MasterMode()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.mode = self.mode.value
        msg.seq = self.mode_seq
        msg.reason = self.mode_reason
        self.pub_mode.publish(msg)
        self.t_last_mode_pub = now

    # --------------------------------------------------------------- command
    def fresh(self, stamp, now, since=None):
        if stamp is None or now - stamp > self.cmd_timeout:
            return False
        return since is None or stamp >= since

    def speed_cap(self, now):
        """Lowest speed_limit asked by a live mission (normal mode), None when none."""
        caps = [(m.speed_limit, m.mode.value) for m in self.missions.values()
                if m.speed_limit > 0.0 and self.mission_alive(m, now)]
        cap = min(caps) if caps else None
        if cap != self.reported_cap:
            if cap is None:
                self.console.info('speed limit off')
            else:
                self.console.info(f'speed limit {cap[0]:.2f} m/s ({cap[1]})')
            self.reported_cap = cap
        return cap[0] if cap else None

    def select_cmd(self, now):
        stanley = self.stanley_cmd if self.fresh(self.stanley_stamp, now) else None
        if stanley is not None and self.mode == Mode.NORMAL:
            cap = self.speed_cap(now)
            if cap is not None:
                stanley = limit_speed(stanley, cap)
        if self.mode == Mode.NORMAL:
            return stanley if stanley is not None else Twist()
        m = self.missions[self.mode]
        if self.fresh(m.cmd_stamp, now, since=self.mode_since):
            return m.cmd
        if now - self.mode_since <= self.gap_timeout:
            return stanley if stanley is not None else Twist()  # transition gap
        if not self.gap_warned:
            self.gap_warned = True
            self.console.warn(f'no {m.cmd_topic} for {self.gap_timeout:.1f} s in '
                              f'[{self.mode.value.upper()}] -> stop')
        return Twist()

    # --------------------------------------------------------------- console
    def startup_check(self, now, statuses):
        waited = now - self.t_start
        all_ok = all(st.state == OK for st in statuses)
        if not ((all_ok and waited >= self.startup_min) or waited >= self.startup_timeout):
            if waited - (self.t_last_wait_log - self.t_start) >= 5.0:
                self.t_last_wait_log = now
                missing = [w.name for w, st in zip(self.watches, statuses) if st.state != OK]
                self.console.info(f'... waiting ({waited:.0f} s): {", ".join(missing)}')
            return
        self.console.system_check(self.watches, statuses, self.mode, waited, self.quiet)
        self.reported = [st.state for st in statuses]
        self.pending = [(st.state, now) for st in statuses]
        self.t_last_summary = now
        self.started = True

    def report_topic_changes(self, now, statuses):
        """Print a topic state change once it has held for state_hold_sec (no flapping)."""
        for i, (w, st) in enumerate(zip(self.watches, statuses)):
            cand, since = self.pending[i]
            if st.state != cand:
                self.pending[i] = (st.state, now)
                continue
            if cand != self.reported[i] and now - since >= self.state_hold:
                self.console.topic_change(w, self.reported[i], st)
                self.reported[i] = cand

    def report_mission_changes(self, now):
        """Print mission state / phase changes (a lost state topic is reported by the monitor)."""
        for mode, m in self.missions.items():
            cur = (m.state, m.phase)
            if cur != self.reported_mission[mode] and self.mission_alive(m, now):
                old = self.state_label(*self.reported_mission[mode])
                self.console.mission_change(mode.value, old, self.state_label(*cur), m.reason)
                self.reported_mission[mode] = cur

    @staticmethod
    def state_label(state, phase):
        name = STATE_NAMES.get(state, '?')
        return f'{name}/{phase}' if phase else name

    def periodic_summary(self, now, statuses):
        if self.status_period <= 0 or now - self.t_last_summary < self.status_period:
            return
        self.t_last_summary = now
        self.console.summary(self.mode, self.watches, statuses, self.last_cmd)

    # ---------------------------------------------------------------- helpers
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def destroy_node(self):
        try:
            self.pub_cmd.publish(Twist())  # leave the robot stopped
        except Exception:  # context already gone
            pass
        super().destroy_node()


_stop_requested = False


def _request_stop(signum, frame):
    # Only set a flag: an exception raised from a signal handler can land inside rclpy's C
    # code (e.g. while taking a message) and come out as a RuntimeError instead.
    global _stop_requested
    _stop_requested = True


def main(args=None):
    # Handle SIGINT / SIGTERM ourselves so the ROS context is still valid when the
    # stop command is published in destroy_node().
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    node = MasterNode()
    try:
        while rclpy.ok() and not _stop_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()  # publishes a stop command
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
