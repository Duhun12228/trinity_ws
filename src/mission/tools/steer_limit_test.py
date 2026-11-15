#!/usr/bin/env python3
"""
Steering limit test for the LIMO (Ackermann mode). LIFT THE ROBOT FIRST (wheels off the ground).

Commands an inner-wheel steering angle through /cmd_vel, using the inverse of the
limo_base driver mapping (limo_driver.cpp, twistCmdCallback):
    r = wheelbase / tan(inner) + track / 2,   angular.z = v / r
Each angle is held for --hold seconds; a zero command is sent at the end / on Ctrl+C.

The driver clamps the inner angle to max_inner_angle_ (limo_driver.h, now 38 deg),
so angles above that are not reached unless the clamp is raised.

Examples
  python3 steer_limit_test.py --angles 38 --side left          # is 38 deg already at the stop?
  python3 steer_limit_test.py --angles 38 40 42 44 46 --side both --hold 3
"""
import argparse
import math
import time

from geometry_msgs.msg import Twist
import rclpy

WHEELBASE = 0.2   # m, limo_driver.h wheelbase_
TRACK = 0.172     # m, limo_driver.h track_


def turn_radius(inner_deg):
    """Turning radius [m] for an inner-wheel angle; inf for 0 deg (straight)."""
    if abs(inner_deg) < 1e-6:
        return math.inf
    return WHEELBASE / math.tan(math.radians(abs(inner_deg))) + TRACK / 2.0


def angular_for_inner(inner_deg, v):
    """angular.z that makes the driver command this inner-wheel angle at speed v."""
    r = turn_radius(inner_deg)
    return 0.0 if math.isinf(r) else math.copysign(v / r, inner_deg)


def central_deg(inner_deg):
    r = turn_radius(inner_deg)
    return (0.0 if math.isinf(r) else math.degrees(math.atan(WHEELBASE / r))), r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--angles', type=float, nargs='+', required=True,
                    help='inner wheel angles [deg] to command, in order')
    ap.add_argument('--side', choices=['left', 'right', 'both'], default='left')
    ap.add_argument('--hold', type=float, default=2.0, help='seconds per angle')
    ap.add_argument('--speed', type=float, default=0.05,
                    help='m/s sent with the steering (wheels must be off the ground)')
    ap.add_argument('--topic', default='/cmd_vel')
    args = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node('steer_limit_test')
    pub = node.create_publisher(Twist, args.topic, 10)
    signs = {'left': [1], 'right': [-1], 'both': [1, -1]}[args.side]

    def send(v, w, seconds):
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        end = time.time() + seconds
        while time.time() < end:
            pub.publish(msg)
            time.sleep(0.05)

    try:
        input('Robot lifted, wheels free? Enter to start, Ctrl+C to abort ')
        for sign in signs:
            for a in args.angles:
                c, r = central_deg(a)
                w = angular_for_inner(sign * a, args.speed)
                side = 'left' if sign > 0 else 'right'
                print(f'{side:5s} inner {a:5.1f} deg  (centre {c:4.1f} deg, radius {r:.3f} m)  '
                      f'-> cmd v {args.speed:.2f} w {w:+.3f}')
                send(args.speed, w, args.hold)
            send(0.0, 0.0, 1.0)
    except KeyboardInterrupt:
        pass
    finally:
        send(0.0, 0.0, 0.5)
        print('stopped')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
