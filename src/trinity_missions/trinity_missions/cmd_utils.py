"""Helpers on lane-follow (stanley) Twist commands."""
from geometry_msgs.msg import Twist


def limit_speed(cmd, v_max):
    """
    Cap the forward speed at v_max keeping the steering angle.

    stanley gives w = v tan(delta) / wheelbase, so w is scaled with v. Commands already
    at or below v_max (corner slowdown, lost line, stop) pass through unchanged.
    """
    out = Twist()
    v, w = cmd.linear.x, cmd.angular.z
    if v > v_max:
        out.linear.x = v_max
        out.angular.z = w * v_max / v
    else:
        out.linear.x, out.angular.z = v, w
    return out
