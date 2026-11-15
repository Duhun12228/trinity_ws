# Master -> everyone: current driving mode. Latched (transient_local) so late joiners get it.

std_msgs/Header header
string mode       # "normal", "slow", "crosswalk", "3road", "cone", "parking"
uint32 seq        # +1 on every mode change (tells a re-entry of the same mode apart)
string reason     # why the mode changed, e.g. "slow requested", "slow done", "manual"
