# Mission node -> master: state of one mission, published periodically (also when unchanged)
# so the master can tell a dead node from an idle one.

uint8 IDLE=0      # nothing going on (also during the post-mission cooldown)
uint8 ARMED=1     # pre-condition seen, waiting for the start point (master only displays it)
uint8 REQUEST=2   # wants to run now; the master grants it by switching /master/mode
uint8 RUNNING=3   # granted and running, publishing its cmd_vel
uint8 DONE=4      # finished; the master returns to normal

std_msgs/Header header
string mission    # mode name, e.g. "slow"
uint8 state
string reason     # why the last state/phase change happened, e.g. "sign_slow x3"
string phase      # optional step inside RUNNING, e.g. "approach", "stop", "wait_green"
float32 speed_limit  # m/s, > 0: the master caps its normal-mode (lane follow) speed at this
                     # while it is set, e.g. inside a section; 0 = no limit
