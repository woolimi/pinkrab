# Which arms hold torque, as the driver last set it.
#
# Published when torque changes and latched (transient local), so a node that
# comes up later - the bridge, and through it the page - hears the current state
# rather than waiting for the next change.

builtin_interfaces/Time stamp
string[] arms
bool[] enabled
