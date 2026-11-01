# What the safety node believes about the robot right now.
#
# The safety node owns every field here and publishes them together. A client
# asks for a change with a service and then waits for the next frame of this
# message; it never sets these values on its own, because a page that draws
# its own optimistic guess is a page that can lie about a moving arm.
#
# Published at a steady rate whether or not anything changed, so a client that
# has not seen a frame for a second knows it has lost the robot rather than
# that nothing is happening.

builtin_interfaces/Time stamp

# Whether the gate is passing commands to the driver at all. False until a mode
# is chosen, and false again the moment the stop is called. Whether the servos
# also hold torque is the driver's to say - it owns the bus - and an armed gate
# in front of a limp arm is a normal state, not a contradiction.
bool armed

# One of the MODE_ constants below.
uint8 mode

uint8 MODE_IDLE=0
uint8 MODE_HANDHELD=1
uint8 MODE_STUDIO=2
# A trained policy drives: the rollout node publishes, nothing else does.
uint8 MODE_POLICY=3
# The handhelds correct the policy: the arms taken over where they stand,
# begun held, each jaw held at what the policy last sent until its hand's
# opening reaches it.
uint8 MODE_CORRECT=4

# What the gate did to the last command, empty when it passed untouched.
# Written for a human to read in one line: which two bodies would have met,
# which joint was pulled back to its limit, or why nothing is being passed at
# all.
string gate_reason
