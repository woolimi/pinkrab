// The simulation, from the page's side.
//
// The server steps MuJoCo and pushes fourteen joint angles down the page's
// socket (live.js); the page pushes poses back up the same socket.
//
// The split this module exists to keep straight:
//
//   command(pose)  ask the servos for a pose. Physics decides whether they get
//                  there - the desk, the board, the other arm and a force limit
//                  all have a say in between.
//   place(pose)    put the robot at a pose outright. For poses that come from
//                  somewhere else - zero, the servos - which are facts about the
//                  rig rather than requests.
//
// connect() throws when the server has no simulation. There is no reduced page to
// fall back to: contact, the joint limits and the scene's own dimensions all come
// from here, and a viewer that quietly did without them would look the same while
// meaning something else.

import { say, watch } from './live.js';

// How long a burst of commands is gathered before one is sent. About a frame: long
// enough that a pointer drag is one message rather than five, short enough that the
// delay is not something a hand can feel.
const COALESCE_MS = 16;

/** Open the simulation the server is stepping. Throws if there is not one. */
export async function connect() {
  const response = await fetch('/physics');
  const info = await response.json();
  if (!info?.available) {
    throw new Error(info?.reason ?? 'the server is not running a simulation');
  }
  return new Physics(info);
}

class Physics {
  constructor(info) {
    this.info = info;
    this.scene = info.scene;
    // Whether there are arms on a bus behind this. The page says so out loud: a
    // simulation and a simulation with two arms moving beside you on a desk must
    // not look the same.
    this.real = !!info.real;
    this.joints = info.joints ?? [];
    this.limits = info.limits ?? {};
    // The desk and the board, as the scene has them. The page draws these itself,
    // so it has to be told their size by the thing that collides them.
    this.scenery = info.scenery ?? {};
    this.listeners = [];
    this.last = null;

    // Commands are coalesced onto the next frame. A slider drag fires far faster
    // than the simulation can be told anything useful about, and a message per
    // pointer event is a queue that outlives the gesture - the arm then keeps
    // moving after you let go, replaying where you have been.
    this.pending = null;
    this.queued = false;

    watch('physics', (frame) => {
      this.last = frame;
      for (const listener of this.listeners) listener(frame);
    });
  }

  /** Called with every frame the server sends: { qpos, ctrl, contacts, time }. */
  onFrame(listener) { this.listeners.push(listener); }

  /** Ask for a pose. Merged with anything else asked for in the same moment. */
  command(pose) {
    this.pending = { ...(this.pending ?? {}), ...pose };
    if (this.queued) return;
    this.queued = true;
    // A timer, not requestAnimationFrame. A page in a background tab gets no
    // animation frames at all, so a command issued there - by a script, or by a
    // tab that lost focus mid-gesture - would sit in `pending` and never be sent,
    // leaving the arm holding a pose nobody asked it to hold. Timers are throttled
    // in the background rather than stopped, which is the difference that matters.
    setTimeout(() => {
      this.queued = false;
      // Newer joint angles supersede older ones rather than queueing behind
      // them, so what arrives is where the pointer is now. One dropped while
      // the socket is down is not worth interrupting anyone over: the next
      // frame of the gesture carries the same intent.
      if (this.pending) say({ ctrl: this.pending });
      this.pending = null;
    }, COALESCE_MS);
  }

  /** Put the robot at a pose outright, without driving it there. */
  place(pose) {
    this.pending = null;   // whatever was being asked for is moot now
    say({ place: pose });
  }
}
