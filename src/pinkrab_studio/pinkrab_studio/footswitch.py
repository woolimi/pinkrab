"""The foot switch, read for the studio.

Three pedals on the floor ask the robot for the same three things the pedals
on the page do - jog, follow the handheld, stop - and they ask through the same
queue, `POST /real/request` on the studio, so the gate hears one voice. The
unit is a USB keyboard with its keys set to F16, F17 and F19; reading its
keyboard interface directly, rather than waiting for keystrokes to land in a
focused window, is what makes a press count while the operator is looking at
the robot and not at the page.

A process of its own, like the camera and for the same reason: on macOS a
keyboard's HID interface opens for root, and even root is refused while the
application the studio was started from - the terminal - lacks the Input
Monitoring grant, which is given to an application and not to a console
script. So the studio launches this through `sudo -n`, with the sudoers line
beside the camera's, the terminal holds the grant, and it posts to the studio
over localhost. On Linux it
opens as the user who plugged it in (given the udev rule) and sudo is not
involved.

    pinkrab-footswitch                 # posts to the studio on its usual port
    pinkrab-footswitch --port 8730
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

from pinkrab_core.config import SETTINGS

MODE_IDLE = 0
MODE_HANDHELD = 1
MODE_STUDIO = 2

# Retries, in seconds: quick for a cable, slower for a studio not yet up.
RETRY_S = 2.0
READ_TIMEOUT_MS = 200
POST_TIMEOUT_S = 1.0

USAGE_PAGE_DESKTOP = 1
USAGE_KEYBOARD = 6


def keyboard_interface() -> bytes | None:
    """The path of the unit's keyboard interface, or None when it is not plugged in.

    Picked by usage where the backend says one: hidapi over IOKit does, and
    so does hidapi over hidraw. hidapi over libusb - what pip's wheel is on
    Linux - reports no usages at all without the right to open the device,
    which a fresh machine has not given yet, and the unit then looked
    unplugged instead of unopenable. On this unit the keyboard is interface
    0, so that is taken when nothing says otherwise, and opening it is what
    fails - with the line about the udev rule.
    """
    import hid

    vendor, product = SETTINGS.footswitch_usb
    entries = hid.enumerate(vendor, product)
    for entry in entries:
        if entry.get("usage_page") == USAGE_PAGE_DESKTOP and entry.get("usage") == USAGE_KEYBOARD:
            return entry["path"]
    for entry in entries:
        if entry.get("interface_number") == 0:
            return entry["path"]
    return None


# IOHIDDeviceOpen's refusals, as a person would act on them.
IORETURN = {
    -536870174: "macOS refused it even to root: Input Monitoring is off for the app the studio was started "
                "from - System Settings > Privacy & Security > Input Monitoring, turn that app on, then "
                "start the studio again",
    -536870203: "another program holds the foot switch for itself - quit the one that does",
    -536870207: "IOKit refused it as not privileged although this reader is root - check Input Monitoring "
                "for the app the studio was started from",
}
UNOPENABLE_LINUX = ("the foot switch is plugged in but cannot be opened: the udev rule - sudo cp "
                    "scripts/udev/60-pinkrab.rules /etc/udev/rules.d/ && sudo udevadm control --reload "
                    "&& sudo udevadm trigger, then plug it in again")


def why_unopenable(said: str) -> str:
    """Why the keyboard interface would not open, in the words that fix it.

    `said` is hidapi's own error for the open. On macOS it carries IOKit's
    answer as a hex code, and that code is the difference between "needs
    root" said to a reader that is root and the privacy switch that is
    actually off.
    """
    if sys.platform != "darwin":
        return UNOPENABLE_LINUX
    if os.geteuid() != 0:
        return ("the foot switch is plugged in but cannot be opened: on macOS it needs root - the studio runs "
                "this through sudo, see its sudoers line")
    found = re.search(r"\(0x([0-9A-Fa-f]{8})\)", said or "")
    if not found:
        return "the foot switch is plugged in but cannot be opened, and IOKit would not say why"
    code = int(found.group(1), 16) - (1 << 32)   # IOReturn is signed
    return "the foot switch is plugged in but cannot be opened: " + IORETURN.get(
        code, f"IOKit answered {code & 0xFFFFFFFF:#x}")


def pressed_pedals(report: list[int]) -> set[str]:
    """Which pedals a keyboard report says are down.

    A boot keyboard report is a modifier byte, a reserved byte and up to six
    keycodes; this unit may put a report id in front. The keycodes are looked
    for anywhere past the first byte, which is robust to both shapes and to
    the pedals' own codes never being modifier bits.
    """
    return {SETTINGS.footswitch_pedals[code] for code in report[1:] if code in SETTINGS.footswitch_pedals}


class Studio:
    """The studio, as the pedal talks to it: three posts and nothing read back."""

    def __init__(self, port: int) -> None:
        self.base = f"http://127.0.0.1:{port}"

    def post(self, path: str, body: dict) -> bool:
        data = json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=POST_TIMEOUT_S):
                return True
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def pedal(self, name: str, down: bool = True) -> bool:
        return self.post("/real/pedal", {"name": name, "down": down})

    def drive(self, mode: int) -> bool:
        return self.post("/real/request", {"kind": "drive", "mode": mode, "pedal": True})

    def stop(self) -> bool:
        return self.post("/real/request", {"kind": "mode", "mode": MODE_IDLE, "pedal": True})


class Pedals:
    """Turns reports into presses, and presses into what the studio is asked."""

    ASK = {"jog": lambda s: s.drive(MODE_STUDIO),
           "handheld": lambda s: s.drive(MODE_HANDHELD),
           "stop": lambda s: s.stop()}

    def __init__(self, studio: Studio, say=print) -> None:
        self.studio = studio
        self.say = say
        self._down: set[str] = set()

    def press(self, pedal: str) -> None:
        heard = self.studio.pedal(pedal)
        asked = self.ASK[pedal](self.studio)
        self.say(f"{pedal}" + ("" if heard and asked else " - the studio did not answer"))

    def feed(self, report: list[int]) -> None:
        """A report in: act on the pedals that went down since the last one,
        and tell the studio about the ones that came up - a clutch is held."""
        down = pressed_pedals(report)
        for pedal in sorted(down - self._down):
            self.press(pedal)
        for pedal in sorted(self._down - down):
            self.studio.pedal(pedal, down=False)
        self._down = down

    def release_all(self) -> None:
        for pedal in sorted(self._down):
            self.studio.pedal(pedal, down=False)
        self._down = set()


def serve(port: int) -> int:
    try:
        import hid
    except ImportError:
        print("the hidapi package is not installed - pip install hidapi", flush=True)
        return 2
    pedals = Pedals(Studio(port), say=lambda line: print(f"  {time.strftime('%H:%M:%S')}  {line}", flush=True))
    said = None

    def note(line: str) -> None:
        nonlocal said
        if line != said:
            print(f"  {line}", flush=True)
            said = line

    while True:
        path = keyboard_interface()
        if path is None:
            note("no foot switch plugged in - waiting")
            time.sleep(RETRY_S)
            continue
        device = hid.device()
        try:
            device.open_path(path)
        except OSError:
            try:
                answer = device.error()
            except (OSError, ValueError):
                answer = ""
            note(why_unopenable(answer))
            time.sleep(RETRY_S)
            continue
        note(f"foot switch on: 1 jog, 2 follow the handheld, 3 stop  ->  studio :{port}")
        try:
            while True:
                report = device.read(64, READ_TIMEOUT_MS)
                if report:
                    pedals.feed(list(report))
        except OSError:
            note("the foot switch went away")
            pedals.release_all()
        finally:
            device.close()
        time.sleep(RETRY_S)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read the foot switch and press the studio's pedals.")
    parser.add_argument("--port", type=int, default=SETTINGS.studio_port, help="the studio's port")
    parser.add_argument("--parent", type=int, default=0,
                        help="the studio's pid: leave when it is gone (it cannot stop a root process itself)")
    args = parser.parse_args(argv)
    from pinkrab_core.parent import on_term_interrupt, reap_orphans, watch_parent
    on_term_interrupt()
    # Earlier copies nobody could stop are holding the pedal: put them down first.
    reap_orphans("pinkrab-footswitch")
    watch_parent(args.parent)
    try:
        return serve(args.port)
    except KeyboardInterrupt:
        print("\n  stopped", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
