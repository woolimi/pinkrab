"""A UVC camera's exposure and gain, set while OpenCV streams it.

OpenCV opens the top camera and reads its frames; what it cannot do on macOS
is hold the exposure. AVFoundation has no manual exposure on the Mac, so the
two controls are sent to the camera itself as UVC class requests on its
control pipe, through IOKit - the device stays the kernel's, streaming goes on,
and no root is needed. On Linux the uvcvideo driver offers the same controls as
V4L2 ones on the video node, opened a second time beside the stream.

Either way the values are the camera's own: the exposure in whatever units the
module counts it in, which is not always the 100 us the UVC spec names (its
profile's `exposure_unit_us` in camera.yaml), and the gain over the module's
own range.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import struct
import sys
import uuid
from pathlib import Path


class ControlError(RuntimeError):
    """The camera would not take or give a control."""


# UVC: the camera terminal's and processing unit's selectors, the requests,
# and the auto-exposure mode that is manual.
_CT_AE_MODE = 0x02
_CT_AE_PRIORITY = 0x03
_CT_EXPOSURE_ABSOLUTE = 0x04
_PU_GAIN = 0x04
_SET_CUR, _GET_CUR, _GET_MIN, _GET_MAX = 0x01, 0x81, 0x82, 0x83
_AE_MANUAL = 0x01
_CS_INTERFACE = 0x24
_VC_INPUT_TERMINAL, _VC_PROCESSING_UNIT = 0x02, 0x05
_ITT_CAMERA = 0x0201


class Controls:
    """Exposure and gain of one camera; see `open_controls`."""

    def manual_exposure(self, units: int) -> int:
        """Auto exposure off, the exposure at `units` (clamped to the camera's
        range); the value it now holds."""
        raise NotImplementedError

    def gain_range(self) -> tuple[int, int]:
        raise NotImplementedError

    def gain(self) -> int:
        raise NotImplementedError

    def set_gain(self, value: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


def open_controls(device: int | str, socket: str, usb: tuple[int, int] | None) -> Controls:
    """The controls of the camera OpenCV opened as `device`: on macOS found by
    its location (`socket`, the front of its unique id) and, when its profile
    names them, its USB vendor and product id (`usb`); on Linux by its video
    node (`device`)."""
    if sys.platform == "darwin":
        return _IoKitControls(socket, usb)
    return _V4l2Controls(str(device))


# --- macOS: UVC requests through IOKit -----------------------------------------------

class _DevRequest(ctypes.Structure):
    _fields_ = [("bmRequestType", ctypes.c_uint8), ("bRequest", ctypes.c_uint8),
                ("wValue", ctypes.c_uint16), ("wIndex", ctypes.c_uint16), ("wLength", ctypes.c_uint16),
                ("pData", ctypes.c_void_p), ("wLenDone", ctypes.c_uint32)]


class _Uuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


# IOUSBLib's plug-in, its device interface, and where in the device
# interface's function table the calls used here sit.
_USER_CLIENT = "9dc7b780-9ec0-11d4-a54f-000a27052861"
_PLUGIN = "c244e858-109c-11d4-91d4-0050e4c6426f"
_DEVICE_INTERFACE = "5c8187d0-9ef3-11d4-8b45-000a27052861"
_QUERY_INTERFACE, _RELEASE = 1, 3
_GET_VENDOR, _GET_PRODUCT, _GET_LOCATION = 13, 14, 20
_GET_CONFIGURATION_DESCRIPTOR, _DEVICE_REQUEST = 21, 26


class _IoKitControls(Controls):
    """The camera at one USB location, its control pipe reached through
    IOUSBLib's device interface. Its DeviceRequest needs the device neither
    opened nor seized, so the kernel's UVC driver keeps it and keeps streaming."""

    def __init__(self, socket: str, usb: tuple[int, int] | None) -> None:
        iokit = ctypes.cdll.LoadLibrary(ctypes.util.find_library("IOKit"))
        cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
        cf.CFUUIDGetConstantUUIDWithBytes.restype = ctypes.c_void_p
        cf.CFUUIDGetConstantUUIDWithBytes.argtypes = [ctypes.c_void_p] + [ctypes.c_ubyte] * 16
        iokit.IOServiceMatching.restype = ctypes.c_void_p
        iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
        iokit.IOServiceGetMatchingServices.argtypes = [ctypes.c_uint32, ctypes.c_void_p,
                                                       ctypes.POINTER(ctypes.c_uint32)]
        iokit.IOIteratorNext.restype = ctypes.c_uint32
        iokit.IOIteratorNext.argtypes = [ctypes.c_uint32]
        iokit.IOObjectRelease.argtypes = [ctypes.c_uint32]
        iokit.IOCreatePlugInInterfaceForService.argtypes = [
            ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int32)]

        def cf_uuid(text: str):
            return cf.CFUUIDGetConstantUUIDWithBytes(None, *uuid.UUID(text).bytes)

        services = ctypes.c_uint32()
        if iokit.IOServiceGetMatchingServices(0, iokit.IOServiceMatching(b"IOUSBHostDevice"),
                                              ctypes.byref(services)):
            raise ControlError("the USB devices could not be listed")
        self._device = None
        try:
            while self._device is None:
                service = iokit.IOIteratorNext(services.value)
                if not service:
                    break
                plugin, score = ctypes.c_void_p(), ctypes.c_int32()
                made = iokit.IOCreatePlugInInterfaceForService(service, cf_uuid(_USER_CLIENT), cf_uuid(_PLUGIN),
                                                               ctypes.byref(plugin), ctypes.byref(score))
                iokit.IOObjectRelease(service)
                if made or not plugin:
                    continue
                table = self._table(plugin)
                device, wanted = ctypes.c_void_p(), _Uuid((ctypes.c_ubyte * 16)(*uuid.UUID(_DEVICE_INTERFACE).bytes))
                ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, _Uuid, ctypes.POINTER(ctypes.c_void_p))(
                    table[_QUERY_INTERFACE])(plugin, wanted, ctypes.byref(device))
                ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p)(table[_RELEASE])(plugin)
                if not device:
                    continue
                if (usb is None or self._identity(device) == usb) and self._socket(device).startswith(socket):
                    self._device = device
                else:
                    self._release(device)
        finally:
            iokit.IOObjectRelease(services.value)
        if self._device is None:
            ids = "" if usb is None else f" {usb[0]:04x}:{usb[1]:04x}"
            raise ControlError(f"no camera{ids} at USB location {socket}")
        try:
            table = self._table(self._device)
            self._request = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(_DevRequest))(
                table[_DEVICE_REQUEST])
            descriptor = ctypes.c_void_p()
            if ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint8, ctypes.POINTER(ctypes.c_void_p))(
                    table[_GET_CONFIGURATION_DESCRIPTOR])(self._device, 0, ctypes.byref(descriptor)):
                raise ControlError("the camera's configuration could not be read")
            total = struct.unpack("<H", ctypes.string_at(descriptor.value + 2, 2))[0]
            self._interface, self._terminal, self._unit = _video_control(ctypes.string_at(descriptor.value, total))
        except ControlError:
            self.close()
            raise

    @staticmethod
    def _table(interface: ctypes.c_void_p):
        return ctypes.cast(interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents

    def _identity(self, device: ctypes.c_void_p) -> tuple[int, int]:
        vendor, product = ctypes.c_uint16(), ctypes.c_uint16()
        table = self._table(device)
        for slot, out in ((_GET_VENDOR, vendor), (_GET_PRODUCT, product)):
            ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16))(
                table[slot])(device, ctypes.byref(out))
        return vendor.value, product.value

    def _socket(self, device: ctypes.c_void_p) -> str:
        """The location as the front of AVFoundation's unique id spells it."""
        here = ctypes.c_uint32()
        ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))(
            self._table(device)[_GET_LOCATION])(device, ctypes.byref(here))
        return f"{here.value:#x}"

    def _release(self, device: ctypes.c_void_p) -> None:
        ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p)(self._table(device)[_RELEASE])(device)

    def _transfer(self, request: int, selector: int, entity: int, size: int, value: int | None = None) -> int:
        if self._device is None:
            raise ControlError("the camera's controls are closed")
        out = request == _SET_CUR
        data = (ctypes.c_ubyte * size)(*(value.to_bytes(size, "little", signed=False) if out else bytes(size)))
        req = _DevRequest(0x21 if out else 0xA1, request, selector << 8, (entity << 8) | self._interface,
                          size, ctypes.cast(data, ctypes.c_void_p), 0)
        code = self._request(self._device, ctypes.byref(req))
        if code:
            raise ControlError(f"the camera refused control {selector:#x} of entity {entity} "
                               f"(request {request:#x}): {code & 0xFFFFFFFF:#x}")
        return int.from_bytes(bytes(data), "little")

    def manual_exposure(self, units: int) -> int:
        self._transfer(_SET_CUR, _CT_AE_MODE, self._terminal, 1, _AE_MANUAL)
        try:
            # Frame rate before exposure, where the camera says it has the choice.
            self._transfer(_SET_CUR, _CT_AE_PRIORITY, self._terminal, 1, 0)
        except ControlError:
            pass
        low = self._transfer(_GET_MIN, _CT_EXPOSURE_ABSOLUTE, self._terminal, 4)
        high = self._transfer(_GET_MAX, _CT_EXPOSURE_ABSOLUTE, self._terminal, 4)
        self._transfer(_SET_CUR, _CT_EXPOSURE_ABSOLUTE, self._terminal, 4, min(max(units, low), high))
        return self._transfer(_GET_CUR, _CT_EXPOSURE_ABSOLUTE, self._terminal, 4)

    def gain_range(self) -> tuple[int, int]:
        return (self._transfer(_GET_MIN, _PU_GAIN, self._unit, 2), self._transfer(_GET_MAX, _PU_GAIN, self._unit, 2))

    def gain(self) -> int:
        return self._transfer(_GET_CUR, _PU_GAIN, self._unit, 2)

    def set_gain(self, value: int) -> None:
        self._transfer(_SET_CUR, _PU_GAIN, self._unit, 2, value)

    def close(self) -> None:
        if self._device is not None:
            self._release(self._device)
            self._device = None


def _video_control(config: bytes) -> tuple[int, int, int]:
    """(interface, camera terminal id, processing unit id) of the video control
    interface, from a configuration descriptor."""
    interface = terminal = unit = None
    # Whether the descriptors being read are the control interface's: the
    # streaming interfaces' class descriptors that follow reuse the subtypes.
    in_control = False
    at = 0
    while at + 2 < len(config):
        length, kind = config[at], config[at + 1]
        if length == 0:
            break
        if kind == 0x04:   # an interface: video (0x0E), control (0x01)
            in_control = config[at + 5] == 0x0E and config[at + 6] == 0x01
            if in_control:
                interface = config[at + 2]
        elif kind == _CS_INTERFACE and in_control:
            subtype = config[at + 2]
            if subtype == _VC_INPUT_TERMINAL and struct.unpack_from("<H", config, at + 4)[0] == _ITT_CAMERA:
                terminal = config[at + 3]
            elif subtype == _VC_PROCESSING_UNIT:
                unit = config[at + 3]
        at += length
    if interface is None or terminal is None or unit is None:
        raise ControlError("the camera's descriptors name no camera terminal and processing unit")
    return interface, terminal, unit


# --- Linux: V4L2 controls on the video node -------------------------------------------

_V4L2_CID_EXPOSURE_AUTO = 0x009A0901
_V4L2_CID_EXPOSURE_ABSOLUTE = 0x009A0902
_V4L2_CID_EXPOSURE_AUTO_PRIORITY = 0x009A0903
_V4L2_CID_GAIN = 0x00980913
_V4L2_EXPOSURE_MANUAL = 1
_VIDIOC_G_CTRL = 0xC008561B
_VIDIOC_S_CTRL = 0xC008561C
_VIDIOC_QUERYCTRL = 0xC0445624
_QUERYCTRL = struct.Struct("<II32siiiiI8x")


class _V4l2Controls(Controls):
    def __init__(self, path: str) -> None:
        import os

        self._fd = os.open(str(Path(path).resolve()), os.O_RDWR)

    def _ioctl(self, request: int, payload: bytes) -> bytes:
        import fcntl

        buffer = bytearray(payload)
        try:
            fcntl.ioctl(self._fd, request, buffer)
        except OSError as exc:
            raise ControlError(f"the camera refused a control: {exc}") from exc
        return bytes(buffer)

    def _get(self, control: int) -> int:
        return struct.unpack("<Ii", self._ioctl(_VIDIOC_G_CTRL, struct.pack("<Ii", control, 0)))[1]

    def _set(self, control: int, value: int) -> None:
        self._ioctl(_VIDIOC_S_CTRL, struct.pack("<Ii", control, value))

    def _range(self, control: int) -> tuple[int, int]:
        fields = _QUERYCTRL.unpack(self._ioctl(_VIDIOC_QUERYCTRL, _QUERYCTRL.pack(control, 0, b"", 0, 0, 0, 0, 0)))
        return fields[3], fields[4]

    def manual_exposure(self, units: int) -> int:
        self._set(_V4L2_CID_EXPOSURE_AUTO, _V4L2_EXPOSURE_MANUAL)
        try:
            self._set(_V4L2_CID_EXPOSURE_AUTO_PRIORITY, 0)
        except ControlError:
            pass
        low, high = self._range(_V4L2_CID_EXPOSURE_ABSOLUTE)
        self._set(_V4L2_CID_EXPOSURE_ABSOLUTE, min(max(units, low), high))
        return self._get(_V4L2_CID_EXPOSURE_ABSOLUTE)

    def gain_range(self) -> tuple[int, int]:
        return self._range(_V4L2_CID_GAIN)

    def gain(self) -> int:
        return self._get(_V4L2_CID_GAIN)

    def set_gain(self, value: int) -> None:
        self._set(_V4L2_CID_GAIN, value)

    def close(self) -> None:
        import os

        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
