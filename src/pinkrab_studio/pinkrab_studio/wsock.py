"""WebSocket (RFC 6455), the server's half, on the standard library.

Each page keeps one socket to the studio: the studio says what changed as it
changes, and the page says back what it watches and what it asks of the
simulation. The server is the standard library (see server.py), which has
HTTP and no WebSocket, so the part of the protocol a page uses is here: the
upgrade, text messages both ways, ping and close. No extensions and no
subprotocols - a browser that offers one is answered without it, which the
RFC allows.
"""
from __future__ import annotations

import base64
import hashlib
import select
import socket
import struct
from urllib.parse import urlparse

GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
# What a page says is a pose or a few names. Anything larger is not a page.
MAX_MESSAGE = 1 << 16
# A page that stops reading is let go after this long, rather than holding
# its thread on a send that never ends.
SEND_TIMEOUT_S = 10.0

TEXT, BINARY, CONTINUATION, CLOSE, PING, PONG = 0x1, 0x2, 0x0, 0x8, 0x9, 0xA


class Closed(Exception):
    """The socket is over: closed by either side, or broken."""


def accept(handler) -> WebSocket | None:
    """Answer the upgrade asked on `handler` (a BaseHTTPRequestHandler) and
    return the socket; or answer 400 or 403 and return None.

    A page from another site is refused. A browser opens a WebSocket to any
    host it is told to, with no CORS in the way, and this one moves arms."""
    headers = handler.headers
    key = headers.get("Sec-WebSocket-Key", "").strip()
    if (headers.get("Upgrade", "").lower() != "websocket"
            or "upgrade" not in headers.get("Connection", "").lower()
            or headers.get("Sec-WebSocket-Version") != "13" or not key):
        handler.send_error(400, "expected a WebSocket upgrade")
        return None
    origin = headers.get("Origin")
    if origin is not None and urlparse(origin).netloc != headers.get("Host"):
        handler.send_error(403, "a page from another site")
        return None
    handler.send_response(101, "Switching Protocols")
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept",
                        base64.b64encode(hashlib.sha1(key.encode() + GUID).digest()).decode())
    handler.end_headers()
    handler.close_connection = True
    return WebSocket(handler.connection)


class WebSocket:
    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        # Sends wait this long at most; receives are polled with select.
        sock.settimeout(SEND_TIMEOUT_S)
        self._buffer = bytearray()
        # A message arriving in pieces: its kind and the pieces so far.
        self._kind = 0
        self._parts: list[bytes] = []

    def receive(self, timeout: float) -> list[str]:
        """The text messages that arrive within `timeout` seconds, oldest
        first - none if the page said nothing. Raises Closed."""
        try:
            ready, _, _ = select.select([self._sock], [], [], timeout)
            if not ready:
                return []
            data = self._sock.recv(1 << 16)
        except OSError as exc:
            raise Closed(str(exc)) from exc
        if not data:
            raise Closed("the page went")
        self._buffer += data
        return self._messages()

    def send(self, text: str) -> None:
        self._frame(TEXT, text.encode())

    def ping(self) -> None:
        self._frame(PING, b"")

    def close(self, code: int = 1000) -> None:
        """Say goodbye and let go; quietly, whatever state the socket is in."""
        try:
            self._frame(CLOSE, struct.pack("!H", code))
        except Closed:
            pass
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    # --- frames -----------------------------------------------------------
    def _frame(self, opcode: int, payload: bytes) -> None:
        n = len(payload)
        if n < 126:
            head = struct.pack("!BB", 0x80 | opcode, n)
        elif n < 1 << 16:
            head = struct.pack("!BBH", 0x80 | opcode, 126, n)
        else:
            head = struct.pack("!BBQ", 0x80 | opcode, 127, n)
        try:
            self._sock.sendall(head + payload)
        except OSError as exc:
            raise Closed(str(exc)) from exc

    def _refuse(self, code: int, why: str) -> Closed:
        self.close(code)
        return Closed(why)

    def _messages(self) -> list[str]:
        out = []
        while (frame := self._next_frame()) is not None:
            fin, opcode, payload = frame
            if opcode == CLOSE:
                self.close(struct.unpack("!H", payload[:2])[0] if len(payload) >= 2 else 1000)
                raise Closed("closed by the page")
            if opcode == PING:
                self._frame(PONG, payload)
                continue
            if opcode == PONG:
                continue
            if opcode in (TEXT, BINARY):
                if self._parts:
                    raise self._refuse(1002, "a new message inside an unfinished one")
                self._kind = opcode
            elif opcode == CONTINUATION:
                if not self._parts:
                    raise self._refuse(1002, "a continuation of nothing")
            else:
                raise self._refuse(1002, f"opcode {opcode:#x}")
            self._parts.append(payload)
            if sum(map(len, self._parts)) > MAX_MESSAGE:
                raise self._refuse(1009, "message too big")
            if not fin:
                continue
            message, self._parts = b"".join(self._parts), []
            if self._kind == TEXT:
                try:
                    out.append(message.decode("utf-8"))
                except UnicodeDecodeError:
                    raise self._refuse(1007, "text that is not UTF-8") from None
        return out

    def _next_frame(self) -> tuple[bool, int, bytes] | None:
        """The next whole frame off the buffer, unmasked, or None until more arrives."""
        b = self._buffer
        if len(b) < 2:
            return None
        if b[0] & 0x70:
            raise self._refuse(1002, "reserved bits set, and no extension was agreed")
        fin, opcode = bool(b[0] & 0x80), b[0] & 0x0F
        if not b[1] & 0x80:
            raise self._refuse(1002, "a page's frame must be masked")
        n, at = b[1] & 0x7F, 2
        if n == 126:
            if len(b) < 4:
                return None
            n, at = struct.unpack_from("!H", b, 2)[0], 4
        elif n == 127:
            if len(b) < 10:
                return None
            n, at = struct.unpack_from("!Q", b, 2)[0], 10
        if opcode >= CLOSE and (n > 125 or not fin):
            raise self._refuse(1002, "a control frame too long or in pieces")
        if n > MAX_MESSAGE:
            raise self._refuse(1009, "message too big")
        if len(b) < at + 4 + n:
            return None
        mask = bytes(b[at:at + 4]) * (n // 4 + 1)
        data = bytes(b[at + 4:at + 4 + n])
        del b[:at + 4 + n]
        payload = (int.from_bytes(data, "big") ^ int.from_bytes(mask[:n], "big")).to_bytes(n, "big")
        return fin, opcode, payload
