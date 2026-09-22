"""Request shutdown without interrupting an ecological transaction midway."""

import signal


class StopFlag:
    def __init__(self):
        self.requested = False
        self.previous = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.previous[sig] = signal.signal(sig, self._request)

    def _request(self, signum, frame):
        self.requested = True

    def close(self):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)
