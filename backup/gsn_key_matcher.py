# gsn_key_matcher.py  (LIBRARY, memory-only)

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class LiveKDRPlotter:
    def __init__(self, interval_ms=500):
        self.interval_ms = interval_ms
        self.x = []
        self.correction_load = []
        self.idx = 0

        self.fig, self.ax = plt.subplots()

    def update(self, gsn_raw, corrected):
        def kdr(a, b):
            L = min(len(a), len(b))
            return sum(a[i] != b[i] for i in range(L)) / L

        self.x.append(self.idx)
        self.correction_load.append(kdr(gsn_raw, corrected) * 100)
        self.idx += 1

    def _draw(self, _):
        self.ax.clear()
        self.ax.plot(self.x, self.correction_load, label="Corrected bits")
        self.ax.set_ylim(0, 100)
        self.ax.legend()

    def start(self):
        self.ani = animation.FuncAnimation(
            self.fig, self._draw, interval=self.interval_ms
        )
        plt.show()
