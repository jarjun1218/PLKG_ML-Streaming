# uav_sender.py  (LIBRARY, memory-only, epoch-ready)

import socket
import time


class UAVKeySender:
    def __init__(self, get_key_state, gsn_ip, port=5007, debug=False):
        """
        get_key_state() -> (epoch, raw_key, parity_b64)
        """
        self.get_key_state = get_key_state
        self.gsn_ip = gsn_ip
        self.debug = debug
        self.port = port

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        last_epoch = -1

        while True:
            epoch, raw, parity = self.get_key_state()
            if epoch < 0 or epoch == last_epoch:
                time.sleep(0.05)
                continue

            msg = f"K {epoch} {raw} {parity}"
            sock.sendto(msg.encode(), (self.gsn_ip, self.port))
            print(f"[UAV] send parity epoch={epoch}")

            last_epoch = epoch
