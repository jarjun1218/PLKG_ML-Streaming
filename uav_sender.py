# uav_sender.py  (LIBRARY, memory-only, epoch-ready)

import socket
import time


class UAVKeySender:
    def __init__(
        self,
        get_key_state,
        gsn_ip,
        port=5007,
        debug=False,
        resend_interval=1.0,
        new_epoch_burst=3,
    ):
        """
        get_key_state() -> (epoch, csi_serial_token, helper_b64, confirm_hex)
        csi_serial_token is usually "serial1,serial2" for two-sample models.
        """
        self.get_key_state = get_key_state
        self.gsn_ip = gsn_ip
        self.debug = debug
        self.port = port
        self.resend_interval = resend_interval
        self.new_epoch_burst = new_epoch_burst

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        last_epoch = -1
        last_send_time = 0.0
        last_wait_log = 0.0

        while True:
            epoch, serial, helper, confirm = self.get_key_state()
            now = time.time()
            if epoch < 0 or serial is None or helper is None or confirm is None:
                if now - last_wait_log >= 5.0:
                    print("[UAV] waiting for pending key material before sending helper")
                    last_wait_log = now
                time.sleep(0.05)
                continue

            should_send = False
            repeat_count = 1

            if epoch != last_epoch:
                should_send = True
                repeat_count = self.new_epoch_burst
            elif now - last_send_time >= self.resend_interval:
                should_send = True

            if not should_send:
                time.sleep(0.05)
                continue

            msg = f"R {epoch} {serial} {helper} {confirm}"
            for _ in range(repeat_count):
                sock.sendto(msg.encode(), (self.gsn_ip, self.port))
                time.sleep(0.01)
            if self.debug:
                print(
                    f"[UAV] send helper+confirm epoch={epoch} "
                    f"serial={serial} to {self.gsn_ip}:{self.port}"
                )

            last_epoch = epoch
            last_send_time = time.time()
