import socket


CSI_CONTROL_PORT = 5008
CSI_RESET_MESSAGE = b"RESET_CSI"


def send_csi_reset_request(target_ip="255.255.255.255", port=CSI_CONTROL_PORT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    except Exception:
        pass
    try:
        sock.sendto(CSI_RESET_MESSAGE, (target_ip, port))
    finally:
        sock.close()
