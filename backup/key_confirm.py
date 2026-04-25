import hmac
import hashlib


CONFIRM_DOMAIN = b"PLKG_KEY_CONFIRM_V1"


def make_key_confirm(aes_key: bytes, epoch: int, serial: int, parity: str) -> str:
    message = f"{int(epoch)}|{int(serial)}|{parity}".encode("ascii")
    return hmac.new(aes_key, CONFIRM_DOMAIN + b"|" + message, hashlib.sha256).hexdigest()


def verify_key_confirm(
    aes_key: bytes,
    epoch: int,
    serial: int,
    parity: str,
    confirm_hex: str,
) -> bool:
    expected = make_key_confirm(aes_key, epoch, serial, parity)
    return hmac.compare_digest(expected, confirm_hex)
