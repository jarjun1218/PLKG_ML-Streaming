import hmac
import hashlib


CONFIRM_DOMAIN = b"PLKG_KEY_CONFIRM_V1"


def _serial_text(serial) -> str:
    if isinstance(serial, (tuple, list)):
        return ",".join(str(int(item)) for item in serial)
    return str(serial)


def make_key_confirm(aes_key: bytes, epoch: int, serial, helper: str) -> str:
    message = f"{int(epoch)}|{_serial_text(serial)}|{helper}".encode("ascii")
    return hmac.new(aes_key, CONFIRM_DOMAIN + b"|" + message, hashlib.sha256).hexdigest()


def verify_key_confirm(
    aes_key: bytes,
    epoch: int,
    serial,
    helper: str,
    confirm_hex: str,
) -> bool:
    expected = make_key_confirm(aes_key, epoch, serial, helper)
    return hmac.compare_digest(expected, confirm_hex)
