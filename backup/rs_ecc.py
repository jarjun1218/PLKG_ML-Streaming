# ======================================================
#   rs_ecc.py — RS(255,215) 修正 102-bit Key
#   可修 20 bytes (約 160 bits) 的錯誤
# ======================================================

import reedsolo
import base64

# ======================================================
# RS Parameters
# ======================================================
RS_K = 215         # message length (bytes)
RS_N = 255         # total codeword length
RS_ECC = RS_N - RS_K    # ECC = 40 bytes → 可修 20 bytes

rs = reedsolo.RSCodec(RS_ECC)


# ======================================================
# Utility: force bits to EXACT 102 bits
# ======================================================
def force_102_bits(bits: str) -> str:
    if len(bits) > 102:
        return bits[-102:]   # 截尾
    return bits.zfill(102)   # 補零


# ======================================================
# bitstring <-> bytes
# ======================================================
def bits_to_bytes(bits: str) -> bytes:
    bits = force_102_bits(bits)
    n = int(bits, 2)
    return n.to_bytes(13, "big")   # 102 bits = 13 bytes


def bytes_to_bits(b: bytes, length: int) -> str:
    bits = bin(int.from_bytes(b, "big"))[2:]
    return bits.zfill(length)


# ======================================================
# Alice encoding: produce parity only
# ======================================================
def rs_encode_parity_b64(bits102: str) -> str:
    bits102 = force_102_bits(bits102)

    key_bytes = bits_to_bytes(bits102)

    # ★ RS_K 改成 215，因此 padding 改成 (215 - 13)
    padded = key_bytes + b"\x00" * (RS_K - len(key_bytes))

    codeword = rs.encode(padded)

    parity = codeword[-RS_ECC:]  # last 40 bytes
    return base64.b64encode(parity).decode()


# ======================================================
# Bob decode: repair noisy key
# ======================================================
def rs_decode_key(bits102_noisy: str, parity_b64: str) -> str:
    bits102_noisy = force_102_bits(bits102_noisy)

    noisy_bytes = bits_to_bytes(bits102_noisy)

    # ★ padding 長度改成 (215 - 13)
    noisy_bytes = noisy_bytes + b"\x00" * (RS_K - len(noisy_bytes))

    parity = base64.b64decode(parity_b64)

    codeword_guess = noisy_bytes + parity

    corrected = rs.decode(codeword_guess)[0]

    restored_bytes = corrected[:13]   # restore 102 bits
    restored_bits = bytes_to_bits(restored_bytes, 102)

    return force_102_bits(restored_bits)
