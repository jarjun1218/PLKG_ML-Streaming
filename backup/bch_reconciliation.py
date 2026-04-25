import base64
import hashlib


KEY_BITS = 102
BCH_M = 7
BCH_N = (1 << BCH_M) - 1
BCH_T = 7
PRIMITIVE_POLY = 0x83  # x^7 + x + 1
DOMAIN = b"PLKG-BCH-v1"


def force_102_bits(bits: str) -> str:
    bits = "".join("1" if b == "1" else "0" for b in str(bits))
    if len(bits) > KEY_BITS:
        return bits[-KEY_BITS:]
    return bits.zfill(KEY_BITS)


def _gf_mul_raw(a, b):
    out = 0
    while b:
        if b & 1:
            out ^= a
        b >>= 1
        a <<= 1
        if a & (1 << BCH_M):
            a ^= PRIMITIVE_POLY
    return out & BCH_N


def _build_tables():
    exp = [0] * (BCH_N * 2)
    log = [None] * (BCH_N + 1)
    x = 1
    for i in range(BCH_N):
        exp[i] = x
        log[x] = i
        x = _gf_mul_raw(x, 2)
    if x != 1 or len(set(exp[:BCH_N])) != BCH_N:
        raise RuntimeError("invalid BCH primitive polynomial")
    for i in range(BCH_N, BCH_N * 2):
        exp[i] = exp[i - BCH_N]
    return exp, log


GF_EXP, GF_LOG = _build_tables()


def _gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return GF_EXP[GF_LOG[a] + GF_LOG[b]]


def _gf_div(a, b):
    if b == 0:
        raise ZeroDivisionError("GF division by zero")
    if a == 0:
        return 0
    return GF_EXP[(GF_LOG[a] - GF_LOG[b]) % BCH_N]


def _gf_square(a):
    return _gf_mul(a, a)


def _poly_eval(poly, x):
    out = 0
    for coeff in reversed(poly):
        out = _gf_mul(out, x) ^ coeff
    return out


def _full_syndromes(bits: str, t: int):
    bits = force_102_bits(bits)
    syndromes = []
    for power in range(1, 2 * t + 1):
        acc = 0
        for pos, bit in enumerate(bits):
            if bit == "1":
                acc ^= GF_EXP[(power * pos) % BCH_N]
        syndromes.append(acc)
    return syndromes


def _odd_syndromes(bits: str, t: int):
    full = _full_syndromes(bits, t)
    return [full[i - 1] for i in range(1, 2 * t + 1, 2)]


def _expand_odd_syndromes(odd):
    t = len(odd)
    full = [0] * (2 * t)
    for power, value in zip(range(1, 2 * t + 1, 2), odd):
        full[power - 1] = value
    for power in range(2, 2 * t + 1, 2):
        full[power - 1] = _gf_square(full[(power // 2) - 1])
    return full


def _berlekamp_massey(syndromes):
    c = [1]
    b = [1]
    degree = 0
    offset = 1
    last_discrepancy = 1

    for n, syndrome in enumerate(syndromes):
        discrepancy = syndrome
        for i in range(1, degree + 1):
            if i < len(c):
                discrepancy ^= _gf_mul(c[i], syndromes[n - i])

        if discrepancy == 0:
            offset += 1
            continue

        previous = c[:]
        scale = _gf_div(discrepancy, last_discrepancy)
        needed = len(b) + offset
        if len(c) < needed:
            c.extend([0] * (needed - len(c)))
        for i, coeff in enumerate(b):
            c[i + offset] ^= _gf_mul(scale, coeff)

        if 2 * degree <= n:
            degree = n + 1 - degree
            b = previous
            last_discrepancy = discrepancy
            offset = 1
        else:
            offset += 1

    return c[: degree + 1], degree


def _syndromes_for_positions(positions, t):
    syndromes = []
    for power in range(1, 2 * t + 1):
        acc = 0
        for pos in positions:
            acc ^= GF_EXP[(power * pos) % BCH_N]
        syndromes.append(acc)
    return syndromes


def _decode_error_positions(error_syndromes, t):
    if not any(error_syndromes):
        return []

    locator, degree = _berlekamp_massey(error_syndromes)
    if degree < 1 or degree > t:
        raise ValueError(f"BCH correction failed: locator degree {degree}")

    positions = []
    for pos in range(KEY_BITS):
        x = GF_EXP[(-pos) % BCH_N]
        if _poly_eval(locator, x) == 0:
            positions.append(pos)

    if len(positions) != degree:
        raise ValueError(
            f"BCH correction failed: found {len(positions)} roots for degree {degree}"
        )
    if _syndromes_for_positions(positions, t) != error_syndromes:
        raise ValueError("BCH correction failed: syndrome verification mismatch")
    return positions


def bch_encode_syndrome_b64(bits102: str) -> str:
    odd = _odd_syndromes(bits102, BCH_T)
    payload = bytes([BCH_T]) + bytes(odd)
    return base64.b64encode(payload).decode("ascii")


def _decode_helper(helper_b64: str):
    payload = base64.b64decode(helper_b64)
    if not payload:
        raise ValueError("empty BCH helper")
    t = payload[0]
    odd = list(payload[1:])
    if t < 1 or t > BCH_T or len(odd) != t:
        raise ValueError("invalid BCH helper")
    if any(value > BCH_N for value in odd):
        raise ValueError("invalid BCH syndrome value")
    return t, odd, _expand_odd_syndromes(odd)


def bch_decode_key(bits102_noisy: str, helper_b64: str) -> str:
    t, odd, target_syndromes = _decode_helper(helper_b64)
    local_syndromes = _full_syndromes(bits102_noisy, t)
    error_syndromes = [
        target ^ local
        for target, local in zip(target_syndromes, local_syndromes)
    ]
    error_positions = _decode_error_positions(error_syndromes, t)

    corrected = list(force_102_bits(bits102_noisy))
    for pos in error_positions:
        corrected[pos] = "0" if corrected[pos] == "1" else "1"
    corrected = "".join(corrected)

    if _odd_syndromes(corrected, t) != odd:
        raise ValueError("BCH correction failed: corrected key does not match helper")
    return corrected


def derive_aes_key(bits102: str, helper_b64: str) -> bytes:
    bits = force_102_bits(bits102).encode("ascii")
    helper = helper_b64.encode("ascii")
    return hashlib.sha256(DOMAIN + b"\x00" + bits + b"\x00" + helper).digest()
