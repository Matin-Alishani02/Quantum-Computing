# -*- coding: utf-8 -*-
"""Chapter 22: رمزنگاری و ارتباطات کوانتومی.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. غربال BB84 با NumPy
    import numpy as np

    rng = np.random.default_rng(2026)
    n = 20000
    a_bits = rng.integers(0, 2, n)
    a_basis = rng.integers(0, 2, n)
    b_basis = rng.integers(0, 2, n)
    b_bits = np.where(
        a_basis == b_basis, a_bits, rng.integers(0, 2, n)
    )
    keep = a_basis == b_basis
    print(keep.mean(), np.mean(a_bits[keep] != b_bits[keep]))

    # Example 02: ۷-۲. افزودن intercept-resend
    eve_basis = rng.integers(0, 2, n)
    eve_bits = np.where(
        eve_basis == a_basis, a_bits, rng.integers(0, 2, n)
    )
    b_bits_eve = np.where(
        b_basis == eve_basis, eve_bits, rng.integers(0, 2, n)
    )
    keep = a_basis == b_basis
    print(np.mean(a_bits[keep] != b_bits_eve[keep]))

    # Example 03: ۷-۳. آنتروپی و نرخ ساده
    def h2(q):
        q = np.clip(q, 1e-15, 1 - 1e-15)
        return -q * np.log2(q) - (1 - q) * np.log2(1 - q)


    for q in [0.01, 0.05, 0.09, 0.11]:
        print(q, 1 - 2 * h2(q))

    # Example 04: ۷-۴. حاشیه اندازه محدود
    def hoeffding_margin(m, epsilon):
        return np.sqrt(np.log(1 / epsilon) / (2 * m))


    for m in [1000, 10000, 1000000]:
        print(m, hoeffding_margin(m, 1e-10))

    # Example 05: ۷-۵. آمار پواسون decoy
    from math import factorial


    def poisson(mu, n):
        return np.exp(-mu) * mu**n / factorial(n)


    for mu in [0.05, 0.2, 0.5]:
        probs = [poisson(mu, k) for k in range(5)]
        print(mu, probs, 1 - probs[0] - probs[1])

    # Example 06: ۷-۶. مدل ساده gain
    def gain(mu, eta, dark=1e-6):
        return 1 - np.exp(-eta * mu) + dark


    for length in [0, 50, 100, 150]:
        eta = 10 ** (-0.2 * length / 10)
        print(length, gain(0.5, eta))

    # Example 07: ۷-۷. hash تپلیتز آموزشی
    def toeplitz_hash(bits, out_len, rng):
        n = len(bits)
        seed = rng.integers(0, 2, n + out_len - 1, dtype=np.uint8)
        T = np.empty((out_len, n), dtype=np.uint8)
        for i in range(out_len):
            for j in range(n):
                T[i, j] = seed[out_len - 1 - i + j]
        return (T @ bits.astype(np.uint8)) % 2


    raw = rng.integers(0, 2, 64, dtype=np.uint8)
    print(toeplitz_hash(raw, 24, rng))

    # Example 08: ۷-۸. حالت بل در Qiskit
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    state = Statevector.from_instruction(qc)
    state.seed(2026)
    print(state.sample_counts(4096))

    # Example 09: ۷-۹. هم‌بستگی E91
    from qiskit.quantum_info import SparsePauliOp

    ZZ = SparsePauliOp.from_list([("ZZ", 1.0)])
    XX = SparsePauliOp.from_list([("XX", 1.0)])
    print(float(np.real(state.expectation_value(ZZ))))
    print(float(np.real(state.expectation_value(XX))))

    # Example 10: ۷-۱۰. شبیه‌سازی اتلاف و dark count
    def detection_trial(p_signal, p_dark, shots=100000):
        signal = rng.random(shots) < p_signal
        dark = rng.random(shots) < p_dark
        clicks = signal | dark
        dark_only = (~signal) & dark
        return clicks.mean(), dark_only.sum() / max(
            clicks.sum(), 1
        )


    print(detection_trial(1e-3, 1e-5))

    # Example 11: ۷-۱۱. زمان انتظار تکرارگر
    def wait_for_two_links(p, trials=50000):
        t1 = rng.geometric(p, size=trials)
        t2 = rng.geometric(p, size=trials)
        return np.mean(np.maximum(t1, t2)), np.quantile(
            np.maximum(t1, t2), [0.5, 0.95]
        )


    print(wait_for_two_links(0.01))

    # Example 12: ۷-۱۲. حساسیت طول کلید
    def toy_key_length(n, q, leak, eps):
        return max(
            0, int(n * (1 - h2(q)) - leak - 2 * np.log2(1 / eps))
        )


    for q in [0.02, 0.05, 0.09]:
        print(q, toy_key_length(100000, q, 18000, 1e-10))

    return locals()


if __name__ == "__main__":
    main()
