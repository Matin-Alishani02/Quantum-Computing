# -*- coding: utf-8 -*-
"""Chapter 11: انتقال کوانتومی، کدگذاری چگال و توزیع کلید.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. انتقال همدوس و وفاداری باب
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import (
        Statevector,
        partial_trace,
        state_fidelity,
    )

    theta, phi = 1.1, -0.7
    qc = QuantumCircuit(3)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    # |psi> on q0
    qc.h(1)
    qc.cx(1, 2)
    # Bell pair q1-q2
    qc.cx(0, 1)
    qc.h(0)
    # Bell-basis transform
    qc.cx(1, 2)
    qc.cz(0, 2)
    # deferred corrections
    final = Statevector.from_instruction(qc)
    rho_bob = partial_trace(final, [0, 1])
    target = Statevector(
        [np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)]
    )
    print(state_fidelity(rho_bob, target))

    # Example 02: ۷-۲. آزمون انتقال برای چند حالت
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import (
        Statevector,
        partial_trace,
        state_fidelity,
    )


    def teleport_fidelity(theta, phi):
        qc = QuantumCircuit(3)
        qc.ry(theta, 0)
        qc.rz(phi, 0)
        qc.h(1)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.h(0)
        qc.cx(1, 2)
        qc.cz(0, 2)
        bob = partial_trace(
            Statevector.from_instruction(qc), [0, 1]
        )
        target = Statevector(
            [
                np.cos(theta / 2),
                np.exp(1j * phi) * np.sin(theta / 2),
            ]
        )
        return state_fidelity(bob, target)


    for theta in [0, 0.4, 1.2, np.pi]:
        print(theta, teleport_fidelity(theta, 0.73))

    # Example 03: ۷-۳. چهار پیام کدگذاری چگال
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    for b1 in [0, 1]:
        for b0 in [0, 1]:
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            if b0:
                qc.x(0)
            if b1:
                qc.z(0)
            qc.cx(0, 1)
            qc.h(0)
            probs = Statevector.from_instruction(
                qc
            ).probabilities_dict()
            qiskit_key = max(probs, key=probs.get)
            # q1 q0
            logical = qiskit_key[::-1]
            # q0 q1 = b1 b0
            print(f"message {b1}{b0} -> {logical}")

    # Example 04: ۷-۴. شبیه‌سازی غربال BB84
    import numpy as np

    rng = np.random.default_rng(2026)
    n = 40
    alice_bits = rng.integers(0, 2, n)
    alice_bases = rng.integers(0, 2, n)
    # 0=Z, 1=X
    bob_bases = rng.integers(
        0, 2, n
    )  # Ideal channel: matched basis reproduces the bit; mismatched is random.
    bob_bits = np.where(
        bob_bases == alice_bases,
        alice_bits,
        rng.integers(0, 2, n),
    )
    keep = alice_bases == bob_bases
    print("kept fraction =", keep.mean())
    print("Alice sifted =", alice_bits[keep])
    print("Bob sifted   =", bob_bits[keep])

    # Example 05: ۷-۵. حمله intercept-resend
    import numpy as np

    rng = np.random.default_rng(7)
    n = 200_000
    a = rng.integers(0, 2, n)
    ba = rng.integers(0, 2, n)
    be = rng.integers(0, 2, n)
    e = np.where(be == ba, a, rng.integers(0, 2, n))
    bb = rng.integers(0, 2, n)
    b = np.where(bb == be, e, rng.integers(0, 2, n))
    keep = ba == bb
    qber = np.mean(a[keep] != b[keep])
    print("sifted fraction =", keep.mean(), "QBER =", qber)

    # Example 06: ۷-۶. رمز یک‌بارمصرف روی بیت‌ها
    import numpy as np

    message = np.array([1, 0, 1, 1, 0, 1], dtype=np.uint8)
    key = np.array([0, 1, 1, 0, 1, 0], dtype=np.uint8)
    cipher = message ^ key
    recovered = cipher ^ key
    print("cipher   =", cipher)
    print("recovered=", recovered)
    print(np.array_equal(message, recovered))

    # Example 07: ۷-۷. حاشیه ساده finite-sample
    import numpy as np


    def hoeffding_margin(sample_size, failure_probability):
        return np.sqrt(
            np.log(2 / failure_probability) / (2 * sample_size)
        )


    qber_hat = 0.035
    for m in [1_000, 10_000, 100_000]:
        eps = hoeffding_margin(m, 1e-9)
        print(
            m,
            "estimate=",
            qber_hat,
            "upper toy bound=",
            qber_hat + eps,
        )

    return locals()


if __name__ == "__main__":
    main()
