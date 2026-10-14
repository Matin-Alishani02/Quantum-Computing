# -*- coding: utf-8 -*-
"""Chapter 20: کاهش خطا و تصحیح خطای کوانتومی.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. کد bit flip در Qiskit
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, state_fidelity


    def encoded_state(theta):
        qc = QuantumCircuit(3)
        qc.ry(theta, 0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        return qc


    print(Statevector.from_instruction(encoded_state(0.7)))

    # Example 02: ۷-۲. تزریق و بازیابی یک خطای X
    def corrected(theta, error_qubit=None):
        qc = encoded_state(theta)
        if error_qubit is not None:
            qc.x(error_qubit)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.ccx(2, 1, 0)
        return Statevector.from_instruction(qc)


    for q in [None, 0, 1, 2]:
        print(q, corrected(0.7, q).probabilities())

    # Example 03: ۷-۳. جدول syndrome پائولی
    errors = {
        "I": (1, 1),
        "X0": (-1, 1),
        "X1": (-1, -1),
        "X2": (1, -1),
    }
    for name, signs in errors.items():
        bits = "".join("0" if s == 1 else "1" for s in signs)
        print(name, bits)

    # Example 04: ۷-۴. خطای منطقی repetition
    import numpy as np


    def p_logical(p):
        return 3 * p * p * (1 - p) + p**3


    for p in [1e-3, 1e-2, 0.1, 0.4]:
        print(p, p_logical(p))

    # Example 05: ۷-۵. ZNE خطی روی داده مصنوعی
    scales = np.array([1.0, 2.0, 3.0])
    values = np.array([0.84, 0.72, 0.61])
    coef = np.polyfit(scales, values, deg=1)
    zne = np.polyval(coef, 0.0)
    print(zne)

    # Example 06: ۷-۶. bootstrap عدم قطعیت ZNE
    rng = np.random.default_rng(2026)
    boots = []
    for _ in range(2000):
        noisy = values + rng.normal(0, 0.015, size=3)
        boots.append(np.polyval(np.polyfit(scales, noisy, 1), 0))
    print(np.mean(boots), np.std(boots, ddof=1))

    # Example 07: ۷-۷. وارون‌سازی readout
    M = np.array([[0.97, 0.08], [0.03, 0.92]])
    q_obs = np.array([0.55, 0.45])
    p_est = np.linalg.solve(M, q_obs)
    print(p_est, p_est.sum(), np.linalg.cond(M))

    # Example 08: ۷-۸. جبر پایدارساز
    # Symplectic Pauli: (x|z); anticommute if x.z'+z.x' is odd.
    def anticommutes(a, b):
        x, z = a
        xp, zp = b
        return (np.dot(x, zp) + np.dot(z, xp)) % 2 == 1


    Z0Z1 = (np.array([0, 0, 0]), np.array([1, 1, 0]))
    X1 = (np.array([0, 1, 0]), np.array([0, 0, 0]))
    print(anticommutes(Z0Z1, X1))

    # Example 09: ۷-۹. تخمین فاصله لازم
    def choose_distance(p_ratio, target, A=0.1):
        for d in range(3, 101, 2):
            if A * p_ratio ** ((d + 1) / 2) <= target:
                return d
        return None


    print(choose_distance(0.2, 1e-9))

    # Example 10: 7-11. Monte Carlo repetition code
    rng = np.random.default_rng(2026)


    def monte_carlo_logical(p, trials=200000):
        errors = rng.random((trials, 3)) < p
        failures = errors.sum(axis=1) >= 2
        return failures.mean()


    for p in [0.01, 0.05, 0.1]:
        print(p, monte_carlo_logical(p), p_logical(p))

    # Example 11: ۷-۱۲. بقای حافظه منطقی
    def survival_probability(p_cycle, rounds):
        return (1 - p_cycle) ** rounds


    for rounds in [10, 100, 1000, 10000]:
        print(rounds, survival_probability(1e-4, rounds))

    # Example 12: ۷-۱۳. آزمون فیزیکی‌بودن ZNE
    def clip_pauli_expectation(value):
        return float(np.clip(value, -1.0, 1.0))


    raw_zne = 1.07
    print(
        "raw estimator=",
        raw_zne,
        "clipped display=",
        clip_pauli_expectation(raw_zne),
    )  # Clipping changes bias; report both rather than hiding the issue.

    return locals()


if __name__ == "__main__":
    main()
