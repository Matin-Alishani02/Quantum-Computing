# -*- coding: utf-8 -*-
"""Chapter 19: نویز، واهمدوسی و مدل‌های کانال کوانتومی.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. ماتریس چگالی و خلوص
    import numpy as np


    def purity(rho):
        return float(np.real(np.trace(rho @ rho)))


    rho = 0.5 * np.eye(2, dtype=complex)
    print(np.trace(rho), np.linalg.eigvalsh(rho), purity(rho))

    # Example 02: ۷-۲. ردگیری جزئی حالت بل
    from qiskit.quantum_info import Statevector, partial_trace

    bell = Statevector([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
    rho_a = partial_trace(bell, [1])
    print(rho_a.data, purity(rho_a.data))

    # Example 03: ۷-۳. اعمال مجموعه Kraus با NumPy
    def apply_kraus(rho, ops):
        return sum(K @ rho @ K.conj().T for K in ops)


    gamma = 0.3
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], complex)
    rho1 = np.array([[0, 0], [0, 1]], complex)
    print(apply_kraus(rho1, [K0, K1]))

    # Example 04: ۷-۴. آزمون completeness
    def completeness(ops):
        return sum(K.conj().T @ K for K in ops)


    print(np.allclose(completeness([K0, K1]), np.eye(2)))

    # Example 05: ۷-۵. نمایش Kraus و Choi در Qiskit
    from qiskit.quantum_info import Kraus, Choi, DensityMatrix

    channel = Kraus([K0, K1])
    choi = Choi(channel)
    print(np.linalg.eigvalsh(choi.data))
    rho_out = DensityMatrix(rho1).evolve(channel)
    print(rho_out.data)

    # Example 06: ۷-۶. کانال dephasing
    Z = np.array([[1, 0], [0, -1]], complex)
    I = np.eye(2, dtype=complex)
    p = 0.2
    phase_flip = [np.sqrt(1 - p) * I, np.sqrt(p) * Z]
    plus = 0.5 * np.array([[1, 1], [1, 1]], complex)
    print(apply_kraus(plus, phase_flip))

    # Example 07: ۷-۷. کانال depolarizing با قرارداد روشن
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    p = 0.25
    # E(rho)=(1-p)rho+p I/2
    ops = [
        np.sqrt(1 - 3 * p / 4) * I,
        np.sqrt(p / 4) * X,
        np.sqrt(p / 4) * Y,
        np.sqrt(p / 4) * Z,
    ]
    print(apply_kraus(plus, ops))

    # Example 08: ۷-۸. شبیه‌سازی منحنی T1
    rng = np.random.default_rng(2026)
    T1 = 45.0
    t = np.linspace(0, 160, 17)
    shots = 2000
    true_p = np.exp(-t / T1)
    counts1 = rng.binomial(shots, true_p)
    observed = counts1 / shots
    print(np.c_[t[:5], observed[:5]])

    # Example 09: ۷-۹. ترکیب کانال‌ها
    from qiskit.quantum_info import SuperOp

    E1 = SuperOp(Kraus([K0, K1]))
    E2 = SuperOp(Kraus(phase_flip))
    combined = E1.compose(E2)
    print(combined.data.shape)

    # Example 10: ۷-۱۰. نگاشت بردار بلوخ
    paulis = [X, Y, Z]


    def bloch(rho):
        return np.array(
            [np.real(np.trace(rho @ P)) for P in paulis]
        )


    for name, rho in [
        ("in", plus),
        ("out", apply_kraus(plus, [K0, K1])),
    ]:
        print(name, bloch(rho))

    # Example 11: ۷-۱۱. برازش log-linear ساده T1
    mask = observed > 0
    slope, intercept = np.polyfit(
        t[mask], np.log(observed[mask]), 1
    )
    T1_hat = -1 / slope
    print(T1_hat)
    # For real data, include SPAM offset and uncertainties.

    # Example 12: ۷-۱۲. خطای coherent در برابر stochastic
    delta = 0.01
    k = np.arange(1, 201)
    p_coherent = np.sin(k * delta / 2) ** 2
    p_stochastic = 0.5 * (1 - (1 - 2 * delta**2 / 4) ** k)
    print(p_coherent[[9, 49, 99]], p_stochastic[[9, 49, 99]])

    # Example 13: 7-13. Choi rank
    def numerical_rank(matrix, tol=1e-10):
        return int(np.sum(np.linalg.eigvalsh(matrix) > tol))


    print(
        "amplitude-damping Choi rank=", numerical_rank(choi.data)
    )
    print(
        "dephasing Choi rank=",
        numerical_rank(Choi(Kraus(phase_flip)).data),
    )

    return locals()


if __name__ == "__main__":
    main()
