# -*- coding: utf-8 -*-
"""Chapter 05: اصول پایه مکانیک کوانتومی.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. نرمال‌سازی یک تابع موج گوسی
    import numpy as np

    x = np.linspace(-8, 8, 20001)
    sigma = 1.2
    psi = np.exp(-(x**2) / (4 * sigma**2)).astype(complex)
    norm = np.sqrt(np.trapezoid(np.abs(psi) ** 2, x))
    psi /= norm
    print(
        "integral probability =",
        np.trapezoid(np.abs(psi) ** 2, x),
    )

    # Example 02: ۷-۲. نمونه‌گیری از قاعده بورن
    import numpy as np

    rng = np.random.default_rng(2026)
    psi = np.array([np.sqrt(3) / 2, 0.5j])
    probabilities = np.abs(psi) ** 2
    samples = rng.choice([0, 1], size=10000, p=probabilities)
    print("frequency 0 =", np.mean(samples == 0))
    print("frequency 1 =", np.mean(samples == 1))

    # Example 03: ۷-۳. مقدار انتظار و واریانس
    import numpy as np

    psi = np.array([np.sqrt(3) / 2, 0.5j])
    Z = np.array([[1, 0], [0, -1]], complex)
    mean = np.vdot(psi, Z @ psi).real
    second = np.vdot(psi, Z @ Z @ psi).real
    variance = second - mean**2
    print(mean, variance, np.sqrt(variance))

    # Example 04: ۷-۴. تکامل با تجزیه ویژه
    import numpy as np

    H = np.array([[1.0, 0.35], [0.35, -1.0]], complex)
    psi0 = np.array([1.0, 0.0], complex)
    E, V = np.linalg.eigh(H)
    for t in [0.0, 0.5, 1.0, 1.5]:
        U = V @ np.diag(np.exp(-1j * E * t)) @ V.conj().T
        psi_t = U @ psi0
        print(t, np.abs(psi_t) ** 2, np.linalg.norm(psi_t))

    # Example 05: ۷-۵. پیش‌بینی با Qiskit
    import numpy as np
    from qiskit.quantum_info import Statevector, Operator

    psi = Statevector([np.sqrt(3) / 2, 0.5j])
    Z = Operator([[1, 0], [0, -1]])
    print("probabilities =", psi.probabilities_dict())
    print("<Z> =", psi.expectation_value(Z))
    print(
        "<Z^2> =",
        psi.expectation_value(Operator(Z.data @ Z.data)),
    )

    # Example 06: ۷-۶. اندازه‌گیری متوالی به‌صورت عددی
    import numpy as np

    zero = np.array([1, 0], complex)
    P0 = np.array([[1, 0], [0, 0]], complex)
    P1 = np.array([[0, 0], [0, 1]], complex)
    plus = np.array([1, 1], complex) / np.sqrt(2)
    minus = np.array([1, -1], complex) / np.sqrt(2)
    Qp = np.outer(plus, plus.conj())
    Qm = np.outer(minus, minus.conj())
    for name_a, P in [("z0", P0), ("z1", P1)]:
        for name_b, Q in [("x+", Qp), ("x-", Qm)]:
            probability = np.linalg.norm(Q @ P @ zero) ** 2
            print(name_a, "then", name_b, probability)

    # Example 07: ۷-۷. آزمون عددی یک کمیت پایسته
    import numpy as np

    X = np.array([[0, 1], [1, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    H = 0.7 * Z
    psi0 = np.array([1, 1], complex) / np.sqrt(2)
    E, V = np.linalg.eigh(H)
    for t in np.linspace(0, 4, 9):
        U = V @ np.diag(np.exp(-1j * E * t)) @ V.conj().T
        psi = U @ psi0
        ex = np.vdot(psi, X @ psi).real
        ez = np.vdot(psi, Z @ psi).real
        print(
            f"t={t:.1f}  <X>={ex:+.3f}  <Z>={ez:+.3f}  norm={np.linalg.norm(psi):.6f}"
        )

    return locals()


if __name__ == "__main__":
    main()
