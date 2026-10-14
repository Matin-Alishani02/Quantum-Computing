# -*- coding: utf-8 -*-
"""Chapter 02: اعداد مختلط، بردارها و فضاهای برداری.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۲. عدد مختلط و نمایش قطبی
    import numpy as np

    z = 3 - 4j
    print("z =", z)
    print("conjugate =", np.conj(z))
    print("magnitude =", np.abs(z))
    print("phase =", np.angle(z))
    print(
        "polar reconstruction =",
        np.abs(z) * np.exp(1j * np.angle(z)),
    )

    # Example 02: ۷-۳. نُرم، ضرب داخلی و تصویر
    import numpy as np

    u = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    v = np.array([1, -1j], dtype=complex) / np.sqrt(2)
    print("norm u =", np.linalg.norm(u))
    print("<u|v> =", np.vdot(u, v))
    # conjugates u
    w = np.array([1 + 1j, 2], dtype=complex)
    w = w / np.linalg.norm(w)
    projection = np.vdot(u, w) * u
    residual = w - projection
    print("orthogonality check =", np.vdot(u, residual))

    # Example 03: ۷-۴. یک گرام–اشمیت ساده
    import numpy as np


    def gram_schmidt(columns, tol=1e-12):
        basis = []
        for v in np.asarray(columns, dtype=complex).T:
            w = v.copy()
            for q in basis:
                w -= np.vdot(q, w) * q
            n = np.linalg.norm(w)
            if n <= tol:
                raise ValueError(
                    "Vectors are dependent or ill-conditioned"
                )
            basis.append(w / n)
        return np.column_stack(basis)


    A = np.array([[1, 1], [1j, 1]], dtype=complex)
    Q = gram_schmidt(A)
    print(Q.conj().T @ Q)

    # Example 04: ۷-۵. تبدیل بردار به حالت Qiskit
    import numpy as np
    from qiskit.quantum_info import Statevector

    alpha = np.sqrt(3) / 2
    beta = 0.5j
    psi = Statevector([alpha, beta])
    print("state =", psi.data)
    print("probabilities =", psi.probabilities_dict())
    print("norm =", np.vdot(psi.data, psi.data).real)

    # Example 05: ۷-۶. آزمایش فاز نسبی با تغییر پایه
    import numpy as np

    H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    plus = np.array([1, 1], complex) / np.sqrt(2)
    minus = np.array([1, -1], complex) / np.sqrt(2)
    for name, state in [("plus", plus), ("minus", minus)]:
        before = np.abs(state) ** 2
        after = np.abs(H @ state) ** 2
        print(name, "before=", before, "after H=", after)

    # Example 06: ۷-۷. پایه فوریه: نمونه‌ای از پایه مختلط
    import numpy as np

    N = 4
    j, k = np.meshgrid(np.arange(N), np.arange(N))
    F = np.exp(2j * np.pi * j * k / N) / np.sqrt(N)
    print(
        "unitarity error =",
        np.linalg.norm(F.conj().T @ F - np.eye(N)),
    )
    psi = np.array([1, 0, 0, 0], dtype=complex)
    coeff_fourier = F.conj().T @ psi
    print("Fourier coordinates =", coeff_fourier)
    print("probabilities =", np.abs(coeff_fourier) ** 2)

    return locals()


if __name__ == "__main__":
    main()
