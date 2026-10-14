# -*- coding: utf-8 -*-
"""Chapter 03: ماتریس‌ها، مقادیر ویژه و ضرب تانسوری.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. ضرب و ترتیب ماتریس‌ها
    import numpy as np

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    zero = np.array([1, 0], dtype=complex)
    print("H @ X @ |0> =", H @ X @ zero)
    print("X @ H @ |0> =", X @ H @ zero)
    print("commutator =\n", H @ X - X @ H)

    # Example 02: ۷-۲. مقادیر ویژه هرمیتی با eigh
    import numpy as np

    A = np.array([[2, 1 - 1j], [1 + 1j, 3]], dtype=complex)
    assert np.allclose(A, A.conj().T)
    values, vectors = np.linalg.eigh(A)
    print("eigenvalues =", values)
    print("V^dagger V =\n", vectors.conj().T @ vectors)
    print(
        "reconstruction =\n",
        vectors @ np.diag(values) @ vectors.conj().T,
    )

    # Example 03: ۷-۳. ساخت نمایی یک هامیلتونی دوبعدی
    import numpy as np

    H = np.array([[1, 0.4], [0.4, -1]], dtype=complex)
    t = 0.7
    # use units with hbar = 1
    E, V = np.linalg.eigh(H)
    U = V @ np.diag(np.exp(-1j * E * t)) @ V.conj().T
    print(
        "unitarity error =",
        np.linalg.norm(U.conj().T @ U - np.eye(2)),
    )

    # Example 04: ۷-۴. ضرب تانسوری و ترتیب پایه
    import numpy as np

    zero = np.array([1, 0], dtype=complex)
    one = np.array([0, 1], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    ket01 = np.kron(zero, one)
    print("|01> =", ket01)
    print("(X tensor I)|01> =", np.kron(X, I) @ ket01)

    # Example 05: ۷-۵. Operator و Statevector در Qiskit
    import numpy as np
    from qiskit.quantum_info import Operator, Statevector

    H = Operator(
        np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    )
    psi = Statevector.from_label("0")
    out = psi.evolve(H)
    print(out.data)
    print(out.probabilities_dict())
    print(
        "H is unitary:",
        np.allclose(H.data.conj().T @ H.data, np.eye(2)),
    )

    # Example 06: ۷-۶. مقدارهای منفرد و حساسیت
    import numpy as np

    A = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-6]])
    U, s, Vh = np.linalg.svd(A)
    print("singular values =", s)
    print("condition number =", s[0] / s[-1])
    b = np.array([2.0, 2.0 + 1e-6])
    x = np.linalg.solve(A, b)
    print("solution =", x)

    # Example 07: ۷-۷. بسط یک عملگر در پایه پائولی
    import numpy as np

    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    paulis = {"I": I, "X": X, "Y": Y, "Z": Z}
    A = np.array([[2, 1 - 0.5j], [1 + 0.5j, -1]], complex)
    coeff = {
        k: np.trace(P.conj().T @ A) / 2 for k, P in paulis.items()
    }
    rebuilt = sum(coeff[k] * paulis[k] for k in paulis)
    print(coeff)
    print("error =", np.linalg.norm(A - rebuilt))

    # Example 08: ۷-۸. ساخت عملگر کنترل‌شده از بلوک‌ها
    import numpy as np

    I = np.eye(2, dtype=complex)
    P0 = np.array([[1, 0], [0, 0]], complex)
    P1 = np.array([[0, 0], [0, 1]], complex)
    U = np.array([[0, 1], [1, 0]], complex)
    # X
    controlled_sum = np.kron(P0, I) + np.kron(P1, U)
    controlled_block = np.block(
        [[I, np.zeros((2, 2))], [np.zeros((2, 2)), U]]
    )
    print(np.allclose(controlled_sum, controlled_block))
    print(
        "unitarity error =",
        np.linalg.norm(
            controlled_sum.conj().T @ controlled_sum - np.eye(4)
        ),
    )

    return locals()


if __name__ == "__main__":
    main()
