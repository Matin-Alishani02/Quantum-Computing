# -*- coding: utf-8 -*-
"""Chapter 04: نمادگذاری دیراک و فضای هیلبرت.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. کت، برا و براکت با NumPy
    import numpy as np

    ket_psi = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    bra_psi = ket_psi.conj().T
    ket_phi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    inner = bra_psi @ ket_phi
    outer = np.outer(ket_psi, ket_phi.conj())
    print("<psi|phi> =", inner)
    print("|psi><phi| =\n", outer)

    # Example 02: ۷-۲. ساخت پروژکتور و آزمون آن
    import numpy as np

    psi = np.array([1, 1j], complex) / np.sqrt(2)
    P = np.outer(psi, psi.conj())
    print("Hermitian:", np.allclose(P, P.conj().T))
    print("Idempotent:", np.allclose(P @ P, P))
    print("eigenvalues:", np.linalg.eigvalsh(P))

    # Example 03: ۷-۳. درج همانی و بازسازی عملگر
    import numpy as np

    A = np.array([[2, 1 - 1j], [1 + 1j, -1]], complex)
    basis = np.eye(2, dtype=complex)
    A_rebuilt = np.zeros_like(A)
    for i in range(2):
        for j in range(2):
            element = np.vdot(basis[:, i], A @ basis[:, j])
            A_rebuilt += element * np.outer(
                basis[:, i], basis[:, j].conj()
            )
    print(np.allclose(A, A_rebuilt))

    # Example 04: ۷-۴. هم‌پوشانی و فاز کلی در Qiskit
    import numpy as np
    from qiskit.quantum_info import Statevector

    psi = Statevector([1 / np.sqrt(2), 1j / np.sqrt(2)])
    phi = Statevector(1j * psi.data)
    chi = Statevector([1 / np.sqrt(2), -1j / np.sqrt(2)])
    print("<psi|chi> =", psi.inner(chi))
    print("psi equiv phi:", psi.equiv(phi))
    print("probabilities:", psi.probabilities_dict())

    # Example 05: ۷-۵. مقدار انتظار با Operator
    import numpy as np
    from qiskit.quantum_info import Statevector, Operator

    psi = Statevector([np.sqrt(3) / 2, 0.5j])
    Z = Operator([[1, 0], [0, -1]])
    print("<Z> =", psi.expectation_value(Z))  # Direct check
    print(np.vdot(psi.data, Z.data @ psi.data))

    # Example 06: ۷-۶. تغییر پایه و ماتریس یک عملگر
    import numpy as np

    H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    A_z = np.array([[2, 0], [0, -1]], complex)
    psi_z = np.array([1, 0], complex)
    # Columns of H are the X-basis vectors written in the Z basis.
    psi_x = H.conj().T @ psi_z
    A_x = H.conj().T @ A_z @ H
    print("state in X coordinates =", psi_x)
    print("operator in X coordinates =\n", A_x)
    print(
        "expectations:",
        np.vdot(psi_z, A_z @ psi_z),
        np.vdot(psi_x, A_x @ psi_x),
    )

    # Example 07: ۷-۷. تقریب عددی بستار پیوسته
    import numpy as np

    x = np.linspace(-5, 5, 401)
    dx = x[1] - x[0]
    psi = np.exp(-(x**2)).astype(complex)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    # Discrete inner product: <f|g> ≈ sum(conj(f)*g)*dx
    norm = np.sum(np.conj(psi) * psi) * dx
    identity_action = (np.eye(len(x)) / dx) @ psi * dx
    print("norm =", norm.real)
    print(
        "closure error =", np.linalg.norm(identity_action - psi)
    )

    # Example 08: ۷-۸. پروژکتور یک زیرفضا و مکمل آن
    import numpy as np

    generators = np.array([[1, 1], [1j, 0], [0, 1]], complex)
    Q, _ = np.linalg.qr(generators)
    P = Q @ Q.conj().T
    I = np.eye(3, dtype=complex)
    psi = np.array([1, 2j, -1], complex)
    psi /= np.linalg.norm(psi)
    inside = P @ psi
    outside = (I - P) @ psi
    print(
        "reconstruction error =",
        np.linalg.norm(psi - inside - outside),
    )
    print("orthogonality =", np.vdot(inside, outside))
    print("P^2-P error =", np.linalg.norm(P @ P - P))

    return locals()


if __name__ == "__main__":
    main()
