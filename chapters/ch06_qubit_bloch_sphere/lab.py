# -*- coding: utf-8 -*-
"""Chapter 06: کیوبیت، برهم‌نهی، اندازه‌گیری و کره بلوخ.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. تبدیل دامنه‌ها به بردار بلوخ
    import numpy as np


    def bloch_from_state(state):
        state = np.asarray(state, dtype=complex)
        state = state / np.linalg.norm(state)
        alpha, beta = state
        return np.array(
            [
                2 * np.real(np.conj(alpha) * beta),
                2 * np.imag(np.conj(alpha) * beta),
                np.abs(alpha) ** 2 - np.abs(beta) ** 2,
            ]
        )


    psi = [np.sqrt(3) / 2, 0.5j]
    r = bloch_from_state(psi)
    print(r, np.linalg.norm(r))

    # Example 02: ۷-۲. آماده‌سازی حالت با Qiskit
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    theta = np.pi / 3
    phi = np.pi / 2
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    psi = Statevector.from_instruction(qc)
    print(qc.draw(output="text"))
    print(psi.data)
    print(psi.probabilities_dict())

    # Example 03: ۷-۳. نمونه‌گیری در سه پایه
    from qiskit.primitives import StatevectorSampler


    def measured_copy(prep, basis):
        qc = prep.copy()
        if basis == "X":
            qc.h(0)
        elif basis == "Y":
            qc.sdg(0)
            qc.h(0)
        qc.measure_all()
        return qc


    sampler = StatevectorSampler(seed=2026)
    for basis in ["Z", "X", "Y"]:
        circuit = measured_copy(qc, basis)
        result = sampler.run([circuit], shots=4096).result()[0]
        print(basis, result.data.meas.get_counts())

    # Example 04: ۷-۴. حالت مخلوط و توپ بلوخ
    import numpy as np
    from qiskit.quantum_info import DensityMatrix

    rho = DensityMatrix([[0.8, 0.0], [0.0, 0.2]])
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    r = np.array([np.trace(rho.data @ P).real for P in [X, Y, Z]])
    print("Bloch vector =", r)
    print("purity =", np.trace(rho.data @ rho.data).real)

    # Example 05: ۷-۵. توموگرافی ساده از شمارش‌ها
    def mean_pm(counts):
        n0 = counts.get("0", 0)
        n1 = counts.get("1", 0)
        return (n0 - n1) / (n0 + n1)


    # Replace with the three counts dictionaries produced above.
    # rx = mean_pm(counts_x)
    # ry = mean_pm(counts_y)
    # rz = mean_pm(counts_z)
    # rho_hat = (I + rx*X + ry*Y + rz*Z) / 2

    # Example 06: ۷-۶. کانال کاهش فاز در مختصات بلوخ
    import numpy as np


    def dephase(rho, eta):
        Z = np.array([[1, 0], [0, -1]], complex)
        p = (1 - eta) / 2
        return (1 - p) * rho + p * (Z @ rho @ Z)


    plus = np.array([1, 1], complex) / np.sqrt(2)
    rho = np.outer(plus, plus.conj())
    for eta in [1.0, 0.7, 0.3, 0.0]:
        out = dephase(rho, eta)
        purity = np.trace(out @ out).real
        print(eta, out, "purity=", purity)

    return locals()


if __name__ == "__main__":
    main()
