# -*- coding: utf-8 -*-
"""Chapter 09: درهم‌تنیدگی و حالت‌های بل.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. ساخت Φ+ و مشاهده دامنه‌ها
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    phi_plus = Statevector.from_instruction(qc)
    print(qc.draw(output="text"))
    print(phi_plus.data)
    print(phi_plus.probabilities_dict())

    # Example 02: ۷-۲. رد جزئی، خلوص و آنتروپی
    from qiskit.quantum_info import partial_trace, purity, entropy

    rho0 = partial_trace(phi_plus, [1])
    rho1 = partial_trace(phi_plus, [0])
    print(rho0.data)
    print("purity q0 =", purity(rho0))
    print("entropy q0 =", entropy(rho0, base=2))
    print("entropy q1 =", entropy(rho1, base=2))

    # Example 03: ۷-۳. دترمینان و concurrence
    import numpy as np
    from qiskit.quantum_info import Statevector, concurrence


    def det_test(amplitudes):
        C = np.asarray(amplitudes, complex).reshape(2, 2)
        return np.linalg.det(C)


    product = Statevector([0.5, 0.5, 0.5, 0.5])
    bell = Statevector([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
    for state in [product, bell]:
        print(det_test(state.data), concurrence(state))

    # Example 04: ۷-۴. محاسبه هم‌بستگی‌های پائولی
    from qiskit.quantum_info import SparsePauliOp

    for label in ["ZZ", "XX", "YY"]:
        op = SparsePauliOp.from_list([(label, 1.0)])
        value = phi_plus.expectation_value(op).real
        print(label, value)

    # Example 05: ۷-۵. مقدار CHSH
    import numpy as np
    from qiskit.quantum_info import Operator

    X = np.array([[0, 1], [1, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    A0, A1 = Z, X
    B0, B1 = (Z + X) / np.sqrt(2), (Z - X) / np.sqrt(2)


    def corr(A, B):
        # Qiskit state order is q1 tensor q0: B acts on q1,
        # A on q0.
        return phi_plus.expectation_value(
            Operator(np.kron(B, A))
        ).real


    S = corr(A0, B0) + corr(A0, B1) + corr(A1, B0) - corr(A1, B1)
    print(S, 2 * np.sqrt(2))

    # Example 06: ۷-۶. حالت ورنر و کاهش درهم‌تنیدگی
    import numpy as np
    from qiskit.quantum_info import DensityMatrix, concurrence

    rho_bell = DensityMatrix(phi_plus).data
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rho = v * rho_bell + (1 - v) * np.eye(4) / 4
        print(v, concurrence(DensityMatrix(rho)))

    # Example 07: ۷-۷. مقایسه GHZ و W
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import (
        Statevector,
        partial_trace,
        purity,
    )

    ghz_circuit = QuantumCircuit(3)
    ghz_circuit.h(0)
    ghz_circuit.cx(0, 1)
    ghz_circuit.cx(0, 2)
    ghz = Statevector.from_instruction(ghz_circuit)
    w_data = np.array(
        [0, 1, 1, 0, 1, 0, 0, 0], complex
    ) / np.sqrt(3)
    w = Statevector(w_data)
    for name, state in [("GHZ", ghz), ("W", w)]:
        reduced_pair = partial_trace(state, [2])
        print(name, "pair purity =", purity(reduced_pair))

    # Example 08: ۷-۸. negativity با ترانهاده جزئی
    import numpy as np
    from qiskit.quantum_info import DensityMatrix

    rho = DensityMatrix(
        phi_plus
    ).data  # Axes: row q1, row q0, column q1, column q0.
    rho4 = rho.reshape(2, 2, 2, 2)
    partial_transpose_q1 = rho4.transpose(2, 1, 0, 3).reshape(
        4, 4
    )
    eigenvalues = np.linalg.eigvalsh(partial_transpose_q1)
    negativity = np.sum(np.abs(eigenvalues[eigenvalues < 0]))
    print(eigenvalues, negativity)

    # Example 09: ۷-۹. تعویض درهم‌تنیدگی با پس‌گزینی
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, concurrence

    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    # Bell q0-q1
    qc.h(2)
    qc.cx(2, 3)
    # Bell q2-q3
    qc.cx(1, 2)
    qc.h(1)
    # Bell transform on q1-q2
    state = Statevector.from_instruction(qc)
    tensor = state.data.reshape(2, 2, 2, 2)
    # q3,q2,q1,q0
    conditional = tensor[:, 0, 0, :].reshape(4)
    conditional /= np.linalg.norm(conditional)
    remote_pair = Statevector(conditional)
    # axes q3,q0
    print(remote_pair.probabilities_dict())
    print("concurrence =", concurrence(remote_pair))

    # Example 10: ۷-۱۰. ناوردایی زیر چرخش‌های محلی
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import (
        Statevector,
        partial_trace,
        entropy,
        concurrence,
    )

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rx(0.37, 0)
    qc.ry(-0.91, 1)
    qc.rz(0.22, 1)
    rotated = Statevector.from_instruction(qc)
    rho0 = partial_trace(rotated, [1])
    print("entropy =", entropy(rho0, base=2))
    print("concurrence =", concurrence(rotated))

    return locals()


if __name__ == "__main__":
    main()
