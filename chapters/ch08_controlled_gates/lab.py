# -*- coding: utf-8 -*-
"""Chapter 08: سامانه‌های چندکیوبیتی و دروازه‌های کنترل‌شده.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. ساخت ماتریس controlled-U با NumPy
    import numpy as np


    def controlled(U):
        I = np.eye(2, dtype=complex)
        zero = np.array([[1, 0], [0, 0]], complex)
        one = np.array([[0, 0], [0, 1]], complex)
        return np.kron(zero, I) + np.kron(one, U)


    X = np.array([[0, 1], [1, 0]], complex)
    CX = controlled(X)
    print(CX)
    print(np.linalg.norm(CX.conj().T @ CX - np.eye(4)))

    # Example 02: ۷-۲. جدول نگاشت CNOT در Qiskit
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    for c in [0, 1]:
        for t in [0, 1]:
            qc = QuantumCircuit(2)
            if c:
                qc.x(0)
            if t:
                qc.x(1)
            qc.cx(0, 1)
            probs = Statevector.from_instruction(
                qc
            ).probabilities_dict()
            printed = max(probs, key=probs.get)
            # q1 q0
            logical = printed[::-1]
            # q0 q1
            print(f"{c}{t} -> {logical}  (Qiskit key {printed})")

    # Example 03: ۷-۳. هم‌ارزی SWAP با سه CX
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import SwapGate
    from qiskit.quantum_info import Operator

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.cx(0, 1)
    print(Operator(qc).equiv(Operator(SwapGate())))
    print(qc.count_ops(), "depth =", qc.depth())

    # Example 04: ۷-۴. CNOT و شکست همانندسازی حالت |+⟩
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    state = Statevector.from_instruction(qc)
    print(state.probabilities_dict())
    print(state.data)

    # Example 05: ۷-۵. جدول کامل تافولی
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                qc = QuantumCircuit(3)
                for bit, q in zip([a, b, c], [0, 1, 2]):
                    if bit:
                        qc.x(q)
                qc.ccx(0, 1, 2)
                probs = Statevector.from_instruction(
                    qc
                ).probabilities_dict()
                out = max(probs, key=probs.get)[::-1]
                print(f"{a}{b}{c} -> {out}")

    # Example 06: ۷-۶. کنترل روی صفر
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    zero_control = QuantumCircuit(2)
    zero_control.x(0)
    zero_control.cx(0, 1)
    zero_control.x(0)
    print(zero_control.draw(output="text"))
    print(Operator(zero_control).data)

    # Example 07: ۷-۷. احتمال‌های یک زیرسامانه
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(0.8, 2)
    state = Statevector.from_instruction(qc)
    print("all qubits:", state.probabilities_dict())
    print("q0 only:", state.probabilities([0]))
    print("q2 only:", state.probabilities([2]))

    # Example 08: ۷-۸. راستی‌آزمایی قواعد انتشار پائولی
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    cx = QuantumCircuit(2)
    cx.cx(0, 1)
    x_control = QuantumCircuit(2)
    x_control.x(0)
    x_both = QuantumCircuit(2)
    x_both.x(0)
    x_both.x(1)
    left = Operator(x_control.compose(cx))
    right = Operator(cx.compose(x_both))
    print(left.equiv(right))

    # Example 09: ۷-۹. ساخت phase gadget دوکیوبیتی
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    theta = 0.63
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.rz(theta, 1)
    qc.cx(0, 1)
    Z = np.diag([1, -1])
    ZZ = np.kron(Z, Z)
    target = (
        np.cos(theta / 2) * np.eye(4)
        - 1j * np.sin(theta / 2) * ZZ
    )
    print(Operator(qc).equiv(Operator(target)))

    return locals()


if __name__ == "__main__":
    main()
