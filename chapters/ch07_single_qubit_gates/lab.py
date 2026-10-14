# -*- coding: utf-8 -*-
"""Chapter 07: دروازه‌های تک‌کیوبیتی.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. بررسی یکانی‌بودن با NumPy
    import numpy as np

    X = np.array([[0, 1], [1, 0]], complex)
    H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    for name, gate in [("X", X), ("H", H)]:
        error = np.linalg.norm(gate.conj().T @ gate - np.eye(2))
        print(name, "unitarity error =", error)

    # Example 02: ۷-۲. ساخت و خواندن حالت نهایی در Qiskit
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.s(0)
    qc.ry(0.4, 0)
    state = Statevector.from_instruction(qc)
    print(qc.draw(output="text"))
    print(state.data)
    print(state.probabilities_dict())

    # Example 03: ۷-۳. جاروب فاز در تداخل H-P-H
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    for phi in np.linspace(0, 2 * np.pi, 9):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(phi, 0)
        qc.h(0)
        p0 = Statevector.from_instruction(qc).probabilities()[0]
        theory = np.cos(phi / 2) ** 2
        print(
            f"{phi:.3f}  simulated={p0:.6f}  theory={theory:.6f}"
        )

    # Example 04: ۷-۴. آزمون هم‌ارزی تا فاز کلی
    import numpy as np
    from qiskit.circuit.library import PhaseGate, RZGate
    from qiskit.quantum_info import Operator

    lam = 0.73
    P = Operator(PhaseGate(lam))
    Rz = Operator(RZGate(lam))
    print("equivalent up to global phase:", P.equiv(Rz))
    print(
        "direct matrix distance:",
        np.linalg.norm(P.data - Rz.data),
    )

    # Example 05: ۷-۵. ساخت دروازه عمومی با زاویه‌های اویلر
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    phi, theta, lam = 0.2, 1.1, -0.7
    qc = QuantumCircuit(1)
    qc.rz(lam, 0)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    print(Operator(qc).data)
    print("depth =", qc.depth(), "ops =", qc.count_ops())

    # Example 06: ۷-۶. معکوس‌سازی خودکار مدار
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    block = QuantumCircuit(1, name="U")
    block.rx(0.31, 0)
    block.rz(-0.82, 0)
    block.h(0)
    identity_test = block.compose(block.inverse())
    print(
        Operator(identity_test).equiv(Operator([[1, 0], [0, 1]]))
    )
    print(identity_test.draw(output="text"))

    # Example 07: ۷-۷. مدار پارامتری و مقداردهی
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Statevector

    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    for value in [0, 0.5, 1.0, 1.5]:
        bound = qc.assign_parameters({theta: value})
        print(
            value,
            Statevector.from_instruction(bound).probabilities(),
        )

    # Example 08: ۷-۸. جدول مزدوج‌سازی H و S
    import numpy as np
    from qiskit.circuit.library import (
        HGate,
        SGate,
        XGate,
        YGate,
        ZGate,
    )
    from qiskit.quantum_info import Operator

    paulis = {
        "X": Operator(XGate()),
        "Y": Operator(YGate()),
        "Z": Operator(ZGate()),
    }
    for gate_name, gate in [
        ("H", Operator(HGate())),
        ("S", Operator(SGate())),
    ]:
        for p_name, p in paulis.items():
            transformed = Operator(
                gate.data @ p.data @ gate.adjoint().data
            )
            matches = [
                name
                for name, target in paulis.items()
                if transformed.equiv(target)
            ]
            print(gate_name, p_name, matches)

    # Example 09: ۷-۹. وفاداری خطای بیش‌چرخش
    import numpy as np


    def rz(angle):
        return np.diag(
            [np.exp(-0.5j * angle), np.exp(0.5j * angle)]
        )


    def average_unitary_fidelity(U, V):
        d = U.shape[0]
        return (abs(np.trace(U.conj().T @ V)) ** 2 + d) / (
            d * (d + 1)
        )


    theta = 0.8
    for eps in [0.0, 0.01, 0.05, 0.1]:
        print(
            eps,
            average_unitary_fidelity(rz(theta), rz(theta + eps)),
        )

    return locals()


if __name__ == "__main__":
    main()
