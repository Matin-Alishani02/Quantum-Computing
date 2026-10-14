# -*- coding: utf-8 -*-
"""Chapter 10: طراحی و تحلیل مدارهای کوانتومی.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. ساخت بلوک ماژولار و وارون آن
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    block = QuantumCircuit(2, name="BellPrep")
    block.h(0)
    block.cx(0, 1)
    gate = block.to_gate()
    qc = QuantumCircuit(2)
    qc.append(gate, [0, 1])
    qc.append(gate.inverse(), [0, 1])
    print(Operator(qc).equiv(Operator(np.eye(4))))

    # Example 02: ۷-۲. compose با نگاشت سیم
    from qiskit import QuantumCircuit

    sub = QuantumCircuit(2, name="sub")
    sub.h(0)
    sub.cx(0, 1)
    main = QuantumCircuit(4)
    main.x(0)
    main.compose(sub, qubits=[2, 0], inplace=True)
    print(main.draw(output="text"))

    # Example 03: ۷-۳. سنجش ساختار مدار
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(4)
    qc.h(range(4))
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.rz(0.4, 1)
    qc.cx(1, 2)
    print("width =", qc.num_qubits)
    print("size =", qc.size())
    print("depth =", qc.depth())
    print("ops =", qc.count_ops())

    # Example 04: ۷-۴. هم‌ارزی دقیق و تا فاز
    import numpy as np
    from qiskit.circuit.library import XGate, RXGate
    from qiskit.quantum_info import Operator

    X = Operator(XGate())
    Rx_pi = Operator(RXGate(np.pi))
    print("up to phase:", X.equiv(Rx_pi))
    print("matrix norm:", np.linalg.norm(X.data - Rx_pi.data))

    # Example 05: ۷-۵. ترنسپایل روی اتصال خطی
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 3)
    qc.rz(0.3, 3)
    qc.cx(3, 1)
    coupling = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]]
    tqc = transpile(
        qc,
        basis_gates=["u", "cx"],
        coupling_map=coupling,
        optimization_level=3,
        seed_transpiler=17,
    )
    print("before:", qc.depth(), qc.count_ops())
    print("after :", tqc.depth(), tqc.count_ops())
    print(tqc.draw(output="text"))

    # Example 06: ۷-۶. راستی‌آزمایی یک هویت
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    left = QuantumCircuit(2)
    left.h(0)
    left.h(1)
    left.cx(0, 1)
    left.h(0)
    left.h(1)
    right = QuantumCircuit(2)
    right.cx(1, 0)
    print(Operator(left).equiv(Operator(right)))

    # Example 07: ۷-۷. شات ایده‌آل و خطای نمونه‌گیری
    import numpy as np

    rng = np.random.default_rng(2026)
    p0 = 0.37
    for shots in [100, 1000, 10000]:
        estimates = [
            np.mean(rng.random(shots) < p0) for _ in range(300)
        ]
        print(
            shots,
            np.mean(estimates),
            np.std(estimates),
            "theory",
            np.sqrt(p0 * (1 - p0) / shots),
        )

    # Example 08: ۷-۸. آزمون تصادفی دو پیاده‌سازی
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import (
        random_statevector,
        state_fidelity,
    )

    u = QuantumCircuit(2)
    u.h(0)
    u.h(1)
    u.cx(0, 1)
    u.h(0)
    u.h(1)
    v = QuantumCircuit(2)
    v.cx(1, 0)
    for seed in range(8):
        psi = random_statevector(4, seed=seed)
        out_u = psi.evolve(u)
        out_v = psi.evolve(v)
        print(seed, state_fidelity(out_u, out_v))

    # Example 09: ۷-۹. قاعده parameter-shift
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, SparsePauliOp

    Z = SparsePauliOp.from_list([("Z", 1.0)])


    def f(theta):
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        return (
            Statevector.from_instruction(qc)
            .expectation_value(Z)
            .real
        )


    theta = 0.73
    gradient = 0.5 * (f(theta + np.pi / 2) - f(theta - np.pi / 2))
    print(gradient, -np.sin(theta))

    # Example 10: ۷-۱۰. فاصله کل تغییرات
    import numpy as np


    def tvd(p, q):
        p = np.asarray(p, float)
        q = np.asarray(q, float)
        return 0.5 * np.sum(np.abs(p - q))


    target = np.array([0.50, 0.00, 0.00, 0.50])
    observed = np.array([0.46, 0.03, 0.02, 0.49])
    print("TVD =", tvd(target, observed))
    print("normalized:", target.sum(), observed.sum())

    return locals()


if __name__ == "__main__":
    main()
