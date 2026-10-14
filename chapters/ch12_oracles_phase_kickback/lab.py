# -*- coding: utf-8 -*-
"""Chapter 12: اوراکل‌ها، تداخل و بازگشت فاز.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. اوراکل یک‌بیتی fx=x
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    uf = QuantumCircuit(2, name="U_f")
    uf.cx(0, 1)
    print(Operator(uf).data)
    print(
        Operator(uf.compose(uf)).equiv(Operator.from_label("II"))
    )

    # Example 02: ۷-۲. مشاهده بازگشت فاز
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.x(1)
    qc.h(1)
    # target = |->
    qc.cx(0, 1)
    # f(x)=x
    sv = Statevector.from_instruction(qc)
    print(sv.to_dict())

    # Example 03: ۷-۳. تبدیل فاز به بیت
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.x(1)
    qc.h(1)
    qc.cx(0, 1)
    qc.h(0)
    probs = Statevector.from_instruction(qc).probabilities_dict(
        qargs=[0]
    )
    print(probs)

    # Example 04: ۷-۴. ساخت اوراکل فاز برای الگوی دلخواه
    from qiskit import QuantumCircuit


    def mark_101():
        oracle = QuantumCircuit(3, name="mark 101")
        oracle.x(1)
        oracle.h(2)
        oracle.mcx([0, 1], 2)
        oracle.h(2)
        oracle.x(1)
        return oracle


    print(mark_101().draw(output="text"))

    # Example 05: ۷-۵. استخراج طیف والش
    import numpy as np


    def walsh_spectrum(truth):
        g = 1 - 2 * np.asarray(truth, dtype=int)
        h = g.astype(float)
        step = 1
        while step < len(h):
            for i in range(0, len(h), 2 * step):
                a = h[i : i + step].copy()
                b = h[i + step : i + 2 * step].copy()
                h[i : i + step] = a + b
                h[i + step : i + 2 * step] = a - b
            step *= 2
        return h / len(h)


    print(walsh_spectrum([0, 1, 1, 0]))

    # Example 06: ۷-۶. آزمون جدول درستی
    from qiskit.quantum_info import Statevector


    def basis_output(circuit, bits):
        sv = Statevector.from_label(bits).evolve(circuit)
        return max(
            sv.probabilities_dict(),
            key=sv.probabilities_dict().get,
        )


    uf = QuantumCircuit(2)
    uf.cx(0, 1)
    for label in ["00", "01", "10", "11"]:
        print(label, "->", basis_output(uf, label))

    # Example 07: ۷-۷. بایاس و احتمال خروجی صفر
    import numpy as np

    truth = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    bias = np.mean(1 - 2 * truth)
    print("amplitude 000 =", bias)
    print("probability 000 =", bias**2)

    # Example 08: ۷-۸. الگوی compute–phase–uncompute
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(3)
    # q2 is workspace
    qc.h([0, 1])
    qc.ccx(0, 1, 2)
    # compute x0 AND x1
    qc.z(2)
    # phase when workspace=1
    qc.ccx(0, 1, 2)
    # uncompute
    sv = Statevector.from_instruction(qc)
    print(sv.probabilities_dict(qargs=[2]))

    # Example 09: ۷-۹. آزمون پاک‌بودن workspace
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    clean = QuantumCircuit(4)
    # q0,q1 data; q2 work; q3 target
    clean.h([0, 1])
    clean.ccx(0, 1, 2)
    clean.cx(2, 3)
    clean.ccx(0, 1, 2)
    sv = Statevector.from_instruction(clean)
    print("workspace:", sv.probabilities_dict(qargs=[2]))
    print("target   :", sv.probabilities_dict(qargs=[3]))

    return locals()


if __name__ == "__main__":
    main()
