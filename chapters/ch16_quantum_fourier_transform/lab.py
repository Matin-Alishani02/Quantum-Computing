# -*- coding: utf-8 -*-
"""Chapter 16: تبدیل فوریه کوانتومی.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. ساخت دستی QFT
    import numpy as np
    from qiskit import QuantumCircuit


    def qft_manual(n, swaps=True):
        qc = QuantumCircuit(n, name="QFT")
        for target in reversed(range(n)):
            qc.h(target)
            for control in reversed(range(target)):
                angle = np.pi / (2 ** (target - control))
                qc.cp(angle, control, target)
        if swaps:
            for q in range(n // 2):
                qc.swap(q, n - q - 1)
        return qc


    print(qft_manual(4).draw(output="text"))

    # Example 02: ۷-۲. مقایسه با QFTGate
    from qiskit.circuit.library import QFTGate
    from qiskit.quantum_info import Operator

    manual = Operator(qft_manual(4))
    library = Operator(QFTGate(4))
    print(manual.equiv(library))

    # Example 03: ۷-۳. دامنه‌های QFT یک حالت پایه
    from qiskit.quantum_info import Statevector

    sv = Statevector.from_label("001").evolve(QFTGate(3))
    for state, amp in sv.to_dict().items():
        print(
            state,
            round(abs(amp), 4),
            round(float(np.angle(amp)), 4),
        )

    # Example 04: ۷-۴. آزمون inverse
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import QFTGate
    from qiskit.quantum_info import (
        random_statevector,
        state_fidelity,
    )

    n = 5
    psi = random_statevector(2**n, seed=17)
    qc = QuantumCircuit(n)
    qc.append(QFTGate(n), range(n))
    qc.append(QFTGate(n).inverse(), range(n))
    out = psi.evolve(qc)
    print(state_fidelity(psi, out))

    # Example 05: ۷-۵. ماتریس نظری
    import numpy as np


    def fourier_matrix(N):
        x = np.arange(N)
        return np.exp(2j * np.pi * np.outer(x, x) / N) / np.sqrt(
            N
        )


    F = fourier_matrix(8)
    print(np.linalg.norm(F.conj().T @ F - np.eye(8)))
    print(Operator(QFTGate(3)).equiv(Operator(F)))

    # Example 06: ۷-۶. حالت دوره‌ای
    import numpy as np
    from qiskit.quantum_info import Statevector
    from qiskit.circuit.library import QFTGate

    N = 16
    r = 4
    amps = np.zeros(N, complex)
    amps[::r] = 1 / 2
    sv = Statevector(amps).evolve(QFTGate(4))
    print(
        {
            k: round(v, 3)
            for k, v in sv.probabilities_dict().items()
            if v > 1e-9
        }
    )

    # Example 07: ۷-۷. بازیابی فاز دقیق با inverse QFT
    import numpy as np
    from qiskit.quantum_info import Statevector
    from qiskit.circuit.library import QFTGate

    n = 3
    N = 2**n
    phi = 3 / N
    amps = np.exp(2j * np.pi * np.arange(N) * phi) / np.sqrt(N)
    out = Statevector(amps).evolve(QFTGate(n).inverse())
    print(out.probabilities_dict())

    # Example 08: ۷-۸. QFT تقریبی دستی
    def qft_approx(n, max_distance):
        qc = QuantumCircuit(n, name="AQFT")
        for target in reversed(range(n)):
            qc.h(target)
            for control in reversed(range(target)):
                distance = target - control
                if distance <= max_distance:
                    qc.cp(np.pi / (2**distance), control, target)
        for q in range(n // 2):
            qc.swap(q, n - q - 1)
        return qc


    for m in [1, 2, 3, 7]:
        print(m, qft_approx(8, m).count_ops())

    # Example 09: ۷-۹. سنجش خطای تقریبی
    from qiskit.quantum_info import Operator
    import numpy as np

    exact = Operator(QFTGate(6)).data
    for m in [1, 2, 3, 4, 5]:
        approx = Operator(qft_approx(6, m)).data
        phase = np.vdot(exact.ravel(), approx.ravel())
        phase = phase / abs(phase)
        print(
            m,
            np.linalg.norm(
                exact - phase.conjugate() * approx, ord=2
            ),
        )

    # Example 10: ۷-۱۰. بررسی bit reversal بدون SWAP
    from qiskit.quantum_info import Operator

    n = 4
    with_swaps = Operator(qft_manual(n, swaps=True))
    without_swaps = Operator(qft_manual(n, swaps=False))
    reverse = QuantumCircuit(n)
    for q in range(n // 2):
        reverse.swap(q, n - q - 1)
    recovered = Operator(without_swaps.compose(reverse))
    print(recovered.equiv(with_swaps))

    return locals()


if __name__ == "__main__":
    main()
