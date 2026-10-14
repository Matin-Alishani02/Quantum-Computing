# -*- coding: utf-8 -*-
"""Chapter 01: رایانش کوانتومی چیست؟.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۲. آزمایش اول: ساخت هادامارد فقط با NumPy
    import numpy as np

    ket0 = np.array([1, 0], dtype=complex)
    H = (1 / np.sqrt(2)) * np.array(
        [[1, 1], [1, -1]], dtype=complex
    )
    psi = H @ ket0
    probabilities = np.abs(psi) ** 2
    print("state =", psi)
    print("P(0), P(1) =", probabilities)
    print("normalization =", probabilities.sum())

    # Example 02: ۷-۳. آزمایش دوم: یک کیوبیت با Qiskit
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()
    print(qc.draw(output="text"))
    sampler = StatevectorSampler(seed=2026)
    job = sampler.run([qc], shots=1024)
    result = job.result()[0]
    counts = result.data.meas.get_counts()
    print(counts)

    # Example 03: ۷-۴. آزمایش سوم: ساخت یک جفت بل
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    sampler = StatevectorSampler(seed=2026)
    result = sampler.run([qc], shots=2048).result()[0]
    counts = result.data.meas.get_counts()
    print(qc.draw(output="text"))
    print(counts)

    return locals()


if __name__ == "__main__":
    main()
