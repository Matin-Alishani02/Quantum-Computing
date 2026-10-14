# -*- coding: utf-8 -*-
"""Chapter 13: الگوریتم‌های دویچ و دویچ–جوزا.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. مدار عمومی دویچ
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector


    def deutsch(oracle):
        qc = QuantumCircuit(2)
        qc.x(1)
        qc.h([0, 1])
        qc.compose(oracle, inplace=True)
        qc.h(0)
        return Statevector.from_instruction(
            qc
        ).probabilities_dict(qargs=[0])


    uf = QuantumCircuit(2)
    uf.cx(0, 1)
    print(deutsch(uf))

    # Example 02: ۷-۲. چهار اوراکل یک‌بیتی
    from qiskit import QuantumCircuit


    def oracle(kind):
        u = QuantumCircuit(2, name=kind)
        if kind == "one":
            u.x(1)
        elif kind == "x":
            u.cx(0, 1)
        elif kind == "not_x":
            u.x(1)
            u.cx(0, 1)
        return u


    for name in ["zero", "one", "x", "not_x"]:
        print(name, deutsch(oracle(name)))

    # Example 03: ۷-۳. مدار n‌بیتی با phase oracle
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector


    def dj_phase(n, oracle):
        qc = QuantumCircuit(n)
        qc.h(range(n))
        qc.compose(oracle, inplace=True)
        qc.h(range(n))
        return Statevector.from_instruction(
            qc
        ).probabilities_dict()


    phase = QuantumCircuit(4)
    phase.z(0)
    phase.z(2)
    # f(x)=x0 xor x2
    print(dj_phase(4, phase))

    # Example 04: ۷-۴. phase oracle از جدول درستی
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import DiagonalGate

    truth = [0, 1, 1, 0, 1, 0, 0, 1]
    # balanced
    signs = [1 if bit == 0 else -1 for bit in truth]
    oracle = QuantumCircuit(3)
    oracle.append(DiagonalGate(signs), range(3))
    print(dj_phase(3, oracle))

    # Example 05: ۷-۵. طبقه‌بندی جدول درستی
    def promise_class(truth):
        ones = sum(truth)
        if ones in (0, len(truth)):
            return "constant"
        if 2 * ones == len(truth):
            return "balanced"
        return "outside promise"


    for t in ([0] * 8, [0, 1] * 4, [0] * 6 + [1] * 2):
        print(promise_class(t))

    # Example 06: ۷-۶. شبیه‌سازی الگوریتم تصادفی کلاسیک
    import numpy as np


    def randomized_test(truth, k, seed=7):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(truth), size=k, replace=False)
        return (
            "balanced"
            if len({truth[i] for i in idx}) > 1
            else "constant"
        )


    truth = [0] * 32 + [1] * 32
    for k in [2, 4, 8, 12]:
        errors = sum(
            randomized_test(truth, k, seed=s) == "constant"
            for s in range(500)
        )
        print(k, errors / 500)

    # Example 07: ۷-۷. نمونه‌برداری شات از Statevector
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(3)
    qc.h(range(3))
    qc.z(1)
    qc.h(range(3))
    sv = Statevector.from_instruction(qc)
    sv.seed(2026)
    print(sv.sample_counts(shots=1000, qargs=[0, 1, 2]))

    # Example 08: ۷-۸. مدل ساده خطای خوانش
    import numpy as np


    def flip_readout(bitstrings, error, seed=3):
        rng = np.random.default_rng(seed)
        out = []
        for s in bitstrings:
            out.append(
                "".join(
                    str(int(b) ^ int(rng.random() < error))
                    for b in s
                )
            )
        return out


    ideal = ["0000"] * 2000
    noisy = flip_readout(ideal, 0.02)
    print(noisy.count("0000") / len(noisy))

    # Example 09: ۷-۹. قاعده آستانه و عدم‌قطعیت
    import math


    def classify_zero_count(zero_count, shots, threshold=0.5):
        p = zero_count / shots
        se = math.sqrt(max(p * (1 - p), 0) / shots)
        label = "constant" if p >= threshold else "balanced"
        return (
            label,
            p,
            (max(0, p - 1.96 * se), min(1, p + 1.96 * se)),
        )


    print(classify_zero_count(920, 1000, 0.5))
    print(classify_zero_count(35, 1000, 0.5))

    # Example 10: ۷-۱۰. انتخاب k کلاسیک از بودجه خطا
    import math


    def samples_for_one_sided_error(delta):
        return math.ceil(1 + math.log2(1 / delta))


    for delta in [0.1, 0.01, 0.001, 1e-6]:
        k = samples_for_one_sided_error(delta)
        print(delta, k, 2 ** (1 - k))

    return locals()


if __name__ == "__main__":
    main()
