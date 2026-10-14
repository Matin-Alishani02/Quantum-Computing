# -*- coding: utf-8 -*-
"""Chapter 15: الگوریتم جست‌وجوی گروور.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. phase oracle برای یک رشته
    from qiskit import QuantumCircuit


    def mark_bitstring(target):
        n = len(target)
        oracle = QuantumCircuit(n, name=f"mark {target}")
        for q, bit in enumerate(reversed(target)):
            if bit == "0":
                oracle.x(q)
        oracle.h(n - 1)
        oracle.mcx(list(range(n - 1)), n - 1)
        oracle.h(n - 1)
        for q, bit in enumerate(reversed(target)):
            if bit == "0":
                oracle.x(q)
        return oracle

    # Example 02: ۷-۲. ساخت diffuser
    from qiskit import QuantumCircuit


    def diffuser(n):
        d = QuantumCircuit(n, name="D")
        d.h(range(n))
        d.x(range(n))
        d.h(n - 1)
        d.mcx(list(range(n - 1)), n - 1)
        d.h(n - 1)
        d.x(range(n))
        d.h(range(n))
        return d

    # Example 03: ۷-۳. اجرای گروور کوچک
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    n = 3
    target = "101"
    iterations = 2
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.compose(mark_bitstring(target), inplace=True)
        qc.compose(diffuser(n), inplace=True)
    probs = Statevector.from_instruction(qc).probabilities_dict()
    print(
        sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[
            :4
        ]
    )

    # Example 04: ۷-۴. منحنی نظری تکرار
    import numpy as np


    def grover_curve(N, M, max_k):
        theta = np.arcsin(np.sqrt(M / N))
        k = np.arange(max_k + 1)
        return k, np.sin((2 * k + 1) * theta) ** 2


    k, p = grover_curve(64, 1, 14)
    print(list(zip(k, np.round(p, 4))))

    # Example 05: ۷-۵. انتخاب بهترین عدد صحیح
    import numpy as np


    def best_integer_k(N, M):
        theta = np.arcsin(np.sqrt(M / N))
        guess = np.pi / (4 * theta) - 0.5
        candidates = {
            max(0, int(np.floor(guess))),
            max(0, int(np.ceil(guess))),
        }
        return max(
            candidates,
            key=lambda k: np.sin((2 * k + 1) * theta) ** 2,
        )


    for M in [1, 2, 4, 8, 16, 32]:
        print(M, best_integer_k(64, M))

    # Example 06: ۷-۶. بررسی diffuser با Operator
    import numpy as np
    from qiskit.quantum_info import Operator

    n = 3
    N = 2**n
    s = np.ones(N) / np.sqrt(N)
    D = 2 * np.outer(s, s) - np.eye(N)
    print(Operator(diffuser(n)).equiv(Operator(D)))

    # Example 07: ۷-۷. دو جواب
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    n = 3
    oracle = QuantumCircuit(n)
    oracle.compose(mark_bitstring("001"), inplace=True)
    oracle.compose(mark_bitstring("110"), inplace=True)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser(n), inplace=True)
    print(Statevector.from_instruction(qc).probabilities_dict())

    # Example 08: ۷-۸. مقایسه مدار و فرمول
    from qiskit.quantum_info import Statevector
    import numpy as np


    def circuit_success(n, target, k):
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for _ in range(k):
            qc.compose(mark_bitstring(target), inplace=True)
            qc.compose(diffuser(n), inplace=True)
        return (
            Statevector.from_instruction(qc)
            .probabilities_dict()
            .get(target, 0.0)
        )


    for k in range(6):
        empirical = circuit_success(3, "101", k)
        theory = (
            np.sin((2 * k + 1) * np.arcsin(1 / np.sqrt(8))) ** 2
        )
        print(k, empirical, theory)

    # Example 09: ۷-۹. مدل ساده افت با عمق
    import numpy as np

    N = 64
    M = 1
    theta = np.arcsin(np.sqrt(M / N))
    for k in range(12):
        ideal = np.sin((2 * k + 1) * theta) ** 2
        noisy = ideal * (0.94**k) + (1 - 0.94**k) * (M / N)
        print(k, round(ideal, 3), round(noisy, 3))

    # Example 10: ۷-۱۰. تعداد تکرار مستقل برای اطمینان
    import math


    def runs_for_confidence(success_probability, failure_budget):
        if success_probability >= 1:
            return 1
        if success_probability <= 0:
            return math.inf
        return math.ceil(
            math.log(failure_budget)
            / math.log(1 - success_probability)
        )


    for p in [0.50, 0.80, 0.95]:
        print(p, runs_for_confidence(p, 1e-3))

    return locals()


if __name__ == "__main__":
    main()
