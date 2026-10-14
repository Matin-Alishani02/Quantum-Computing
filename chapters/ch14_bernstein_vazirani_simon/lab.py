# -*- coding: utf-8 -*-
"""Chapter 14: الگوریتم برنشتاین–وزیرانی و سایمون.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. ساخت oracle برنشتاین–وزیرانی
    from qiskit import QuantumCircuit


    def bv_oracle(secret):
        n = len(secret)
        oracle = QuantumCircuit(n + 1, name="BV oracle")
        for j, bit in enumerate(reversed(secret)):
            if bit == "1":
                oracle.cx(j, n)
        return oracle


    print(bv_oracle("1011").draw(output="text"))

    # Example 02: ۷-۲. اجرای BV با Statevector
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    secret = "1011"
    n = len(secret)
    qc = QuantumCircuit(n + 1)
    qc.x(n)
    qc.h(range(n + 1))
    qc.compose(bv_oracle(secret), inplace=True)
    qc.h(range(n))
    probs = Statevector.from_instruction(qc).probabilities_dict(
        qargs=range(n)
    )
    print(max(probs, key=probs.get), max(probs.values()))

    # Example 03: ۷-۳. بررسی offset
    from qiskit.quantum_info import Operator

    base = bv_oracle("110")
    with_offset = bv_oracle("110")
    with_offset.x(3)
    # target flip adds b=1
    print(base.draw("text"))
    print(with_offset.draw("text"))

    # Example 04: ۷-۴. oracle خطی سایمون برای s=101
    from qiskit import QuantumCircuit


    def simon_oracle_101():
        u = QuantumCircuit(5, name="Simon f")
        # q0..q2 input; q3,q4 output
        u.cx(0, 3)
        u.cx(2, 3)
        # out0=x0 xor x2
        u.cx(1, 4)
        # out1=x1
        return u


    print(simon_oracle_101().draw(output="text"))

    # Example 05: ۷-۵. توزیع نمونه‌های سایمون
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(5)
    qc.h([0, 1, 2])
    qc.compose(simon_oracle_101(), inplace=True)
    qc.h([0, 1, 2])
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities_dict(qargs=[0, 1, 2])
    print({k: round(v, 3) for k, v in probs.items() if v > 1e-9})

    # Example 06: ۷-۶. حذف گاوسی GF(۲)
    import numpy as np


    def rref_gf2(a):
        a = np.asarray(a, dtype=np.uint8).copy() % 2
        pivots = []
        row = 0
        for col in range(a.shape[1]):
            hits = np.flatnonzero(a[row:, col])
            if len(hits) == 0:
                continue
            p = row + hits[0]
            a[[row, p]] = a[[p, row]]
            for r in range(a.shape[0]):
                if r != row and a[r, col]:
                    a[r] ^= a[row]
            pivots.append(col)
            row += 1
            if row == a.shape[0]:
                break
        return a, pivots


    print(rref_gf2([[0, 1, 0], [1, 0, 1]]))

    # Example 07: ۷-۷. یافتن نامزد غیرصفر
    import itertools, numpy as np


    def nonzero_null_vector(rows, n):
        a = np.asarray(rows, dtype=np.uint8).reshape(-1, n) % 2
        for bits in itertools.product([0, 1], repeat=n):
            v = np.array(bits, dtype=np.uint8)
            if v.any() and np.all((a @ v) % 2 == 0):
                return "".join(map(str, bits))
        return None


    print(nonzero_null_vector([[0, 1, 0], [1, 0, 1]], 3))

    # Example 08: ۷-۸. گردآوری تا رتبه n−۱
    import numpy as np


    def gf2_rank(rows, n):
        if not rows:
            return 0
        _, pivots = rref_gf2(
            np.array(rows, dtype=np.uint8).reshape(-1, n)
        )
        return len(pivots)


    rows = []
    for sample in ["000", "010", "111", "101", "010"]:
        y = [int(b) for b in sample]
        if gf2_rank(rows + [y], 3) > gf2_rank(rows, 3):
            rows.append(y)
        if gf2_rank(rows, 3) == 2:
            break
    print(rows, nonzero_null_vector(rows, 3))

    # Example 09: ۷-۹. شبیه‌سازی تعداد نمونه لازم
    import numpy as np


    def trials_to_full_rank(n, repeats=1000, seed=5):
        rng = np.random.default_rng(seed)
        totals = []
        # use secret with last bit 1; sample y satisfying
        # y·s=0 by rejection
        s = np.ones(n, dtype=np.uint8)
        for _ in range(repeats):
            rows = []
            t = 0
            while gf2_rank(rows, n) < n - 1:
                y = rng.integers(0, 2, n, dtype=np.uint8)
                t += 1
                if int(y @ s) % 2 == 0:
                    rows.append(y)
            totals.append(t)
        return np.mean(totals)


    print(trials_to_full_rank(8))

    # Example 10: ۷-۱۰. آزمون وعده یک نگاشت خطی
    import itertools, numpy as np

    A = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    secret = np.array([1, 0, 1], dtype=np.uint8)
    outputs = {}
    for bits in itertools.product([0, 1], repeat=3):
        x = np.array(bits, dtype=np.uint8)
        y = tuple((A @ x % 2).tolist())
        outputs.setdefault(y, []).append(tuple(bits))
    print(outputs)
    print(
        all(
            len(v) == 2 and tuple(np.bitwise_xor(*v)) == (1, 0, 1)
            for v in outputs.values()
        )
    )

    return locals()


if __name__ == "__main__":
    main()
