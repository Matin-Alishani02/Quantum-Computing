# -*- coding: utf-8 -*-
"""Chapter 17: تخمین فاز و الگوریتم شور.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. تابع کسر مسلسل
    def convergents(num, den):
        a = []
        while den:
            q, num, den = num // den, den, num % den
            a.append(q)
        p0, p1, q0, q1 = 0, 1, 1, 0
        for x in a:
            p0, p1 = p1, x * p1 + p0
            q0, q1 = q1, x * q1 + q0
            yield p1, q1

    # Example 02: ۷-۲. اعتبارسنجی نامزد مرتبه
    from math import gcd


    def valid_order(a, r, N):
        return r > 0 and gcd(a, N) == 1 and pow(a, r, N) == 1


    print(valid_order(2, 4, 15))
    # True

    # Example 03: ۷-۳. ساخت مدار QPE برای یک دروازهٔ فاز
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import QFTGate


    def qpe_phase(phi, m):
        qc = QuantumCircuit(m + 1, m)
        target = m
        qc.x(target)
        qc.h(range(m))
        for j in range(m):
            qc.cp(2 * np.pi * phi * (2**j), j, target)
        qc.append(QFTGate(m).inverse(), range(m))
        qc.measure(range(m), range(m))
        return qc

    # Example 04: ۷-۴. نمونه‌برداری با StatevectorSampler
    from qiskit.primitives import StatevectorSampler

    qc = qpe_phase(phi=3 / 8, m=3)
    sampler = StatevectorSampler(default_shots=1024, seed=2026)
    result = sampler.run([qc]).result()[0]
    counts = result.data.c.get_counts()
    print(counts)

    # Example 05: ۷-۵. تبدیل رشته به فاز با کنترل ترتیب بیت
    def phase_from_bitstring(bitstring):
        clean = bitstring.replace(" ", "")
        return int(clean, 2) / (2 ** len(clean))


    for bits in ["011", "0101"]:
        print(bits, phase_from_bitstring(bits))

    # Example 06: ۷-۶. بازیابی مرتبه از یک نمونه
    def recover_order(c, m, a, N):
        for _, r in convergents(c, 2**m):
            if 0 < r < N and valid_order(a, r, N):
                return r
        return None


    print(recover_order(c=16, m=6, a=2, N=15))

    # Example 07: ۷-۷. استخراج عامل‌ها
    def factors_from_order(a, r, N):
        if r is None or r % 2:
            return None
        x = pow(a, r // 2, N)
        if x in (1, N - 1):
            return None
        p, q = gcd(x - 1, N), gcd(x + 1, N)
        return (p, q) if 1 < p < N and 1 < q < N else None


    print(factors_from_order(2, 4, 15))

    # Example 08: ۷-۸. آزمون چند مقدار کوچک
    for N, a in [(15, 2), (21, 2), (33, 2)]:
        r = 1
        while pow(a, r, N) != 1:
            r += 1
        print(N, a, r, factors_from_order(a, r, N))

    # Example 09: ۷-۹. بررسی توزیع ایدئال بدون shots
    from qiskit.quantum_info import Statevector

    qc = qpe_phase(1 / 3, 4).remove_final_measurements(
        inplace=False
    )
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities_dict(qargs=range(4))
    print(
        sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[
            :4
        ]
    )

    # Example 10: ۷-۱۰. توزیع تحلیلی QPE
    def qpe_probabilities(phi, m):
        c = np.arange(2**m)
        x = np.arange(2**m)[:, None]
        amp = np.exp(2j * np.pi * x * (phi - c / (2**m))).mean(
            axis=0
        )
        return np.abs(amp) ** 2


    print(np.argsort(qpe_probabilities(1 / 3, 5))[-4:][::-1])

    # Example 11: ۷-۱۱. ترکیب نامزدهای چند run
    from math import lcm


    def combine_orders(candidates, a, N):
        r = 1
        for value in candidates:
            r = lcm(r, value)
            if valid_order(a, r, N):
                return r
        return None


    print(combine_orders([3, 4], a=2, N=13))

    # Example 12: ۷-۱۲. تشخیص توان کامل
    def perfect_power(N):
        for k in range(2, N.bit_length() + 1):
            b = round(N ** (1 / k))
            for x in range(max(2, b - 1), b + 2):
                if x**k == N:
                    return x, k
        return None


    print(perfect_power(81))

    return locals()


if __name__ == "__main__":
    main()
