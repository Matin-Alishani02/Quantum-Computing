# -*- coding: utf-8 -*-
"""Chapter 26: محدودیت‌ها، برتری کوانتومی و آینده فناوری.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. قانون آمدال
    def amdahl(f, sq):
        return 1 / ((1 - f) + f / sq)


    for f in [0.1, 0.5, 0.9, 0.99]:
        print(f, amdahl(f, 1000))

    # Example 02: ۷-۲. crossover دو مدل
    import numpy as np


    def classical(n):
        return 3e-9 * 1.35**n


    def quantum(n):
        return 2e-4 * n**3 + 0.03 * n


    for n in range(5, 200):
        if quantum(n) < classical(n):
            print(
                "first modeled crossover",
                n,
                classical(n),
                quantum(n),
            )
            break

    # Example 03: ۷-۳. حساسیت به پیشرفت کلاسیک
    for improvement in [1, 10, 1000, 1e6]:
        candidates = [
            n
            for n in range(5, 250)
            if quantum(n) < classical(n) / improvement
        ]
        print(improvement, candidates[0] if candidates else None)

    # Example 04: ۷-۴. تکرار تا موفقیت
    def repeats(p, target=0.99):
        return (
            np.inf
            if p <= 0
            else (
                1
                if p >= 1
                else int(
                    np.ceil(np.log(1 - target) / np.log(1 - p))
                )
            )
        )


    for p in [0.001, 0.01, 0.1, 0.5]:
        print(p, repeats(p))

    # Example 05: ۷-۵. فاصله کد آموزشی
    def logical_error(p, pth, d, A=0.1):
        return A * (p / pth) ** ((d + 1) / 2)


    for d in [3, 5, 7, 9, 11, 15]:
        print(d, logical_error(1e-3, 1e-2, d), 2 * d * d)

    # Example 06: ۷-۶. انتخاب d از بودجه شکست
    def choose_distance(p, pth, locations, eps):
        for d in range(3, 100, 2):
            if locations * logical_error(p, pth, d) < eps:
                return d


    print(choose_distance(1e-3, 1e-2, 10**10, 1e-2))

    # Example 07: ۷-۷. throughput کارخانه
    def runtime_from_tdepth(
        tdepth,
        cycle_us,
        factories,
        tcount=None,
        factory_period_us=None,
    ):
        if tcount is None:
            tcount = 10 * tdepth
        if factory_period_us is None:
            factory_period_us = cycle_us
        critical_path = tdepth * cycle_us * 1e-6
        state_supply = (
            tcount * factory_period_us * 1e-6 / factories
        )
        return max(critical_path, state_supply)


    for factories in [1, 4, 16, 64]:
        print(
            factories, runtime_from_tdepth(10**10, 10, factories)
        )

    # Example 08: ۷-۸. حافظه شبیه‌ساز کلاسیک
    def statevector_gib(n, bytes_per_amp=16):
        return 2**n * bytes_per_amp / 2**30


    for n in [30, 35, 40, 45, 50]:
        print(n, statevector_gib(n))

    # Example 09: ۷-۹. compilation روی اتصال خطی
    from qiskit import QuantumCircuit, transpile
    from qiskit.transpiler import CouplingMap

    qc = QuantumCircuit(8)
    for q in range(8):
        qc.h(q)
    for i in range(4):
        qc.cx(i, 7 - i)
    line = CouplingMap.from_line(8, bidirectional=True)
    for seed in range(5):
        out = transpile(
            qc,
            coupling_map=line,
            basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=2,
            seed_transpiler=seed,
        )
        print(seed, out.depth(), out.count_ops())

    # Example 10: ۷-۱۰. انرژی و هزینه راه‌حل
    def cost_solution(
        power_kw, hours, price_kwh, cloud, labor, valid
    ):
        energy = power_kw * hours * price_kwh
        return (energy + cloud + labor) / valid


    print(cost_solution(45, 3, 0.32, 120, 600, 25))

    # Example 11: ۷-۱۱. سناریوی مونت‌کارلو
    rng = np.random.default_rng(2026)
    trials = 100000
    p_physical = rng.beta(7, 3, size=trials)
    p_qec = rng.beta(5, 5, size=trials)
    p_algorithm = rng.beta(4, 6, size=trials)
    success = (
        (rng.random(trials) < p_physical)
        & (rng.random(trials) < p_qec)
        & (rng.random(trials) < p_algorithm)
    )
    print(
        success.mean(),
        np.quantile(
            p_physical * p_qec * p_algorithm, [0.05, 0.5, 0.95]
        ),
    )

    # Example 12: ۷-۱۲. scorecard ادعا
    weights = {
        "task": 0.2,
        "baseline": 0.2,
        "verification": 0.2,
        "resources": 0.2,
        "scaling": 0.2,
    }
    evidence = {
        "task": 4,
        "baseline": 2,
        "verification": 3,
        "resources": 1,
        "scaling": 2,
    }
    score = sum(weights[k] * evidence[k] for k in weights)
    print(
        score, evidence
    )  # امتیازها باید پیش از دیدن نتیجه و با شواهد پیوست‌شده تعیین شوند.

    return locals()


if __name__ == "__main__":
    main()
