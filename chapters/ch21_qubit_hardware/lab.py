# -*- coding: utf-8 -*-
"""Chapter 21: فناوری‌های ساخت کیوبیت.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. جمعیت گرمایی
    import numpy as np

    h = 6.62607015e-34
    kB = 1.380649e-23


    def excited_fraction(freq_hz, temp_k):
        ratio = np.exp(-h * freq_hz / (kB * temp_k))
        return ratio / (1 + ratio)


    print(excited_fraction(5e9, 0.015))

    # Example 02: ۷-۲. طیف تقریبی ترانسمون
    def transmon_scales(EJ_over_h, EC_over_h):
        f01 = np.sqrt(8 * EJ_over_h * EC_over_h) - EC_over_h
        anharm = -EC_over_h
        return f01, anharm


    print(transmon_scales(18e9, 0.25e9))

    # Example 03: ۷-۳. نسبت زمان انسجام به گیت
    platforms = {
        "A": {"T2": 100e-6, "gate": 50e-9},
        "B": {"T2": 1.0, "gate": 100e-6},
    }
    for name, x in platforms.items():
        print(name, x["T2"] / x["gate"])

    # Example 04: ۷-۴. survival فوتونی
    def survival(efficiencies):
        return float(np.prod(efficiencies))


    print(survival([0.995] * 50))
    print(survival([0.99, 0.97, 0.95, 0.92]))

    # Example 05: ۷-۵. مدار آزمون topology
    from qiskit import QuantumCircuit, transpile
    from qiskit.transpiler import CouplingMap

    qc = QuantumCircuit(5)
    qc.h(0)
    for q in [4, 3, 2, 1]:
        qc.cx(0, q)
    line = CouplingMap.from_line(5, bidirectional=True)
    compiled = transpile(
        qc,
        coupling_map=line,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=2,
        seed_transpiler=2026,
    )
    print(compiled.depth(), compiled.count_ops())

    # Example 06: ۷-۶. مقایسه با اتصال کامل
    full_edges = [
        (i, j) for i in range(5) for j in range(5) if i != j
    ]
    full = CouplingMap(full_edges)
    compiled_full = transpile(
        qc,
        coupling_map=full,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=2,
        seed_transpiler=2026,
    )
    print(compiled_full.depth(), compiled_full.count_ops())

    # Example 07: ۷-۷. شمارش هزینه SWAP
    def two_qubit_count(circuit):
        ops = circuit.count_ops()
        return int(
            ops.get("cx", 0)
            + ops.get("cz", 0)
            + ops.get("ecr", 0)
        )


    print(
        two_qubit_count(compiled), two_qubit_count(compiled_full)
    )

    # Example 08: ۷-۸. ماتریس تصمیم شفاف
    criteria = [
        "speed",
        "coherence",
        "connectivity",
        "fabrication",
        "network",
    ]
    weights = np.array(
        [0.25, 0.20, 0.20, 0.20, 0.15]
    )  # Hypothetical pedagogical scores, not current hardware benchmarks.
    scores = {
        "platform_A": np.array([5, 3, 3, 5, 2]),
        "platform_B": np.array([2, 5, 5, 2, 3]),
    }
    for name, s in scores.items():
        print(name, float(weights @ s))

    # Example 09: ۷-۹. حساسیت رتبه‌بندی به وزن‌ها
    rng = np.random.default_rng(2026)
    wins = {name: 0 for name in scores}
    for _ in range(10000):
        w = rng.dirichlet(np.ones(len(criteria)))
        winner = max(scores, key=lambda name: w @ scores[name])
        wins[winner] += 1
    print(wins)

    # Example 10: ۷-۱۱. شبیه‌سازی yield
    rng = np.random.default_rng(2026)


    def usable_chips(n_qubits, per_qubit_yield, chips=100000):
        good = rng.random((chips, n_qubits)) < per_qubit_yield
        return good.all(axis=1).mean(), good.sum(axis=1)


    full, counts = usable_chips(100, 0.995)
    print(full, np.quantile(counts, [0.05, 0.5, 0.95]))

    # Example 11: ۷-۱۲. برخورد فرکانسی ساده
    def collision_rate(
        n, sigma_mhz, min_spacing_mhz, trials=5000
    ):
        bad = 0
        for _ in range(trials):
            f = np.sort(rng.normal(5000, sigma_mhz, size=n))
            bad += np.any(np.diff(f) < min_spacing_mhz)
        return bad / trials


    print(collision_rate(12, 250, 40))

    # Example 12: ۷-۱۳. نقص یال و هزینه routing
    broken = CouplingMap(
        [
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 4),
            (4, 3),
        ]
    )
    compiled_broken = transpile(
        qc,
        coupling_map=broken,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=2,
        seed_transpiler=2026,
    )
    print(compiled_broken.depth(), compiled_broken.count_ops())

    return locals()


if __name__ == "__main__":
    main()
