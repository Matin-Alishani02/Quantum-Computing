# -*- coding: utf-8 -*-
"""Chapter 18: الگوریتم‌های کوانتومی تغییرپذیر.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. انرژی یک‌کیوبیتی با NumPy
    import numpy as np


    def energy_1q(theta):
        psi = np.array([np.cos(theta / 2), np.sin(theta / 2)])
        Z = np.diag([1.0, -1.0])
        return float(np.real(psi.conj() @ Z @ psi))


    grid = np.linspace(0, 2 * np.pi, 101)
    best = min(grid, key=energy_1q)
    print(best, energy_1q(best))

    # Example 02: ۷-۲. مدار ansatz در Qiskit
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    theta = ParameterVector("theta", 2)
    ansatz = QuantumCircuit(2)
    ansatz.ry(theta[0], 0)
    ansatz.ry(theta[1], 1)
    ansatz.cx(0, 1)
    print(ansatz)

    # Example 03: ۷-۳. امیدریاضی دقیق SparsePauliOp
    from qiskit.quantum_info import SparsePauliOp, Statevector

    H = SparsePauliOp.from_list(
        [("ZZ", -1.0), ("IX", -0.5), ("XI", -0.5)]
    )


    def exact_energy(values):
        bound = ansatz.assign_parameters(dict(zip(theta, values)))
        state = Statevector.from_instruction(bound)
        return float(np.real(state.expectation_value(H)))


    print(exact_energy([0.2, -0.4]))

    # Example 04: ۷-۴. جست‌وجوی شبکه‌ای بازتولیدپذیر
    grid = np.linspace(-np.pi, np.pi, 41)
    best_E, best_theta = np.inf, None
    for a in grid:
        for b in grid:
            E = exact_energy([a, b])
            if E < best_E:
                best_E, best_theta = E, (a, b)
    print(best_E, best_theta)

    # Example 05: ۷-۵. parameter-shift چندپارامتری
    def parameter_shift(values, index):
        plus, minus = np.array(values, float), np.array(
            values, float
        )
        plus[index] += np.pi / 2
        minus[index] -= np.pi / 2
        return 0.5 * (exact_energy(plus) - exact_energy(minus))


    print([parameter_shift([0.2, -0.4], j) for j in range(2)])

    # Example 06: ۷-۶. یک نزول گرادیانی ساده
    values = np.array([0.3, -0.2])
    eta = 0.15
    for step in range(40):
        grad = np.array(
            [parameter_shift(values, j) for j in range(2)]
        )
        values -= eta * grad
    print(values, exact_energy(values))

    # Example 07: ۷-۷. QAOA برای گراف مثلث
    def qaoa_triangle(gamma, beta):
        qc = QuantumCircuit(3)
        qc.h(range(3))
        for i, j in [(0, 1), (1, 2), (0, 2)]:
            qc.cx(i, j)
            qc.rz(-gamma, j)
            qc.cx(i, j)
        for q in range(3):
            qc.rx(2 * beta, q)
        return qc


    state = Statevector.from_instruction(qaoa_triangle(0.7, 0.4))
    print(state.probabilities_dict())

    # Example 08: ۷-۸. هزینهٔ کلاسیک نمونه‌ها
    edges = [(0, 1), (1, 2), (0, 2)]


    def cut_value(bits):
        # Qiskit strings show the highest-index qubit on the left.
        q = list(map(int, bits[::-1]))
        return sum(q[i] != q[j] for i, j in edges)


    probs = state.probabilities_dict()
    expected = sum(p * cut_value(b) for b, p in probs.items())
    print(expected)

    # Example 09: ۷-۹. شبیه‌سازی shots با NumPy
    rng = np.random.default_rng(2026)
    labels = list(probs)
    p = np.array([probs[x] for x in labels])
    samples = rng.choice(labels, size=1024, p=p / p.sum())
    values = np.array([cut_value(x) for x in samples])
    print(
        values.mean(), values.std(ddof=1) / np.sqrt(len(values))
    )

    # Example 10: ۷-۱۰. baseline دقیق
    matrix = H.to_matrix()
    exact_ground = np.linalg.eigvalsh(matrix)[0]
    print(
        "exact=",
        exact_ground,
        "VQE=",
        exact_energy(values=[0.0, 0.0]),
    )

    # Example 11: ۷-۱۱. تخصیص تطبیقی shots
    coeff = np.array([1.0, 0.5, 0.5])
    sigma = np.array([0.3, 0.9, 0.6])
    # pilot estimates
    budget = 6000
    raw = np.abs(coeff) * sigma
    shots = np.maximum(
        1, np.rint(budget * raw / raw.sum())
    ).astype(int)
    print(shots, shots.sum())

    # Example 12: ۷-۱۲. چند شروع مستقل
    rng = np.random.default_rng(2026)
    results = []
    for seed in range(12):
        values = rng.uniform(-np.pi, np.pi, size=2)
        for _ in range(30):
            grad = np.array(
                [parameter_shift(values, j) for j in range(2)]
            )
            values -= 0.12 * grad
        results.append(exact_energy(values))
    print(
        np.min(results), np.mean(results), np.std(results, ddof=1)
    )

    # Example 13: ۷-۱۳. کران overlap از انرژی
    evals = np.linalg.eigvalsh(H.to_matrix())
    E0, E1 = evals[:2]
    Etrial = min(results)
    gap = E1 - E0
    lower_bound = max(0.0, 1.0 - (Etrial - E0) / gap)
    print("ground-state fidelity >=", lower_bound)

    return locals()


if __name__ == "__main__":
    main()
