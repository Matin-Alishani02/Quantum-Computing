# -*- coding: utf-8 -*-
"""Chapter 24: بهینه‌سازی کوانتومی.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. ارزیابی Max-Cut
    import numpy as np
    from itertools import product

    edges = [(0, 1), (1, 2), (2, 0), (2, 3)]


    def cut_value(bits):
        return sum(bits[i] != bits[j] for i, j in edges)


    all_bits = list(product([0, 1], repeat=4))
    values = np.array([cut_value(x) for x in all_bits])
    print(
        values.max(),
        [
            x
            for x, v in zip(all_bits, values)
            if v == values.max()
        ],
    )

    # Example 02: ۷-۲. ساخت QUBO Max-Cut
    def maxcut_qubo(n, edges):
        Q = np.zeros((n, n))
        for i, j in edges:
            Q[i, i] -= 1
            Q[j, j] -= 1
            Q[i, j] += 2
        return Q


    Q = maxcut_qubo(4, edges)
    for x in all_bits[:5]:
        print(x, np.array(x) @ Q @ np.array(x), -cut_value(x))

    # Example 03: ۷-۳. تحلیل penalty
    def penalized(x, lam):
        x = np.array(x)
        objective = -10 * x[0] * x[1]
        violation = (x.sum() - 1) ** 2
        return objective + lam * violation


    for lam in [1, 5, 11]:
        scores = [
            penalized(x, lam) for x in product([0, 1], repeat=2)
        ]
        print(lam, np.argmin(scores), scores)

    # Example 04: ۷-۴. مدار QAOA با p=1
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector


    def qaoa_state(n, edges, gamma, beta):
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for i, j in edges:
            qc.rzz(-gamma, i, j)
        for q in range(n):
            qc.rx(2 * beta, q)
        return Statevector.from_instruction(qc)


    def expected_cut(state, edges):
        probs = state.probabilities()
        return sum(
            p * cut_value(tuple(map(int, format(k, "04b")[::-1])))
            for k, p in enumerate(probs)
        )


    print(expected_cut(qaoa_state(4, edges, 1.0, 0.5), edges))

    # Example 05: ۷-۵. جست‌وجوی شبکه‌ای زاویه‌ها
    gammas = np.linspace(0, 2 * np.pi, 61)
    betas = np.linspace(0, np.pi, 41)
    best = (-1, None)
    for g in gammas:
        for b in betas:
            val = expected_cut(qaoa_state(4, edges, g, b), edges)
            if val > best[0]:
                best = (val, (g, b))
    print(best)

    # Example 06: ۷-۶. نمونه‌برداری پاسخ نهایی
    state = qaoa_state(4, edges, *best[1])
    state.seed(2026)
    counts = state.sample_counts(5000)
    ranked = sorted(
        counts.items(), key=lambda kv: kv[1], reverse=True
    )
    print(
        [
            (s, c, cut_value(tuple(map(int, s[::-1]))))
            for s, c in ranked[:8]
        ]
    )

    # Example 07: ۷-۷. baseline تصادفی
    rng = np.random.default_rng(2026)


    def random_best(k, trials=10000):
        out = []
        for _ in range(trials):
            samples = rng.integers(0, 2, size=(k, 4))
            out.append(max(cut_value(tuple(x)) for x in samples))
        return np.mean(out), np.quantile(out, [0.05, 0.5, 0.95])


    for k in [1, 5, 20]:
        print(k, random_best(k))

    # Example 08: 7-8. local search
    def one_flip_search(bits):
        bits = list(bits)
        improved = True
        while improved:
            improved = False
            current = cut_value(bits)
            for q in range(len(bits)):
                cand = bits.copy()
                cand[q] ^= 1
                if cut_value(cand) > current:
                    bits = cand
                    improved = True
                    break
        return tuple(bits), cut_value(bits)


    print(
        [one_flip_search(rng.integers(0, 2, 4)) for _ in range(8)]
    )

    # Example 09: ۷-۹. success probability و TTS
    def tts_runs(p, target=0.99):
        if p <= 0:
            return np.inf
        if p >= 1:
            return 1
        return int(np.ceil(np.log(1 - target) / np.log(1 - p)))


    probs = state.probabilities()
    p_opt = sum(
        p
        for k, p in enumerate(probs)
        if cut_value(tuple(map(int, format(k, "04b")[::-1])))
        == values.max()
    )
    print(p_opt, tts_runs(p_opt))

    # Example 10: ۷-۱۰. CVaR از نمونه‌ها
    def cvar_best(scores, alpha=0.2):
        scores = np.sort(np.asarray(scores))[::-1]
        m = max(1, int(np.ceil(alpha * len(scores))))
        return scores[:m].mean()


    samples = state.sample_memory(5000)
    scores = [
        cut_value(tuple(map(int, s[::-1]))) for s in samples
    ]
    print(np.mean(scores), cvar_best(scores, 0.1))

    # Example 11: ۷-۱۱. تحلیل چند seed
    def noisy_objective(g, b, shots, seed):
        st = qaoa_state(4, edges, g, b)
        st.seed(seed)
        ss = st.sample_memory(shots)
        return np.mean(
            [cut_value(tuple(map(int, s[::-1]))) for s in ss]
        )


    for seed in range(5):
        print(seed, noisy_objective(*best[1], 400, seed))

    # Example 12: ۷-۱۲. حساب هزینه end-to-end
    cost = {
        "mapping_s": 0.03,
        "compile_s": 0.12,
        "evals": 250,
        "shots_per_eval": 1000,
        "circuit_s": 80e-6,
        "decode_s": 0.02,
    }
    total = (
        cost["mapping_s"]
        + cost["compile_s"]
        + cost["evals"]
        * cost["shots_per_eval"]
        * cost["circuit_s"]
        + cost["decode_s"]
    )
    print(total, cost)

    return locals()


if __name__ == "__main__":
    main()
