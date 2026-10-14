# -*- coding: utf-8 -*-
"""Chapter 25: یادگیری ماشین کوانتومی.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. داده مصنوعی و split
    import numpy as np

    rng = np.random.default_rng(2026)
    X = rng.uniform(-1, 1, size=(160, 2))
    y = (
        (X[:, 0] * X[:, 1] + 0.25 * np.sin(3 * X[:, 0])) > 0
    ).astype(float)
    idx = rng.permutation(len(X))
    train = idx[:110]
    test = idx[110:]
    mu = X[train].mean(0)
    sd = X[train].std(0)
    Xtr = (X[train] - mu) / sd
    Xte = (X[test] - mu) / sd
    ytr, yte = y[train], y[test]
    print(Xtr.shape, Xte.shape, ytr.mean(), yte.mean())

    # Example 02: ۷-۲. feature map زاویه‌ای
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector


    def feature_state(x):
        qc = QuantumCircuit(2)
        qc.ry(x[0], 0)
        qc.ry(x[1], 1)
        qc.cz(0, 1)
        qc.rz(x[0] * x[1], 1)
        return Statevector.from_instruction(qc).data


    print(feature_state(Xtr[0]))

    # Example 03: ۷-۳. Gram matrix کوانتومی
    def qkernel(A, B):
        SA = np.array([feature_state(x) for x in A])
        SB = np.array([feature_state(x) for x in B])
        return np.abs(SA.conj() @ SB.T) ** 2


    Ktr = qkernel(Xtr, Xtr)
    Kte = qkernel(Xte, Xtr)
    print(Ktr.shape, np.linalg.eigvalsh(Ktr).min())

    # Example 04: 7-4. kernel ridge classifier
    def kernel_ridge_predict(K_train, K_test, y, lam=0.1):
        target = 2 * y - 1
        alpha = np.linalg.solve(
            K_train + lam * np.eye(len(y)), target
        )
        score = K_test @ alpha
        return score, (score >= 0).astype(float)


    score, pred = kernel_ridge_predict(Ktr, Kte, ytr, 0.2)
    print(np.mean(pred == yte))

    # Example 05: 7-5. baseline RBF
    def rbf(A, B, gamma=1.0):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d2)


    for gamma in [0.1, 0.5, 1, 2, 5]:
        Kr = rbf(Xtr, Xtr, gamma)
        Krt = rbf(Xte, Xtr, gamma)
        _, pr = kernel_ridge_predict(Kr, Krt, ytr, 0.2)
        print(gamma, np.mean(pr == yte))

    # Example 06: ۷-۶. طیف و alignment
    target = 2 * ytr - 1


    def alignment(K, t):
        Y = np.outer(t, t)
        return np.sum(K * Y) / np.sqrt(
            np.sum(K * K) * np.sum(Y * Y)
        )


    evals = np.linalg.eigvalsh(Ktr)[::-1]
    print(alignment(Ktr, target), evals[:10], evals.sum())

    # Example 07: ۷-۷. مدل پارامتری یک کیوبیت
    Z = np.array([[1, 0], [0, -1]], complex)


    def qnn_value(x, theta):
        qc = QuantumCircuit(1)
        qc.ry(x, 0)
        qc.rz(x * x, 0)
        qc.ry(theta, 0)
        s = Statevector.from_instruction(qc).data
        return float(np.real(s.conj() @ Z @ s))


    print(qnn_value(0.7, 0.4))

    # Example 08: ۷-۸. بررسی parameter-shift
    x = 0.7
    theta = 0.4
    shift = (
        qnn_value(x, theta + np.pi / 2)
        - qnn_value(x, theta - np.pi / 2)
    ) / 2
    h = 1e-6
    finite = (
        qnn_value(x, theta + h) - qnn_value(x, theta - h)
    ) / (2 * h)
    print(
        shift, finite
    )  # این قاعده برای پارامترِ تنها دروازه RY بالا معتبر است.
    # پارامتر مشترک میان چند گیت باید رخدادبه‌رخداد مشتق‌گیری شود.

    # Example 09: ۷-۹. شات و uncertainty
    def estimate_pm1(mean, shots, seed):
        r = np.random.default_rng(seed)
        z = np.where(r.random(shots) < (1 + mean) / 2, 1, -1)
        return z.mean(), z.std(ddof=1) / np.sqrt(shots)


    true = qnn_value(0.7, 0.4)
    for shots in [100, 1000, 10000]:
        print(shots, estimate_pm1(true, shots, 2026))

    # Example 10: ۷-۱۰. kernel noisy آموزشی
    def depolarized_kernel(K, p, n_qubits=2):
        floor = 1 / (2**n_qubits)
        return (1 - p) ** 2 * K + (1 - (1 - p) ** 2) * floor


    for p in [0, 0.02, 0.1, 0.3]:
        Kn = depolarized_kernel(Ktr, p)
        print(
            p, alignment(Kn, target), np.linalg.eigvalsh(Kn)[-1]
        )

    # Example 11: 7-11. random-label audit
    yr = rng.permutation(ytr)
    _, train_pred = kernel_ridge_predict(Ktr, Ktr, yr, 1e-4)
    _, test_pred = kernel_ridge_predict(Ktr, Kte, yr, 1e-4)
    print("train random-label", np.mean(train_pred == yr))
    print("test true-label", np.mean(test_pred == yte))

    # Example 12: ۷-۱۲. هزینه ارزیابی kernel
    def kernel_circuits(n_train, n_test):
        return n_train * (n_train + 1) // 2 + n_train * n_test


    for n in [100, 1000, 10000]:
        print(n, kernel_circuits(n, n // 5))

    return locals()


if __name__ == "__main__":
    main()
