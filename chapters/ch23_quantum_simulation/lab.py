# -*- coding: utf-8 -*-
"""Chapter 23: شبیه‌سازی سامانه‌های کوانتومی.
Examples run in book order and share the same chapter context.
"""


def main():

    # Example 01: ۷-۱. ساخت ماتریس‌های چندکیوبیتی
    import numpy as np
    from functools import reduce

    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]], complex)
    Z = np.diag([1, -1])


    def kron_all(ops):
        return reduce(np.kron, ops)


    def op_on(pauli, q, n):
        return kron_all(
            [pauli if i == q else I for i in range(n)]
        )


    print(op_on(X, 1, 3).shape)

    # Example 02: ۷-۲. هامیلتونی Ising
    def ising_hamiltonian(n, J=1.0, h=0.7):
        H = np.zeros((2**n, 2**n), complex)
        for i in range(n - 1):
            H -= J * op_on(Z, i, n) @ op_on(Z, i + 1, n)
        for i in range(n):
            H -= h * op_on(X, i, n)
        return H


    H = ising_hamiltonian(4)
    print(np.linalg.eigvalsh(H)[:4])

    # Example 03: ۷-۳. تکامل دقیق
    def evolve_exact(H, psi0, t):
        e, V = np.linalg.eigh(H)
        return V @ (np.exp(-1j * t * e) * (V.conj().T @ psi0))


    psi0 = np.zeros(16, complex)
    psi0[0] = 1
    psi = evolve_exact(H, psi0, 1.2)
    print(np.vdot(psi, psi).real)

    # Example 04: ۷-۴. تکامل پائولی در Qiskit
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit.synthesis import SuzukiTrotter

    n = 4
    terms = [("IIZZ", -1), ("IZZI", -1), ("ZZII", -1)]
    terms += [
        ("XIII", -0.7),
        ("IXII", -0.7),
        ("IIXI", -0.7),
        ("IIIX", -0.7),
    ]
    Hq = SparsePauliOp.from_list(terms)
    qc = QuantumCircuit(n)
    qc.append(
        PauliEvolutionGate(
            Hq, time=1.2, synthesis=SuzukiTrotter(order=2, reps=4)
        ),
        range(n),
    )
    psi_q = Statevector.from_instruction(qc)
    print(psi_q.probabilities_dict())

    # Example 05: ۷-۵. خطای Trotter بر حسب reps
    exact = Statevector(evolve_exact(H, psi0, 1.2))
    for reps in [1, 2, 4, 8, 16]:
        qc = QuantumCircuit(n)
        gate = PauliEvolutionGate(
            Hq,
            time=1.2,
            synthesis=SuzukiTrotter(order=2, reps=reps),
        )
        qc.append(gate, range(n))
        approx = Statevector.from_instruction(qc)
        fidelity = abs(np.vdot(exact.data, approx.data)) ** 2
        print(reps, 1 - fidelity)

    # Example 06: ۷-۶. امید مغناطش
    mag = sum(op_on(Z, q, n) for q in range(n)) / n
    print(float(np.real(psi.conj() @ mag @ psi)))

    # Example 07: ۷-۷. برآورد شاتی یک رشته پائولی
    def sample_pauli_expectation(mean, shots, rng):
        p_plus = (1 + mean) / 2
        samples = np.where(rng.random(shots) < p_plus, 1, -1)
        return samples.mean(), samples.std(ddof=1) / np.sqrt(
            shots
        )


    rng = np.random.default_rng(2026)
    for shots in [100, 1000, 10000]:
        print(shots, sample_pauli_expectation(0.35, shots, rng))

    # Example 08: ۷-۸. VQE تک‌پارامتری آموزشی
    from qiskit.quantum_info import Operator


    def ansatz(theta):
        qc = QuantumCircuit(2)
        qc.ry(theta, 0)
        qc.cx(0, 1)
        return Statevector.from_instruction(qc).data


    H2 = np.kron(Z, Z) + 0.7 * (np.kron(X, I) + np.kron(I, X))
    grid = np.linspace(-np.pi, np.pi, 401)
    energies = [
        np.real(ansatz(t).conj() @ H2 @ ansatz(t)) for t in grid
    ]
    print(
        grid[np.argmin(energies)],
        min(energies),
        np.linalg.eigvalsh(H2)[0],
    )

    # Example 09: ۷-۹. Jordan-Wigner برای عملگر عدد
    # n_j = a_j^dagger a_j = (I-Z_j)/2
    def number_operator(j, n):
        return (np.eye(2**n) - op_on(Z, j, n)) / 2


    print(np.diag(number_operator(1, 3)))

    # Example 10: ۷-۱۰. معادله Lindblad یک کیوبیت
    sm = np.array([[0, 1], [0, 0]], complex)


    def lindblad_step(rho, H, L, dt):
        dr = -1j * (H @ rho - rho @ H)
        dr += (
            L @ rho @ L.conj().T
            - (L.conj().T @ L @ rho + rho @ L.conj().T @ L) / 2
        )
        return rho + dt * dr


    rho = np.array([[0, 0], [0, 1]], complex)
    L = np.sqrt(0.4) * sm
    for _ in range(1000):
        rho = lindblad_step(rho, np.zeros((2, 2)), L, 0.002)
    print(np.real(np.diag(rho)), np.trace(rho))

    # Example 11: ۷-۱۱. آزمون قانون پایستگی
    def commutator(A, B):
        return A @ B - B @ A


    parity = kron_all([X] * 4)
    print(np.linalg.norm(commutator(H, parity)))

    # Example 12: ۷-۱۲. بودجه خطا
    budget = {
        "model": 2e-3,
        "mapping": 1e-3,
        "algorithm": 4e-3,
        "sampling": 3e-3,
        "device": 8e-3,
    }
    print(sum(budget.values()), budget)

    return locals()


if __name__ == "__main__":
    main()
