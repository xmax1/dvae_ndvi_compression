import numpy as np

def generate_gauges(embedding, nQubits, n_Gauges):
    gauge_list = []
    zeros = np.zeros(nQubits)
    qubits = [qubit for chain in embedding for qubit in chain]

    ones_gauge = zeros.copy()
    ones_gauge[qubits] += np.ones(len(qubits))
    gauge_list.append(ones_gauge)
    minus_ones_gauge = zeros.copy()
    minus_ones_gauge[qubits] -= np.ones(len(qubits))
    gauge_list.append(minus_ones_gauge)

    for _ in range(n_Gauges - 2):
        random = np.random.choice([-1.,+1.], len(qubits))
        gauge_random = zeros.copy()
        gauge_random[qubits] += random
        gauge_list.append(gauge_random)

    return gauge_list

def ungauge_result(result, gauge):
    return np.asarray(result) * np.asarray(gauge)