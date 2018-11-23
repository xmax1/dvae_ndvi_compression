from network_v2 import connectivity
from plot_network import plot_network

n_ = range(8, 32, 2)

for n in n_:
    nV = int(n/2)
    nH = int(n/2)
    V_embedding, H_embedding, edges, variable_to_qubits, qubits_to_variables = connectivity(nV = nV, nH = nH)

    plot_network(n, V_embedding, H_embedding, edges)
