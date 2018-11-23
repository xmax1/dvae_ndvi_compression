from dwave_sapi2.util import chimera_to_linear_index
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
### This script initialises an RBM
def connectivity(nV, nH):
    ### Structure variables
    # m, n, t: the Chimera lattice consists of an mxn array of unit cells each consisting of 2t qubits
    t = 4
    k = range(t)
    n = 16
    m = 16
    # L/R of the graph respectively
    u = [0, 1]

    start_i = np.random.randint(0, m  - int(ceil(nV/4.)), 1)
    start_j = np.random.randint(0, n - int(ceil(nH/4.)), 1)
    # The ith and jth row and column chimera graph respectively

    ### This is the number of graph structures
    i = range(start_i, start_i + int(ceil(nV/4.)))
    j = range(start_j, start_j + int(ceil(nH/4.)))

    ### This is the corresponding number of nodes in each graph
    kV = [4 for _ in i]
    kH = [4 for _ in i]

    kV[-1] = nV % 4
    kH[-1] = nH % 4
    if nV % 4 == 0: kV[-1] = 4
    if nH % 4 == 0: kH[-1] = 4

    variable = 0
    variable_to_qubits = {}
    qubits_to_variables = {}

    # Vertical embedding
    V_embedding = []
    for k_i, hCell in enumerate(j):
        for node in range(kV[k_i]):
            chain = []
            for vCell in i:
                idx = chimera_to_linear_index([vCell], [hCell], [u[0]], [node], m, n, t)
                chain.append(idx[0])
                qubits_to_variables[idx[0]] = variable
            V_embedding.append(chain)
            variable_to_qubits[(variable, variable)] = chain


            variable += 1

    ## Horizontal embedding
    H_embedding = []
    for k_i, vCell in enumerate(i):
        for node in range(kH[k_i]):
            chain = []
            for hCell in j:
                idx = chimera_to_linear_index([vCell], [hCell], [u[1]], [node], m, n, t)
                chain.append(idx[0])
                qubits_to_variables[idx[0]] = variable
            H_embedding.append(chain)
            variable_to_qubits[(variable, variable)] = chain
            variable += 1
    # Vertical embedding
    # V_embedding = []
    # for hCell in j:
    #     for node in k:
    #         chain = []
    #         for vCell in i:
    #             idx = chimera_to_linear_index([vCell], [hCell], [u[0]], [node], m, n, t)
    #             chain.append(idx[0])
    #             qubits_to_variables[idx[0]] = variable
    #         V_embedding.append(chain)
    #         variable_to_qubits[(variable, variable)] = chain
    #
    #         variable += 1

    # ## Horizontal embedding
    # H_embedding = []
    # for vCell in i:
    #     for node in k:
    #         chain = []
    #         for hCell in j:
    #             idx = chimera_to_linear_index([vCell], [hCell], [u[1]], [node], m, n, t)
    #             chain.append(idx[0])
    #             qubits_to_variables[idx[0]] = variable
    #         H_embedding.append(chain)
    #         variable_to_qubits[(variable, variable)] = chain
    #         variable += 1


    edges = []
    # Generate the interqubit couplings
    for k_i, vCell in enumerate(i):
        for hk in range(kV[k_i]):
            for k_j, hCell in enumerate(j):
                for vk in range(kH[k_j]):


                    q1 = chimera_to_linear_index([vCell], [hCell], [u[0]], [vk], m, n, t)
                    q2 = chimera_to_linear_index([vCell], [hCell], [u[1]], [hk], m, n, t)
                    edges_idx = [q1[0], q2[0]]
                    edges.append(edges_idx)
                    variable_to_qubits[(qubits_to_variables[q1[0]], qubits_to_variables[q2[0]])] = (q1[0], q2[0])

    return V_embedding + H_embedding, edges, variable_to_qubits, qubits_to_variables

def intra_couplings_f(J, embedding):
    for chain in embedding:
        for idx in range(len(chain) - 1):
            J.append((chain[idx], chain[idx+1]))
    return J

def inter_couplings_f(edges):
    tmp = []
    for lst in edges:
        tmp.append((lst[0], lst[1]))
    return tmp


