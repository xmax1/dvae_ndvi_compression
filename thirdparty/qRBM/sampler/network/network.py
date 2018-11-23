from dwave_sapi2.util import chimera_to_linear_index
import matplotlib.pyplot as plt

### This script initialises an RBM
def connectivity():
    ### Structure variables
    # m, n, t: the Chimera lattice consists of an mxn array of unit cells each consisting of 2t qubits
    t = 4
    k = range(t)
    n = 16
    m = 16
    # L/R of the graph respectively
    u = [0, 1]
    # The ith and jth row and column chimera graph respectively
    i = range(0, 8)
    j = range(0, 8)

    variable = 0
    variable_to_qubits = {}
    qubits_to_variables = {}

    # Vertical embedding
    V_embedding = []
    for hCell in j:
        for node in k:
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
    for vCell in i:
        for node in k:
            chain = []
            for hCell in j:
                idx = chimera_to_linear_index([vCell], [hCell], [u[1]], [node], m, n, t)
                chain.append(idx[0])
                qubits_to_variables[idx[0]] = variable
            H_embedding.append(chain)
            variable_to_qubits[(variable, variable)] = chain
            variable += 1

    edges = []
    # Generate the interqubit couplings
    for vCell in i:
        for vk in k:
            for hCell in j:
                for hk in k:
                    q1 = chimera_to_linear_index([vCell], [hCell], [u[0]], [vk], m, n, t)
                    q2 = chimera_to_linear_index([vCell], [hCell], [u[1]], [hk], m, n, t)
                    edges_idx = [q1[0], q2[0]]
                    edges.append(edges_idx)
                    variable_to_qubits[(qubits_to_variables[q1[0]], qubits_to_variables[q2[0]])] = (q1[0], q2[0])

    return V_embedding, H_embedding, edges, variable_to_qubits, qubits_to_variables

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



    # # Generate the vertical intraqubit chains
    # for hCell in j:
    #     for node in k:
    #         for vCell in list(i)[:-1]:
    #             # Use u[0] for vertical couplings
    #             ## Chimera indexing format
    #             # CH_idx1 = (vCell, hCell, u[0], node)
    #             # CH_idx2 = (vCell+1, hCell, u[0], node)
    #             lin_idx1 = chimera_to_linear_index([vCell+1], [hCell], [u[0]], [node], m, n, t)
    #             lin_idx2 = chimera_to_linear_index([vCell], [hCell], [u[0]], [node], m, n, t)
    #
    #         variables_to_qubits[(qubit, qubit)] = tmp
    #         qubit += 1

    # for vCell in i:
    #     for node in k:
    #         for hCell in list(j)[:-1]:
    #             ## Chimera indexing format
    #             # CH_idx1 = (vCell, hCell, u[1], node)
    #             # CH_idx2 = (vCell + 1, hCell, u[1], node)
    #             lin_idx1 = chimera_to_linear_index([vCell + 1], [hCell], [u[1]], [node], m, n, t)
    #             lin_idx2 = chimera_to_linear_index([vCell], [hCell], [u[1]], [node], m, n, t)
    #             H_connectivity[(lin_idx1, lin_idx2)] = (qubit, qubit)
    #         H_qubits.append(qubit)
    #         qubit += 1

# nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
#
# G.add_edge('a', 'b', weight=0.6)
# G.add_edge('a', 'c', weight=0.2)
# G.add_edge('c', 'd', weight=0.1)
# G.add_edge('c', 'e', weight=0.7)
# G.add_edge('c', 'f', weight=0.9)
# G.add_edge('a', 'd', weight=0.3)
#
# elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
# esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]
#
# pos = nx.spring_layout(G)  # positions for all nodes
#
# # nodes
# nx.draw_networkx_nodes(G, pos, node_size=700)
#
# # edges
# nx.draw_networkx_edges(G, pos, edgelist=elarge,
#                        width=6)
# nx.draw_networkx_edges(G, pos, edgelist=esmall,
#                        width=6, alpha=0.5, edge_color='b', style='dashed')
#
# # labels
# nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

