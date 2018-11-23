import networkx as nx
import matplotlib.pyplot as plt
from tools import flatten


def plot_network(n, V_embedding, H_embedding, edges):
    G = nx.Graph()
    i = 1
    for chain in V_embedding:
        j = 5
        for node in chain:
            G.add_node(node, pos=(i,j))
            j += 5
        i += 1
        if i % 5 == 0:
            i += 1
    j = 1
    for chain in H_embedding:
        i = 5
        for node in chain:
            G.add_node(node, pos=(i,j))
            i += 5
        j += 1
        if j % 5 == 0:
            j += 1

    #https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx_nodes.html#networkx.drawing.nx_pylab.draw_networkx_nodes
    for chain in V_embedding:
        for idx in range(len(chain)-1):
            G.add_edge(chain[idx], chain[idx+1], weight=1)
    for chain in H_embedding:
        for idx in range(len(chain)-1):
            G.add_edge(chain[idx], chain[idx+1], weight=1)
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=2)


    pos=nx.get_node_attributes(G,'pos')
    nx.draw_networkx_nodes(G, pos, nodelist=flatten(V_embedding), node_color='red', node_size=10)
    nx.draw_networkx_nodes(G, pos, nodelist=flatten(H_embedding), node_color='blue', node_size=10)
    nx.draw_networkx_edges(G, pos)
    plt.axis('off')
    plt.show()
    plt.savefig('./networks/n_%i.png' % n)
