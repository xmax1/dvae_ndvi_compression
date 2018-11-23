from dwave_sapi2.util import get_hardware_adjacency
from dwave_sapi2.embedding import embed_problem, unembed_answer
from dwave_sapi2.core import solve_ising
from dwave_sapi2.remote import RemoteConnection
from dwave.dwave import url, token
from gauge_mod.gauge_functions import ungauge_result
from network.network_v2 import connectivity

import numpy as np
import itertools
# remote_connection = RemoteConnection(url, token, proxy_url)


class sampler():
    def __init__(self, solver, nodes, inv_BETA=1.):

        remote_connection = RemoteConnection(url, token)
        # solver_names = remote_connection.solver_names()
        self.solver = remote_connection.get_solver(solver)
        self.adj = get_hardware_adjacency(self.solver)

        self.nodes = nodes
        self.nVisible = int(len(nodes) / 2)
        self.nHidden = int(len(nodes) / 2)
        self.inv_BETA = inv_BETA

        self.embedding, self.edges, self.variable_to_qubits, self.qubits_to_variables = connectivity(self.nVisible,
                                                                                                     self.nHidden)

        self.working_qubits = self.solver.properties["qubits"]
        self.working_couplers = [tuple(c) for c in self.solver.properties["couplers"]]
        self.used_working_qubits, self.used_working_intravariable_couplers = self.remove_bad_qubits(self.embedding)
        self.used_working_couplers = self.remove_bad_couplers(self.edges)


        self.J_dict = {}

        ### Code if not wanting to use embed function
        self.h_hdw = np.zeros(2048)
        self.J_hdw = {}
        # self.set_intra_qubit_couplings()

    def sample(self, h_graph, J_graph, gauge=None, nreads=1000, answer_mode='raw'):

        h_graph = np.clip(h_graph, -1., +1)
        J_graph = np.clip(J_graph, -1., +1.)
        self.J_graph_to_dict(J_graph)

        self.graph_to_hdw(h_graph, J_graph)

        # [h_hdw, J_hdw, jc, embeddings] = embed_problem(h_graph * self.inv_BETA * -1., self.J_dict, self.embedding, self.adj)
        # J_hdw.update(jc)

        ### Gauge
        if not gauge == None:
            self.h_hdw = self.h_hdw * gauge
            self.J_hdw = self.apply_gauge(self.J_hdw, gauge)

        ### Sample
        # result = solve_ising(self.solver, h0, j0, num_reads=nreads, answer_mode=answer_mode, auto_scale=False)
        result = solve_ising(self.solver, self.h_hdw, self.J_hdw, num_reads=nreads, answer_mode=answer_mode, auto_scale=False)

        ### Ungauge & ### Unembed answer
        if not gauge == None:
            ungauged_result = ungauge_result(result['solutions'], gauge)
            answer = self.unembed(ungauged_result)
        else:
            answer = self.unembed(result['solutions'])



        if answer_mode == 'raw':
            return answer
        elif answer_mode == 'histogram':
            return self.sample_histogram(answer, result)
        else:
            print 'Answer mode not recognised.'

    def sample_histogram(self, answer, result):
        frequencies = []
        samples = []
        for sample, frequency in itertools.izip(answer, result['num_occurrences']):
            sample = tuple(sample)
            if not sample in samples:
                samples.append(sample)
                frequencies.append(frequency)
            else:
                idx = samples.index(sample)
                frequencies[idx] += frequency
        samples_list = [list(sample) for sample in samples]
        return samples_list, frequencies

    def J_graph_to_dict(self, J):
        for i, visible_node in enumerate(self.nodes[:self.nVisible]):
            for j, hidden_node in enumerate(self.nodes[self.nVisible:]):
                self.J_dict[(visible_node, hidden_node)] = J[visible_node, hidden_node] * self.inv_BETA * -1.
        return

    def apply_gauge(self, j0, gauge):
        # If the gauges on two connected qubits are different, flip the sign on the J value
        tup_change_gauge = [(x, y) for (x, y) in j0.keys() if (gauge[x] == 1 and gauge[y] == -1)]
        tup_change_gauge2 = [(x, y) for (x, y) in j0.keys() if (gauge[x] == -1 and gauge[y] == 1)]
        tup_change_gauge = tup_change_gauge + tup_change_gauge2
        for tup in tup_change_gauge:
            j0[tup] = - j0[tup]
        return j0

    def remove_bad_qubits(self, embedding):
        used_working_qubits = []
        used_working_intravariable_couplers = []
        for chain in embedding:
            chain_tmp = []
            for qubit in chain:
                if not qubit in self.working_qubits: continue
                chain_tmp.append(qubit)
            used_working_qubits.append(chain_tmp)

            for i, qubit in enumerate(chain[:-1]):
                coupler = (chain[i], chain[i+1])
                if not coupler in self.working_couplers: continue
                used_working_intravariable_couplers.append(coupler)

        return used_working_qubits, used_working_intravariable_couplers

    def remove_bad_couplers(self, edges):
        used_working_couplers = []
        for edge in edges:
            if not tuple(edge) in self.working_couplers: continue
            used_working_couplers.append(tuple(edge))
        return used_working_couplers


    def graph_to_hdw(self, h_graph, J_graph):
        for i, chain in enumerate(self.used_working_qubits):
            for qubit in chain:
                self.h_hdw[qubit] = h_graph[i]

        for coupler in self.used_working_intravariable_couplers:
            self.J_hdw[coupler] = -1.

        for edge in self.used_working_couplers:
            self.J_hdw[edge] = J_graph[self.qubits_to_variables[edge[0]], self.qubits_to_variables[edge[1]]]

    def unembed(self, result):
        #broken_chains='vote'
        result = (np.asarray(result) + 1) * 0.5
        answer = np.zeros((len(result), len(self.nodes)))

        for i, logical_variable_chain in enumerate(self.used_working_qubits):
            x = result[:,logical_variable_chain]
            answer[:,i] = np.mean(x, axis=1)

        maj_vote = (answer == 0.5).astype(np.float32) * 0.5 * np.random.choice([-1, +1], size=(len(result), len(answer[0])))
        answer = np.around(answer + maj_vote)

        return answer


        # qubits = self.variable_to_qubits[(visible_node, hidden_node)]
        # if type(qubits) == list: continue
        # if tuple(qubits) not in self.working_couplers: print 'Coupler unused: ', qubits; continue
        # self.J_dict[(visible_node, hidden_node)] = J[visible_node,hidden_node] * self.inv_BETA * -1.


    ### Code if not wanting to use embed function

    #
    # def graph_to_hdw_h(self,h_graph):
    #     for i, el in enumerate(h_graph):
    #         for qubit in self.variable_to_qubits[(i, i)]:
    #             self.h_hdw[qubit] = self.inv_BETA * h_graph[i]
    #     return
    #
    # def graph_to_hdw_J(self,J_graph):
    #     for key, value in self.variable_to_qubits.iteritems():
    #         if key[0] == key[1]:
    #             continue
    #         else:
    #             # self.J_hdw[value] = self.BETA * J_graph[key]
    #             self.J_hdw[value] = J_graph[key]
    #     return
    #
    # def set_intra_qubit_couplings(self):
    #     for key, chain in self.variable_to_qubits.iteritems():
    #         if not key[0] == key[1]: continue
    #         else:
    #             for idx, el in enumerate(chain[:-1]):
    #                 if not (chain[idx], chain[idx+1]) in self.working_couplers: print (chain[idx], chain[idx+1]); continue
    #                 self.J_hdw[(chain[idx], chain[idx+1])] = -2.
    #     return






if __name__ == '__main__':
    print 'End'
    # sam = sampler()
    # h = [1, -1, 1, 1, -1, 1, 1]
    #
    # J = {(0, 6): 10}
    #
    #
    # x = sam.only_sample(h, J, nreads=10, answer_mode='raw')
    #
    # print x['solutions'][0][0:6]
    # embedding = None




