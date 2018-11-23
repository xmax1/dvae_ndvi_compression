import pickle as p

import numpy as np

from sampler.sampler import sampler


class qRBM_():
    def __init__(self, nodes, gauge=None, batch_size=64, lr = 0.0001, load=False, binarize_threshold=None,
                 method_sampler='cd',
                 h_init=[], J_init=[], inv_BETA=1.):

        self.epoch = 0
        self.method_sampler=method_sampler
        self.batch_size = batch_size
        self.nodes = nodes
        self.nNodes = len(nodes)
        self.nVisible = int(len(nodes)/2)
        self.nHidden = int(len(nodes)/2)
        self.nQubits = 2048
        self.lr = lr
        self.inv_BETA = inv_BETA
        self.gauge = gauge

        ### Sampler
        if not method_sampler == 'cd':
            self.sam = sampler(method_sampler, nodes, inv_BETA=inv_BETA)

        self.J_graph = np.zeros((self.nNodes, self.nNodes))
        self.h_graph = np.zeros(self.nNodes)

        if load == True:
            self.h_graph = self.load('h')
            self.J_graph = self.load('J')
        elif h_init == [] and J_init == []:
            self.J_graph = np.random.normal(0,0.1,(self.nNodes, self.nNodes))
        else:
            self.h_graph = h_init
            self.J_graph = J_init

        # plot_network(V_embedding, H_embedding, edges)
        # self.data = dataset(threshold=binarize_threshold)

        self.plotting = {}
        self.plotting['errs_total'] = []
        self.plotting['errs_sq'] = []
        self.plotting['errs'] = []
        self.plotting['epochs'] = []
        self.plotting['hamming'] = []

        # Gradients
        print 'qRBM initialised'

    ### Training
    def grads(self):
        pos_h, pos_J = self.pos_phase(self.batch_size)
        neg_h, neg_J = self.neg_phase(self.batch_size)
        tmp1 = pos_h - neg_h
        tmp2 = pos_J - neg_J
        self.dh = - self.lr * (tmp1)
        self.dJ = - self.lr * (tmp2)
        return

    def update(self):
        self.h_graph = self.h_graph + self.dh
        self.J_graph = self.J_graph + self.dJ
        self.epoch += 1
        return

    def pos_phase(self, batch_size):
        batch = self.data.next_batch(batch_size)
        self.hidden_states = self.sigmoid(np.add(np.matmul(batch, self.J_graph[:self.nVisible, self.nVisible:]), self.h_graph[self.nVisible:]))
        vec = self.data.binarize(np.concatenate((batch, self.hidden_states), axis=1))
        pos_h = np.mean(vec, axis=0)
        pos_J = np.mean(np.matmul(vec.T, vec) / batch_size, axis=0)
        return pos_h, pos_J

    def neg_phase(self, batch_size, answer_mode='raw'):
        if self.method_sampler == 'cd':
            samples = self.sample_cd()

        else:
            samples = np.asarray(self.sam.sample(self.h_graph, self.J_graph, self.gauge, nreads=batch_size, answer_mode=answer_mode))

        neg_h = np.mean(samples, axis=0)
        neg_J = np.mean(np.matmul(samples.T, samples) / batch_size, axis=0)

        return neg_h, neg_J

    ### Graph mapping

    def sample(self, gauge=None, nreads=1000, answer_mode='raw'):
        # print 'Sampling'
        return self.sam.sample(self.h_graph, self.J_graph, gauge, nreads=nreads, answer_mode=answer_mode)

    def sample_cd(self):
        neg_vis = self.sigmoid(np.add(np.matmul(self.hidden_states, self.J_graph[:self.nVisible, self.nVisible:].T),
                                      self.h_graph[:self.nVisible]))
        neg_hidden = self.sigmoid(
            np.add(np.matmul(neg_vis, self.J_graph[:self.nVisible, self.nVisible:]), self.h_graph[self.nVisible:]))

        return self.data.binarize(np.concatenate((neg_vis, neg_hidden), axis=1))


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def l1_norm(self, batch_size):
        pos_h, pos_J = self.pos_phase(batch_size)
        neg_h, neg_J = self.neg_phase(batch_size)
        return np.sum(np.abs(pos_J - neg_J))


    # ### Evaluation
    #
    # def evaluate(self, nreads=32):
    #     idxs = np.random.randint(0, len(self.data.tst_data), nreads)
    #     visible_states = self.data.binarize(self.data.tst_data[idxs])
    #     self.hidden_states = self.sigmoid(
    #         np.add(np.matmul(visible_states, self.J_graph[:self.nVisible, self.nVisible:]),
    #                self.h_graph[self.nVisible:]))
    #     self.hidden_states = self.data.binarize(self.hidden_states)
    #
    #     vec1 = np.concatenate((visible_states, self.hidden_states), axis=1)
    #     vec2, _, _ = self.neg_phase(nreads=nreads)
    #     self.negative_examples = vec2[:self.nVisible]
    #     self.positive_examples = visible_states
    #     self.plotting['errs_sq'].append(np.mean((vec1 - vec2) ** 2))
    #     x = np.mean(self.hamming(vec1, vec2))
    #     self.plotting['hamming'].append(np.mean(self.hamming(vec1, vec2)))
    #     self.plotting['epochs'].append(self.epoch)
    #
    #     return np.mean(self.hamming(vec1, vec2))

    def hamming(self, x, y):
        hamming_weights_min = []
        for row_x in x:
            hamming_weights_all = []
            for row_y in y:
                hamming_weights_all.append(np.sum(np.absolute(row_x - row_y)))
            hamming_weights_all = np.asarray(hamming_weights_all)
            # x = np.argpartition(hamming_weights_all, 5)
            # y = hamming_weights_all[np.argpartition(hamming_weights_all, 5)]

            hamming_weights_min.append(np.mean(hamming_weights_all[np.argpartition(hamming_weights_all, 5)]))
        return hamming_weights_min


    def save_config(self, path='./models/'):
        with open(path + 'h.p', 'wb') as f:
            p.dump(self.h_graph, f)

        with open(path + 'J.p', 'wb') as f:
            p.dump(self.J_graph, f)

    def load(self, string):
        with open('./models/%s.p' % string, 'rb') as f:
            x = p.load(f)
        return x
