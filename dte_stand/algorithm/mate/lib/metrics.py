import numpy as np
import tensorflow as tf
# import gin.tf
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering
from random import random
import time

L1_METRIC_WEIGHTS_COEF = 0.05
# L1_METRIC_PREVIOUS_COEF = 0.5
L2_METRIC_WEIGHTS_COEF = 0.05
# L2_METRIC_PREVIOUS_COEF = 0.5

ACT_VEC_UPDATE = 0
ACT_VEC_NEW_UPDATE = 1
ACT_VEC_USE = 2
ACT_VEC_IGNORE = 3

def tf_l1_metric(tens):
    return tf.norm(tens, ord=1, axis=1)

def tf_l2_metric(tens):
    return tf.norm(tens, ord=2, axis=1)

def tf_coef_l1_metric(tens):
    return tf.reduce_sum(tf.norm(tens, ord=1, axis=1) * tf.constant([1, L1_METRIC_WEIGHTS_COEF]))

def tf_coef_l2_metric(tens):
    '''
    This metric is uses coefficents to put an emphasis on certain tensor values.
    '''
    return tf.reduce_sum(tf.norm(tens, ord=2, axis=1) * tf.constant([1, L2_METRIC_WEIGHTS_COEF]))


def tf_scprint(tens):
    try:
        tf.print(tf.size(tens))
    except Exception as E:
        pass
    tf.print(tens)
    return None


# @gin.configurable
class StatesMemory(object):
    def __init__(self, max_size, n_agents, state_size, n_actions,
                 threshold=0.05, clustering='MiniBatchKMeans', metric='l2'):
        self.mem_vec = np.empty([0, n_agents, state_size])
        self.mem_vec_new = np.empty([0, n_agents, state_size])
        self.act_vec = np.empty([0, n_agents, 16]) # n_actions
        self.act_vec_new = np.empty([0, n_agents, 16]) # n_actions

        self.max_size = max_size # Better to set 2**n
        self.max_new_size = int(self.max_size / 4)
        if self.max_new_size < 32:
            self.max_new_size = 2 * self.max_new_size
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_size = state_size

        # self.cur_size = 0
        self.clustered = False
        self.threshold = threshold # TODO: This should be less than minimal cluster distance
        self.experience_gamma = 1.00 # Probability of using mem_vec
        self.clustering = clustering.lower().strip()
        self.metric = metric.lower().strip()
        if self.clustering == 'minibatchkmeans' and self.metric != 'l2':
            print("WARNING: metric will be set to l2, as MiniBatchKMeans doesn't support other metrics")
            self.metric = 'l2'
        if self.clustering == 'agglomerativeclustering': # synonym for 'agglomerative'
            self.clustering = 'agglomerative'
        if self.metric == 'l1':
            self.calc_proximities = lambda A, x: np.linalg.norm(A - x, axis=-1, ord=1)
        elif self.metric == 'l2':
            self.calc_proximities = lambda A, x: np.linalg.norm(A - x, axis=-1, ord=2)
        elif self.metric == 'cosine':
            self.calc_proximities = lambda A, x: 1 - np.sum(A * x, axis=-1) / (
                np.linalg.norm(A, axis=-1) * np.linalg.norm(x, axis=-1)
                )
        else:
            raise ValueError(f"invalid metric: {self.metric}")
        print(self.threshold, self.metric, self.clustering)


    def init_cluster(self, **kwargs):
        if self.clustering == "minibatchkmeans":
            self.cluster = [MiniBatchKMeans(init='random', max_iter=100, max_no_improvement=10,
                reassignment_ratio=0.0, **kwargs)] * self.n_agents
        elif self.clustering == "agglomerative":
            self.cluster = [AgglomerativeClustering(metric=self.metric,
                linkage='average', **kwargs)] * self.n_agents


    def defragmentate(self, pred, n_agent):
        _, idx = np.unique(pred[::-1], return_index=True, axis=0)
        idx = len(pred) - 1 - idx

        '''SAVE SOME RANDOM VALUES'''
        tmp = np.arange(len(pred) - 1)[~np.isin(np.arange(len(pred) - 1), idx)]
        idx = np.append(idx, np.random.choice(tmp, size=(self.max_size - len(idx)), replace=False), axis=0)

        '''SAVE LAST VALUES OF ARRAY'''
        # i = len(pred) - 1 # Can be 0
        # while (len(idx) != self.max_size):
        #     if (i not in idx):
        #         idx = np.append(idx, [i], axis=0)
        #     i -= 1 # Can be +1

        not_to = idx[idx < self.max_size]
        # tmp = np.arange(self.max_size)
        # not_to = tmp[np.isin(tmp, idx)]
        tmp = np.arange(self.max_size, self.max_size + self.max_new_size)
        fr = tmp[np.isin(tmp, idx)]

        '''LOG MEMORY SIZES'''
        # print("(", n_agent, len(np.unique(pred)), end=" ) ")

        '''LOG MEMORY UPDATES'''
        # print(len(np.unique(pred)), pred, idx, to, fr)

        self.mem_vec[:len(not_to), n_agent, :] = self.mem_vec[not_to, n_agent, :]
        self.act_vec[:len(not_to), n_agent, :] = self.act_vec[not_to, n_agent, :]
        self.mem_vec[len(not_to):, n_agent, :] = self.mem_vec_new[fr - self.max_size, n_agent, :]
        self.act_vec[len(not_to):, n_agent, :] = self.act_vec_new[fr - self.max_size, n_agent, :]


    def update_memory(self, new_states):
        '''
        '''
        if self.experience_gamma == 0.0:
            return ACT_VEC_IGNORE, None

        if self.mem_vec.shape[0] > 0:
            proximities = self.calc_proximities(self.mem_vec, new_states)
            argmins = np.argmin(proximities, axis=0)
            mins = proximities[argmins, np.arange(self.n_agents)]
            if (random() < self.experience_gamma) and (np.sum(mins < self.threshold) == self.n_agents): # found close item, Action of closest obj
                return ACT_VEC_USE, self.act_vec[argmins, np.arange(self.mem_vec.shape[1]), :]

        if self.mem_vec.shape[0] < self.max_size:
            self.mem_vec = np.append(self.mem_vec, [new_states], axis=0) # self.mem_vec.write(self.cur_size, new_states)
            return ACT_VEC_UPDATE, None # HAS TO WRITE ACTION AFTER ACTIONS

        if self.mem_vec_new.shape[0] < self.max_new_size:
            self.mem_vec_new = np.append(self.mem_vec_new, [new_states], axis=0)
            return ACT_VEC_NEW_UPDATE, None # HAS TO WRITE ACTION AFTER ACTIONS TO ADDITIONAL

        print("defragmentating memory...")
        if self.clustered == False:
            self.init_cluster(n_clusters=self.max_size)
            self.clustered = True
            for agent in range(self.n_agents):
                nodes = np.append(self.mem_vec, self.mem_vec_new, axis=0)[:, agent, :]
                pred = self.cluster[agent].fit_predict(nodes)
                self.defragmentate(pred, agent)
                self.init_cluster(n_clusters=self.max_size)
                # self.cluster[agent].fit(self.mem_vec[:, agent, :]) # Maybe not needed
        else:
            # t_start = time.time()
            for agent in range(self.n_agents):
                nodes = np.append(self.mem_vec, self.mem_vec_new, axis=0)[:, agent, :]
                if self.clustering != 'minibatchkmeans':
                    pred = self.cluster[agent].fit_predict(nodes)
                else:
                    self.cluster[agent].partial_fit(nodes) #.partial_fit(self.mem_vec_new[:, agent, :])
                    pred = self.cluster[agent].predict(nodes)
                self.defragmentate(pred, agent)
                self.init_cluster(n_clusters=self.max_size)
                # self.cluster[agent].fit(self.mem_vec[:, agent, :]) # Maybe not needed
            # t_end = time.time()
            # print(f"Memory update algorithm took {t_end - t_start} sec.")

        del self.mem_vec_new
        del self.act_vec_new
        self.mem_vec_new = np.asarray([new_states])
        self.act_vec_new = np.empty([0, self.n_agents, 16]) # self.n_agents
        return ACT_VEC_NEW_UPDATE, None # HAS TO WRITE ACTION AFTER ACTIONS TO ADDITIONAL


    def update_action(self, new_acts):
        self.act_vec = np.append(self.act_vec, [new_acts], axis=0)


    def update_action_new(self, new_acts):
        self.act_vec_new = np.append(self.act_vec_new, [new_acts], axis=0)
