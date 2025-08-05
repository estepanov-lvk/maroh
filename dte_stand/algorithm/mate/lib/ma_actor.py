# import gin.tf
import typing as t
import numpy as np
from pydantic import BaseModel
import tensorflow as tf
from tensorflow import keras
from dte_stand.algorithm.mate.lib.metrics import *
from dte_stand.algorithm.mate.lib.gat import GAT, GATLayer
from dte_stand.algorithm.mate.utils.tf_compile import maybe_compile

from time import sleep
from memory_profiler import profile

class MaActorCfg(BaseModel):
    use_memory: bool = False
    # number of saved states
    memory_size: int = 512
    threshold: float = 0.05
    clustering: str = 'MiniBatchKMeans' # 'MiniBatchKMeans', 'Agglomerative'
    metric: str = 'l2' # ['l2'] for MiniBatchKMeans, ['l2', 'l1', 'cosine'] for Agglomerative

    use_gat: bool = False
    # number of attention heads
    gat_num_heads: int = 20

# @gin.configurable
class MaActor(keras.Model):
    def __init__(self,
                 cfg: t.Optional[dict],
                 graph,
                 adj_matrix,
                 num_actions=1,
                 num_features=2,
                 link_state_size=16,
                 aggregation='min_max',
                 first_hidden_layer_size=128,
                 dropout_rate=0.15,
                 final_hidden_layer_size=64,
                 message_iterations=8,
                 activation_fn='tanh',
                 final_activation_fn='linear'):

        super(MaActor, self).__init__()

        # actor-specific configuration
        self.cfg = MaActorCfg(**cfg) if cfg is not None else MaActorCfg()

        # HYPERPARAMETERS
        self.num_actions = num_actions
        self.num_features = num_features
        self.n_links = graph.number_of_edges()
        self.link_state_size = link_state_size
        self.message_hidden_layer_size = final_hidden_layer_size
        self.aggregation = aggregation
        self.message_iterations = message_iterations

        # FIXED INPUTS
        self.incoming_links = graph.nodes()['graph_data']['incoming_links']
        self.outcoming_links = graph.nodes()['graph_data']['outcoming_links']

        # GAT
        self.gat_used = self.cfg.use_gat
        self.num_heads = self.cfg.gat_num_heads
        self.adj_matrix = adj_matrix

        # MEMORY
        self.memory_used = self.cfg.use_memory
        if self.memory_used:
            self.memory_size = self.cfg.memory_size
            self.memory_threshold = self.cfg.threshold
            self.memory_clustering = self.cfg.clustering
            self.memory_metric = self.cfg.metric
            kwargs = dict(threshold=self.memory_threshold,
                          clustering=self.memory_clustering,
                          metric=self.memory_metric)
            self.mind = [StatesMemory(self.memory_size, graph.number_of_edges(), 2, num_actions, **kwargs)]
            if self.gat_used:
                self.mind += [StatesMemory(self.memory_size, graph.number_of_edges(),
                    first_hidden_layer_size * self.num_heads,
                    num_actions, **kwargs) for _ in range(self.message_iterations - 1)]
            else:
                self.mind += [StatesMemory(self.memory_size, graph.number_of_edges(), link_state_size,
                    num_actions, **kwargs) for _ in range(self.message_iterations - 1)]

        # NEURAL NETWORKS
        self.hidden_layer_initializer = tf.keras.initializers.Orthogonal(gain=np.sqrt(2))
        self.final_layer_initializer = tf.keras.initializers.Orthogonal(gain=0.01)
        self.kernel_regularizer = None
        self.activation_fn = activation_fn
        self.final_hidden_layer_size = final_hidden_layer_size
        self.first_hidden_layer_size = first_hidden_layer_size
        self.dropout_rate = dropout_rate
        self.final_activation_fn = final_activation_fn
        self.define_network()

        # MEMORY LOG
        self.log_message_iterations_done = 0
        self.log_message_iterations_possible = 0
        self.log_message_iterations_done_infer = 0
        self.log_message_iterations_possible_infer = 0
        self.log_message_iterations_done_train = 0
        self.log_message_iterations_possible_train = 0

        # Indicates if model can be called inside tf.function
        self.can_compile = not self.memory_used

    def define_network(self):
        self.create_message = None

        if self.gat_used:
            if self.memory_used:
                self.link_update = []
                for _ in (range(max(1, self.message_iterations - 1))):
                    self.link_update.append(GATLayer(self.first_hidden_layer_size,
                                                self.num_heads,
                                                self.adj_matrix,
                                                kernel_initializer=self.hidden_layer_initializer,
                                                activation='relu',
                                                dropout_rate=self.dropout_rate))
                self.link_update.append(GATLayer(self.link_state_size,
                                            1,
                                            self.adj_matrix,
                                            kernel_initializer=self.hidden_layer_initializer,
                                            activation='relu',
                                            dropout_rate=self.dropout_rate))
            else:
                # self.link_update = GAT(self.final_hidden_layer_size,
                #                        self.n_links * self.num_features, self.num_heads, self.message_iterations,
                #                        kernel_initializer=self.hidden_layer_initializer)
                self.link_update = keras.models.Sequential(name='link_update')
                for _ in (range(max(1, self.message_iterations - 1))):
                    self.link_update.add(GATLayer(self.first_hidden_layer_size,
                                                self.num_heads,
                                                self.adj_matrix,
                                                kernel_initializer=self.hidden_layer_initializer,
                                                activation='relu',
                                                dropout_rate=self.dropout_rate))
                self.link_update.add(GATLayer(self.link_state_size,
                                            1,
                                            self.adj_matrix,
                                            kernel_initializer=self.hidden_layer_initializer,
                                            activation='relu',
                                            dropout_rate=self.dropout_rate))
        else:
            # message
            self.create_message = keras.models.Sequential(name='create_message')
            self.create_message.add(keras.layers.Dense(self.message_hidden_layer_size,
                                                    kernel_initializer=self.hidden_layer_initializer,
                                                    activation=self.activation_fn))
            # self.create_message.add(keras.layers.Dense(32,
            #                                            kernel_initializer=self.hidden_layer_initializer,
            #                                            activation=self.activation_fn))
            self.create_message.add(keras.layers.Dense(self.link_state_size,
                                                    kernel_initializer=self.hidden_layer_initializer,
                                                    activation=self.activation_fn))

            # link update
            self.link_update = keras.models.Sequential(name='link_update')
            self.link_update.add(keras.layers.Dense(self.first_hidden_layer_size,
                                                    kernel_initializer=self.hidden_layer_initializer,
                                                    activation=self.activation_fn))
            self.link_update.add(keras.layers.Dense(self.final_hidden_layer_size,
                                                    kernel_initializer=self.hidden_layer_initializer,
                                                    activation=self.activation_fn))
            # self.link_update.add(keras.layers.Dense(32,
            #                                         kernel_initializer=self.hidden_layer_initializer,
            #                                         activation=self.activation_fn))
            self.link_update.add(keras.layers.Dense(self.link_state_size,
                                                    kernel_initializer=self.hidden_layer_initializer,
                                                    activation=self.activation_fn))
        self.readout = keras.models.Sequential(name='readout')
        self.readout.add(
            keras.layers.Dense(self.first_hidden_layer_size, kernel_initializer=self.hidden_layer_initializer,
                               kernel_regularizer=self.kernel_regularizer, activation=self.activation_fn))
        self.readout.add(keras.layers.Dropout(self.dropout_rate))
        self.readout.add(
            keras.layers.Dense(self.final_hidden_layer_size, kernel_initializer=self.hidden_layer_initializer,
                               kernel_regularizer=self.kernel_regularizer, activation=self.activation_fn))
        self.readout.add(keras.layers.Dropout(self.dropout_rate))
        # self.readout.add(
        #     keras.layers.Dense(32, kernel_initializer=self.hidden_layer_initializer,
        #                        kernel_regularizer=self.kernel_regularizer, activation=self.activation_fn))
        # self.readout.add(keras.layers.Dropout(self.dropout_rate))
        self.readout.add(keras.layers.Dense(self.num_actions, kernel_initializer=self.final_layer_initializer,
                                            kernel_regularizer=self.kernel_regularizer,
                                            activation=self.final_activation_fn))

    def build(self, input_shape=None):
        if self.gat_used:
            if self.memory_used:
                # pass
                for i in range(len(self.link_update)):
                    self.link_update[i].build(input_shape=[None, self.link_state_size])
            else:
                self.link_update.build(input_shape=[None, self.link_state_size])
        else:
            self.create_message.build(input_shape=[None, 2 * self.link_state_size])
            if self.aggregation == 'sum':
                self.link_update.build(input_shape=[None, 2 * self.link_state_size])
            elif self.aggregation == 'min_max':
                self.link_update.build(input_shape=[None, 3 * self.link_state_size])
        self.readout.build(input_shape=[None, self.link_state_size])
        self.built = True

    @maybe_compile('can_compile')
    def message_passing(self, input, recall=False, recall_n_iter=None):
        input_tensor = tf.convert_to_tensor(input)
        link_states = tf.reshape(input_tensor, [self.num_features, self.n_links])
        link_states = tf.transpose(link_states)
        padding = [[0, 0], [0, self.link_state_size - self.num_features]] # TODO Идея: Добавлять здесь еще одно значимое значение вместо части падинга, чтобы указывать, увеличилась или уменьшилась скорость по сравнению с предыдушим стейтом
        link_states = tf.pad(link_states, padding)
        codes = []
        # print("PPP\n", self.adj_matrix)
        if self.gat_used:
            if self.memory_used:
                for message_iteration in range(self.message_iterations):
                    # print(link_states.shape)
                    if message_iteration == 0:
                        link_states_cut = link_states.numpy().astype(float)[:, :2]
                    else:
                        link_states_cut = link_states.numpy().astype(float)
                    code, policy_memmory = self.mind[message_iteration].update_memory(link_states_cut)
                    codes.append(code)
                    if code == ACT_VEC_USE:
                        return codes, policy_memmory
                    link_states = self.link_update[message_iteration](link_states)
            else:
                link_states = self.link_update(link_states)
                # print("PPP", link_states.shape)
        else:
            for message_iteration in range(self.message_iterations):
                # tf_scprint(link_states)
                if self.memory_used:
                    if not recall or message_iteration >= recall_n_iter:
                        if message_iteration == 0:
                            link_states_cut = link_states.numpy().astype(float)[:, :2]
                        else:
                            link_states_cut = link_states.numpy().astype(float)
                        code, policy_memmory = self.mind[message_iteration].update_memory(link_states_cut)
                        codes.append(code)
                        if code == ACT_VEC_USE:
                            return codes, policy_memmory
                    else:
                        # recall and message_iteration < recall_n_iter
                        codes.append(ACT_VEC_IGNORE)

                incoming_link_states = tf.gather(link_states, self.incoming_links) # PROFILER MEMORY LEAK (sometimes by 0.3 MiB)
                outcoming_link_states = tf.gather(link_states, self.outcoming_links)
                # m(.) fun
                message_inputs = tf.cast(tf.concat([incoming_link_states, outcoming_link_states], axis=1), tf.float32)
                messages = self.create_message(message_inputs) # PROFILER MEMORY LEAK (sometimes by 0.3 MiB)
                # a(.) fun
                aggregated_messages = self.message_aggregation(messages)
                # print("1IS", tf.shape(aggregated_messages))
                # u(.) fun
                link_update_input = tf.cast(tf.concat([link_states, aggregated_messages], axis=1), tf.float32)
                link_states = self.link_update(link_update_input) # PROFILER MEMORY LEAK (sometimes by 0.3 MiB)
        # print(tf.shape(link_states))
        # for _ in range(1000000000000000000):
        #     1
        return codes, link_states

    @maybe_compile('can_compile')
    def message_aggregation(self, messages):
        if self.aggregation == 'sum':
            aggregated_messages = tf.math.unsorted_segment_sum(messages, self.outcoming_links,
                                                               num_segments=self.n_links)
        elif self.aggregation == 'min_max':
            agg_max = tf.math.unsorted_segment_max(messages, self.outcoming_links, num_segments=self.n_links)
            agg_min = tf.math.unsorted_segment_min(messages, self.outcoming_links, num_segments=self.n_links)
            aggregated_messages = tf.concat([agg_max, agg_min], axis=1)
        return aggregated_messages

    @maybe_compile('can_compile')
    def call(self, input, recall=False, recall_n_iter=None):
        if not recall:
            codes, link_states = self.message_passing(input)
        else:
            codes, link_states = self.message_passing(input, recall=recall,
                                            recall_n_iter=recall_n_iter)

        # print("\ncodes", codes, end="")
        if self.memory_used:
            if codes[-1] == ACT_VEC_USE:
                policy = self.readout(link_states) # policy = link_states.astype('float32')
                codes = codes[:-1]
            else:
                policy = self.readout(link_states)
            self.log_message_iterations_done += len(codes) # Suppose, when code == ACT_VEC_USE, no message passing iteration is done
            self.log_message_iterations_possible += self.message_iterations
            if not recall:
                self.log_message_iterations_done_infer += len(codes)
                self.log_message_iterations_possible_infer += self.message_iterations
            else:
                self.log_message_iterations_done_train += len(codes)
                self.log_message_iterations_possible_train += self.message_iterations
            for idx, code in enumerate(codes):
                if code == ACT_VEC_UPDATE:
                    self.mind[idx].update_action(link_states) # self.mind[idx].update_action(policy)
                elif code == ACT_VEC_NEW_UPDATE:
                    self.mind[idx].update_action_new(link_states) # self.mind[idx].update_action_new(policy)
        else:
            policy = self.readout(link_states)
        # print("policy", policy)
        policy = tf.reshape(policy, [-1])
        return policy, len(codes)
