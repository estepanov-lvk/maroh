import logging
import os

# import gin.tf
import networkx
import tensorflow as tf
import logging
from networkx import diameter as graph_diameter

from dte_stand.algorithm.mate.agents.ppo_agent import PPOAgent
from dte_stand.algorithm.mate.agents.ma_ppo_agent import MaPPOAgent
from dte_stand.algorithm.mate.environment.environment import Environment
from dte_stand.config import Config

LOG = logging.getLogger(__name__)


# @gin.configurable
class Runner(object):
    def __init__(self,
                 topology_object,
                 hash_function,
                 phi_func,
                 algorithm='PPO',
                 reload_model=False,
                 model_dir='checkpoints/training/gravity_1/PPO_agg_period100/clip0.2/gamma0.95/episode',
                 only_eval=False,
                 base_dir='dte_stand/algorithm/mate/logs',
                 checkpoint_dir='dte_stand/algorithm/mate/checkpoints',
                 save_checkpoints=True,
                 multi_actions=False):

        config = Config.config()
        mate_config = config.mate
        try:
            message_iterations = (
                    mate_config.message_iterations if mate_config.message_iterations > 0
                    else graph_diameter(topology_object)
            )
        except networkx.NetworkXError:
            LOG.exception('Message iterations is not given in config '
                          'and topology does not have all links in both directions')
            raise Exception(
                    'Message iterations is not given in config '
                    'and topology does not have all links in both directions.\n'
                    'Either set message_iterations to a positive value in mate section in config '
                    'or the topology must be strongly connected in both directions.')
        mate_actions = mate_config.actions
        self.save_checkpoints = save_checkpoints
        self.hash_function = hash_function
        self.env = Environment(
                topology_object,
                self.hash_function,
                mate_actions,
                phi_func,
                base_reward=mate_config.reward,
                reward_computation=mate_config.reward_computation,
                min_weight=mate_config.min_weight,
                max_weight=mate_config.max_weight
        )

        if multi_actions:
            agent = MaPPOAgent
        else:
            agent = PPOAgent
        if algorithm == 'PPO':
            self.agent = agent(
                    self.env,
                    mate_config.actor_cfg,
                    mate_actions,
                    phi_func,
                    checkpoint_dir=checkpoint_dir,
                    message_iterations=message_iterations,
                    plot_period=config.plot_period,
                    horizon=mate_config.horizons,
                    eval_period=mate_config.episodes,
                    gamma=mate_config.gamma,
                    gae_lambda=mate_config.gae_lambda,
                    lr_actor=mate_config.lr_actor,
                    lr_critic=mate_config.lr_critic,
                    greedy_eplison=mate_config.greedy_epsilon,
                    n_without_update=mate_config.n_without_update,
                    save_checkpoints=save_checkpoints,
            )
        else:
            assert False, 'RL Algorithm %s is not implemented' % algorithm
        self.base_dir = base_dir
        self.checkpoint_base_dir = checkpoint_dir
        self.only_eval = only_eval

        if reload_model or self.only_eval:
            self.agent.load_saved_model(model_dir, only_eval)

        if self.save_checkpoints and (not os.path.exists(self.checkpoint_base_dir)):
            os.makedirs(self.checkpoint_base_dir)

    def update(self, topology, checkpoint_dir, save_model):
        self.save_checkpoints = save_model
        self.agent.save_checkpoints = save_model
        if self.save_checkpoints:
            self.checkpoint_base_dir = checkpoint_dir
            self.agent.set_checkpoint_dir(checkpoint_dir)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
        self.env.update_topology(topology)
        self.agent.change_sample = False

    def run_experiment(self, topology, current_flows):
        hash_weights = self.agent.train_and_evaluate(topology, current_flows, self.hash_function)
        return hash_weights
