import gym
from normalized_env import NormalizedEnv
from utils import *
import numpy as np
import random
import tensorflow as tf
import time
import os
import logging
from space_conversion import SpaceConversionEnv
import tempfile
import sys
import argparse
import scipy.optimize

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-e', '--env-id', type=str, default="Reacher-v1",
                    help="Environment id")
parser.add_argument('-tpb', '--timesteps-per-batch', default=1000, type=int,
                    help="Minibatch size")
parser.add_argument('-mkl', '--max-kl', default=0.01, type=float,
                    help="Maximum value of KL divergence")
parser.add_argument('-cgd', '--cg-damping', default=0.1, type=float,
                    help="Conjugate gradient damping")
parser.add_argument('-g', '--gamma', default=0.99, type=float,
                    help="Discount Factor")
parser.add_argument('-l', '--lam', default=0.97, type=float,
                    help="Lambda value to reduce variance see GAE")

class TRPOAgent(object):

    def __init__(self, env, args):
        self.env = env
        self.config = config = args
        self.config.max_pathlength = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        print("Observation Space", env.observation_space)
        print("Action Space", env.action_space)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True # don't take full gpu memory
        self.session = tf.Session(config=config)
        self.end_count = 0
        self.train = True
        self.hidden_dim = hidden_dim = env.action_space.shape[0]
        self.obs = obs = tf.placeholder(
            dtype, shape=[
                None, 2 * env.observation_space.shape[0] + env.action_space.shape[0]], name="obs")
        self.prev_obs = np.zeros((1, env.observation_space.shape[0]))
        self.prev_action = np.zeros((1, env.action_space.shape[0]))
        self.hidden = hidden = tf.placeholder(dtype, shape=[None, hidden_dim], name="hidden")
        self.advant = advant = tf.placeholder(dtype, shape=[None], name="advant")
        self.oldhidden_dist = oldhidden_dist = tf.placeholder(dtype, shape=[None, hidden_dim*2], name="oldhidden_dist")

        # Create policy neural network
        hidden_dist_n = create_policy_net(self.obs, [100,50,25,hidden_dim])
        # Create action neural network
        self.action_w = action_w = tf.get_variable("action/w", [hidden_dim, env.action_space.shape[0]])
        self.action_b = action_b = tf.get_variable("action/b", [env.action_space.shape[0]], initializer=tf.constant_initializer(0))
        self.var_list_action = var_list_action = [self.action_w, self.action_b]

        action_determ = tf.matmul(tf.stop_gradient(get_moments(hidden_dist_n, hidden_dim)[0]), action_w) + action_b
        self.hidden_ph = hidden_ph = tf.placeholder(dtype, shape=[None, hidden_dim], name="hidden_ph")
        self.action_ph = tf.matmul(hidden_ph, action_w) + action_b
        self.action = action = tf.placeholder(dtype, shape=[None, env.action_space.shape[0]], name="action")
        self.action_net_dist = action_net_dist = -0.5 * tf.reduce_sum(tf.square((action_determ-action)),reduction_indices=-1)
        self.action_net_loss = action_net_loss = -tf.reduce_mean(action_net_dist * advant)
        self.action_net_grad = flatgrad(action_net_loss, var_list_action)
        self.action_gf = GetFlat(self.session, var_list_action)
        self.action_sff = SetFromFlat(self.session, var_list_action)

        """
        action_net_grads = tf.gradients(action_net_loss, [self.action_w, self.action_b])
        action_net_grads_and_vars = list(zip(action_net_grads, [self.action_w, self.action_b]))
        self.action_net_train_op = tf.train.AdamOptimizer(1e-3).apply_gradients(action_net_grads_and_vars)
        """

        eps = 1e-6
        self.hidden_dist_n = hidden_dist_n
        N = tf.shape(obs)[0]
        Nf = tf.cast(N, dtype)
        logp_n = loglik(hidden, hidden_dist_n, hidden_dim)
        oldlogp_n = loglik(hidden, oldhidden_dist, hidden_dim)
        ratio_n = tf.exp(logp_n - oldlogp_n) # Importance sampling ratio
        surr = -tf.reduce_mean(ratio_n * advant)  # Surrogate loss
        var_list_all = tf.trainable_variables() # remove action net vars
        var_list = []
        for var in var_list_all:
            if not "action" in str(var.name):
                var_list.append(var)
        kl = tf.reduce_mean(kl_div(oldhidden_dist, hidden_dist_n, hidden_dim))
        ent = tf.reduce_mean(entropy(hidden_dist_n, hidden_dim))

        self.losses = [surr, kl, ent]
        self.pg = flatgrad(surr, var_list)
        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = tf.reduce_mean(kl_div(tf.stop_gradient(
            hidden_dist_n), hidden_dist_n, hidden_dim))
        grads = tf.gradients(kl_firstfixed, var_list)
        self.flat_tangent = tf.placeholder(dtype, shape=[None])
        shapes = map(var_shape, var_list)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        self.fvp = flatgrad(gvp, var_list)
        self.gf = GetFlat(self.session, var_list)
        self.sff = SetFromFlat(self.session, var_list)
        self.vf = VF(self.session)
        self.session.run(tf.global_variables_initializer())

    def update_action_net(self, feed_action):
        thprev = self.action_gf()
        def lossandgrad(th):
            self.action_sff(th)
            l,g = self.session.run([self.action_net_loss, self.action_net_grad], feed_action)
            g = g.astype('float64')
            l = -l
            return (l,g)
        theta, _, opt_info = scipy.optimize.fmin_l_bfgs_b(lossandgrad, thprev, maxiter=25)
        self.action_sff(theta)
        new_loss = self.session.run(self.action_net_loss, feed_action)
        return np.asarray([new_loss])

    def act(self, obs, *args):
        obs = np.expand_dims(obs, 0)
        obs_new = np.concatenate([obs, self.prev_obs, self.prev_action], 1)

        hidden_dist_n = self.session.run(self.hidden_dist_n, {self.obs: obs_new})

        if self.train:
            hidden = np.float32(gaussian_sample(hidden_dist_n, self.hidden_dim))
        else:
            hidden = np.float32(deterministic_sample(hidden_dist_n, self.hidden_dim))

        action = self.session.run(self.action_ph, {self.hidden_ph: np.expand_dims(np.copy(hidden),0)})[0]
        action = np.float32(action)

        self.prev_action = np.expand_dims(np.copy(action),0)
        self.prev_obs = obs
        return action, hidden, hidden_dist_n, np.squeeze(obs_new)

    def learn(self):
        config = self.config
        start_time = time.time()
        numeptotal = 0
        i = 0
        while True:
            # Generating paths.
            print("Rollout")
            paths = rollout(
                self.env,
                self,
                config.max_pathlength,
                config.timesteps_per_batch)

            # Computing returns and estimating advantage function.
            for path in paths:
                b = path["baseline"] = self.vf.predict(path)
                path["returns"] = discount(path["rewards"], config.gamma)
                b1 = np.append(b, 0 if path["terminated"] else b[-1])
                deltas = path["rewards"] + config.gamma*b1[1:] - b1[:-1]
                path["advant"] = discount(deltas, config.gamma * config.lam)
                path["advant"] = path["returns"] - path["baseline"]

            # Updating policy.
            hidden_dist_n = np.concatenate([path["hidden_dists"] for path in paths])
            obs_n = np.concatenate([path["obs"] for path in paths])
            action_n = np.concatenate([path["actions"] for path in paths])
            hidden_n = np.concatenate([path["hiddens"] for path in paths])
            baseline_n = np.concatenate([path["baseline"] for path in paths])
            returns_n = np.concatenate([path["returns"] for path in paths])

            # Standardize the advantage function to have mean=0 and std=1.
            advant_n = np.concatenate([path["advant"] for path in paths])
            advant_n -= advant_n.mean()

            # Computing baseline function for next iter.

            advant_n /= (advant_n.std() + 1e-8)

            feed = {self.obs: obs_n,
                    self.hidden: hidden_n,
                self.advant: advant_n,
                    self.oldhidden_dist: hidden_dist_n}

            feed_action = {self.obs: obs_n,
                    self.action: action_n,
                    self.advant: advant_n}

            episoderewards = np.array(
                [path["rewards"].sum() for path in paths])

            print "\n********** Iteration %i ************" % i
            if episoderewards.mean() > 1.1 * self.env.spec.reward_threshold:
                self.train = False
            if not self.train:
                print("Episode mean: %f" % episoderewards.mean())
                self.end_count += 1
                if self.end_count > 100:
                    break
            if self.train:
                # train value network
                self.vf.fit(paths)
                # train action network
                """
                for _ in range(4):
                    _, action_net_loss = self.session.run([self.action_net_train_op, self.action_net_dist], feed_action)
                """
                action_net_loss = self.update_action_net(feed_action)
                # train policy network
                thprev = self.gf()

                def fisher_vector_product(p):
                    feed[self.flat_tangent] = p
                    return self.session.run(self.fvp, feed) + config.cg_damping * p

                g = self.session.run(self.pg, feed_dict=feed)
                stepdir = conjugate_gradient(fisher_vector_product, -g)
                shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / config.max_kl)
                fullstep = stepdir / lm
                neggdotstepdir = -g.dot(stepdir)

                def loss(th):
                    self.sff(th)
                    return self.session.run(self.losses[0], feed_dict=feed)
                theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
                self.sff(theta)

                surrafter, kloldnew, entropy = self.session.run(
                    self.losses, feed_dict=feed)
                if kloldnew > 2.0 * config.max_kl:
                    self.sff(thprev)

                stats = {}

                numeptotal += len(episoderewards)
                stats["Total number of episodes"] = numeptotal
                stats["Average sum of rewards per episode"] = episoderewards.mean()
                stats["Entropy"] = entropy
                exp = explained_variance(np.array(baseline_n), np.array(returns_n))
                stats["Baseline explained"] = exp
                stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
                stats["KL between old and new distribution"] = kloldnew
                stats["Surrogate loss"] = surrafter
                stats["Action Net loss"] = action_net_loss.mean()
                for k, v in stats.iteritems():
                    print(k + ": " + " " * (40 - len(k)) + str(v))
                if entropy != entropy:
                    exit(-1)
                """
                if exp > 0.8:
                    self.train = False
                """
            i += 1

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    args = parser.parse_args()
    env = gym.make(args.env_id)
    env = NormalizedEnv(env, normalize_obs=True)

    agent = TRPOAgent(env, args)
    agent.learn()
