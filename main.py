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
parser.add_argument('-s', '--seed', default=1, type=int,
                    help="Seed")
parser.add_argument('--log-dir', default="/tmp/trpo/", type=str,
                    help="Folder to save")

class TRPOAgent(object):

    def __init__(self, env, args):
        self.env = env
        self.config = config = args
        self.config.max_pathlength = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps') or 1000
        # name of folder
        folder_name = "trpo"
        for key,value in vars(self.config).iteritems():
            if key != 'log_dir':
                folder_name = folder_name + "_{}_{}".format(key, value)
            print key, value
        self.config.log_dir = os.path.join(self.config.log_dir, self.config.env_id.split('-')[0], folder_name)
        print("Observation Space", env.observation_space)
        print("Action Space", env.action_space)
        config_tf = tf.ConfigProto()
        config_tf.gpu_options.allow_growth=True # don't take full gpu memory
        self.session = tf.Session(config=config_tf)
        self.end_count = 0
        self.train = True
        self.obs = obs = tf.placeholder(
            dtype, shape=[
                None, 2 * env.observation_space.shape[0] + env.action_space.shape[0]], name="obs")
        self.prev_obs = np.zeros((1, env.observation_space.shape[0]))
        self.prev_action = np.zeros((1, env.action_space.shape[0]))
        self.action = action = tf.placeholder(dtype, shape=[None, env.action_space.shape[0]], name="action")
        self.advant = advant = tf.placeholder(dtype, shape=[None], name="advant")
        self.oldaction_dist = oldaction_dist = tf.placeholder(dtype, shape=[None, env.action_space.shape[0]*2], name="oldaction_dist")

        # Create neural network
        action_dist_n = create_policy_net(self.obs, [100,50,25], [True, True, True], env.action_space.shape[0])

        eps = 1e-6
        self.action_dist_n = action_dist_n
        N = tf.shape(obs)[0]
        Nf = tf.cast(N, dtype)
        logp_n = loglik(action, action_dist_n, env.action_space.shape[0])
        oldlogp_n = loglik(action, oldaction_dist, env.action_space.shape[0])
        ratio_n = tf.exp(logp_n - oldlogp_n) # Importance sampling ratio
        #surr = -tf.reduce_mean(ratio_n * advant)  # Surrogate loss
        surr = -tf.reduce_mean(logp_n * advant) # Non-importance sampling based surr loss
        var_list = tf.trainable_variables()
        kl = tf.reduce_mean(kl_div(oldaction_dist, action_dist_n, env.action_space.shape[0]))
        ent = tf.reduce_mean(entropy(action_dist_n, env.action_space.shape[0]))

        self.losses = [surr, kl, ent]
        self.pg = flatgrad(surr, var_list)
        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = tf.reduce_mean(kl_div(tf.stop_gradient(
            action_dist_n), action_dist_n, env.action_space.shape[0]))
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
        self.summary_writer = tf.summary.FileWriter(self.config.log_dir)

    def act(self, obs, *args):
        obs = np.expand_dims(obs, 0)
        obs_new = np.concatenate([obs, self.prev_obs, self.prev_action], 1)

        action_dist_n = self.session.run(self.action_dist_n, {self.obs: obs_new})

        # TODO FIX
        if self.train:
            action = np.float32(gaussian_sample(action_dist_n, self.env.action_space.shape[0]))
        else:
            action = np.float32(deterministic_sample(action_dist_n, self.env.action_space.shape[0]))

        self.prev_action = np.expand_dims(np.copy(action),0)
        self.prev_obs = obs
        return action, action_dist_n, np.squeeze(obs_new)

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
            action_dist_n = np.concatenate([path["action_dists"] for path in paths])
            obs_n = np.concatenate([path["obs"] for path in paths])
            action_n = np.concatenate([path["actions"] for path in paths])
            baseline_n = np.concatenate([path["baseline"] for path in paths])
            returns_n = np.concatenate([path["returns"] for path in paths])

            # Standardize the advantage function to have mean=0 and std=1.
            advant_n = np.concatenate([path["advant"] for path in paths])
            advant_n -= advant_n.mean()

            # Computing baseline function for next iter.

            advant_n /= (advant_n.std() + 1e-8)

            feed = {self.obs: obs_n,
                    self.action: action_n,
                self.advant: advant_n,
                    self.oldaction_dist: action_dist_n}


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
                self.vf.fit(paths)
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
                summary = tf.Summary()
                for k, v in stats.iteritems():
                    print(k + ": " + " " * (40 - len(k)) + str(v))
                    if k != "Time elapsed":
                        summary.value.add(tag=k, simple_value=float(v))
                # save stats
                self.summary_writer.add_summary(summary, i)
                self.summary_writer.flush()
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    env = gym.make(args.env_id)
    env = NormalizedEnv(env, normalize_obs=True)

    agent = TRPOAgent(env, args)
    agent.learn()
