import numpy as np
import tensorflow as tf
import random
import scipy.signal
import sys

dtype = tf.float32

def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def rollout(env, agent, max_pathlength, n_timesteps):
    paths = []
    timesteps_sofar = 0
    while timesteps_sofar < n_timesteps:
        obs, actions, rewards, hiddens, hidden_dists = [], [], [], [], []
        ob = env.reset()
        agent.prev_action *= 0.0
        agent.prev_obs *= 0.0
        terminated = False
        for _ in xrange(max_pathlength):
            action, hidden, hidden_dist, ob = agent.act(ob)
            obs.append(ob)
            actions.append(action)
            hiddens.append(hidden)
            hidden_dists.append(hidden_dist)
            #env.render()
            res = env.step(action)
            ob = res[0]
            rewards.append(res[1])
            if res[2]:
                terminated = True
                break

        path = {"obs": np.concatenate(np.expand_dims(obs, 0)),
                "hidden_dists": np.concatenate(hidden_dists), # is it correct ?
                "rewards": np.array(rewards),
                "actions": np.array(actions),
                "hiddens": np.array(hiddens),
                "terminated": terminated,}
        paths.append(path)
        agent.prev_action *= 0.0
        agent.prev_obs *= 0.0
        timesteps_sofar += len(path["rewards"])
    return paths

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class VF(object):
    coeffs = None

    def __init__(self, session):
        self.net = None
        self.session = session

    def create_net(self, shape):
        #print(shape)
        self.x = tf.placeholder(tf.float32, shape=[None, shape], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None], name="y")
        self.net = self.x
        hidden_sizes = [32,32]
        for i in range(len(hidden_sizes)):
            self.net = tf.nn.elu(linear(self.net, hidden_sizes[i], "vf/l{}".format(i), normalized_columns_initializer(0.01)))
        self.net = linear(self.net, 1, "vf/value")
        self.net = tf.reshape(self.net, (-1, ))
        l2 = tf.reduce_mean((self.net - self.y) * (self.net - self.y))
        var_list_all = tf.trainable_variables()
        self.var_list = var_list = []
        for var in var_list_all:
            if "vf" in str(var.name):
                var_list.append(var)
        weight_decay = tf.add_n([1e-3 * tf.nn.l2_loss(var) for var in var_list])
        self.loss = loss = l2+weight_decay
        self.vfg = flatgrad(loss, var_list)
        self.gf = GetFlat(self.session, var_list)
        self.sff = SetFromFlat(self.session, var_list)

        """
        self.train = tf.train.AdamOptimizer().minimize(l2)
        """
        self.session.run(tf.global_variables_initializer())

    def update(self, *args):
        featmat, returns = args[0], args[1]
        thprev = self.gf()
        def lossandgrad(th):
            self.sff(th)
            l,g = self.session.run([self.loss, self.vfg], {self.x: featmat, self.y: returns})
            g = g.astype('float64')
            return (l,g)
        theta, _, opt_info = scipy.optimize.fmin_l_bfgs_b(lossandgrad, thprev, maxiter=25)
        self.sff(theta)

    def _features(self, path):
        o = path["obs"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        hid = path["hidden_dists"].astype('float32')
        act = path["actions"].astype('float32')
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 10.0
        ret = np.concatenate([o, hid, act, al, np.ones((l, 1))], axis=1)
        return ret

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        if self.net is None:
            self.create_net(featmat.shape[1])
        returns = np.concatenate([path["returns"] for path in paths])
        self.update(featmat, returns)
        """
        for _ in range(50):
            self.session.run(self.train, {self.x: featmat, self.y: returns})
        """

    def predict(self, path):
        if self.net is None:
            return np.zeros(len(path["rewards"]))
        else:
            ret = self.session.run(self.net, {self.x: self._features(path)})
            return np.reshape(ret, (ret.shape[0], ))


def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def gaussian_sample(action_dist, action_size):
    return np.random.randn(action_size) * action_dist[0,action_size:] + action_dist[0,:action_size]

def deterministic_sample(action_dist, action_size):
    return action_dist[0,:action_size]

# returns mean and std of gaussian distribution
def get_moments(action_dist, action_size):
    mean = tf.reshape(action_dist[:, :action_size], [tf.shape(action_dist)[0], action_size])
    std = (tf.reshape(action_dist[:, action_size:], [tf.shape(action_dist)[0], action_size]))
    return mean, std

def loglik(action, action_dist, action_size):
    mean, std = get_moments(action_dist, action_size)
    return -0.5 * tf.reduce_sum(tf.square((action-mean) / std),reduction_indices=-1) \
            -0.5 * tf.log(2.0*np.pi)*action_size - tf.reduce_sum(tf.log(std),reduction_indices=-1)

def kl_div(action_dist1, action_dist2, action_size):
    mean1, std1 = get_moments(action_dist1, action_size)
    mean2, std2 = get_moments(action_dist2, action_size)
    numerator = tf.square(mean1 - mean2) + tf.square(std1) - tf.square(std2)
    denominator = 2 * tf.square(std2) + 1e-8
    return tf.reduce_sum(
        numerator/denominator + tf.log(std2) - tf.log(std1),reduction_indices=-1)

def entropy(action_dist, action_size):
    _, std = get_moments(action_dist, action_size)
    return tf.reduce_sum(tf.log(std),reduction_indices=-1) + .5 * np.log(2*np.pi*np.e) * action_size

def create_action_params(latent_dim, action_dim, hidden_sizes=[]):
    hidden_sizes.insert(0, latent_dim)
    hidden_sizes.insert(len(hidden_sizes), action_dim)
    params = []
    for i in range(len(hidden_sizes)-1):
        w = tf.get_variable("action/w{}".format(i), [hidden_sizes[i], hidden_sizes[i+1]])
        b = tf.get_variable("action/b{}".format(i), [hidden_sizes[i+1]], initializer=tf.constant_initializer(0))
        params.append(w)
        params.append(b)
    return params

def feedforward_action_net(x, params, non_linear):
    num_layers = len(non_linear)
    for i in range(num_layers):
        w = params[2*i]
        b = params[2*i + 1]
        x = tf.matmul(x, w) + b
        if non_linear[i]:
            x = tf.nn.tanh(x)
    return x

def create_policy_net(obs, hidden_sizes):
    x = obs
    for i in range(len(hidden_sizes)-1):
        x = tf.nn.tanh(linear(x, hidden_sizes[i], "policy/l{}".format(i), normalized_columns_initializer(0.01)))

    x_mean = linear(x, hidden_sizes[-1], "policy/l{}/mean".format(len(hidden_sizes)-1), normalized_columns_initializer(0.01))
    std_w = tf.Variable(tf.zeros([1,hidden_sizes[-1]]), name="policy/l{}/std".format(len(hidden_sizes)-1))
    x_std = tf.tile(tf.exp(std_w), tf.pack([tf.shape(x_mean)[0],1]))
    x = tf.concat(1, [tf.reshape(x_mean, [-1, hidden_sizes[-1]]), tf.reshape(x_std, [-1, hidden_sizes[-1]])])
    return x

"""
def create_policy_net(obs, hidden_sizes, non_linear, action_size):
    x = obs
    for i in range(len(hidden_sizes)):
        x = linear(x, hidden_sizes[i], "policy/l{}".format(i), normalized_columns_initializer(0.01))
        #x = linear(x, hidden_sizes[i], "policy/l{}".format(i))
        if non_linear[i]:
            x = tf.nn.tanh(x)
    mean = linear(x, action_size, "policy/mean", normalized_columns_initializer(0.01))
    std_w = tf.Variable(tf.zeros([1,action_size]), name="policy/std/w")
    #std_w = tf.get_variable("policy/std/w", [1, action_size], initializer=tf.zeros([1,action_size]))
    std = tf.tile(tf.exp(std_w), tf.pack([tf.shape(mean)[0],1]))
    output = tf.concat(1, [tf.reshape(mean, [-1, action_size]), tf.reshape(std, [-1, action_size])])
    return output
"""

def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))


def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat(0, [tf.reshape(grad, [numel(v)])
                         for (v, grad) in zip(var_list, grads)])


class SetFromFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        assigns = []
        shapes = map(var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(
                tf.assign(
                    v,
                    tf.reshape(
                        theta[
                            start:start +
                            size],
                        shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})


class GetFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat(0, [tf.reshape(v, [numel(v)]) for v in var_list])

    def __call__(self):
        return self.op.eval(session=self.session)

def linesearch(f, x, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x)
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return xnew
    return x


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in xrange(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x

class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def explained_variance(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary
