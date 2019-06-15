import os

import numpy as np
import tensorflow as tf
import bundle_entropy
import tflearn
import ddpg_nets_dm
from replay_memory import ReplayMemory
from helper import variable_summaries


flags = tf.app.flags
FLAGS = flags.FLAGS


# DDPG Agent
#
class Agent:

    def __init__(self, dimO, dimA):
        self.dimA = dimA[0]
        self.dimO = dimO[0]
        dimA = list(dimA)
        dimO = list(dimO)
        nets = ddpg_nets_dm
        tau = FLAGS.tau
        discount = FLAGS.discount
        pl2norm = FLAGS.pl2norm
        l2norm = FLAGS.l2norm
        plearning_rate = FLAGS.prate
        learning_rate = FLAGS.rate
        outheta = FLAGS.outheta
        ousigma = FLAGS.ousigma
 
        # init replay memory
        self.rm = ReplayMemory(FLAGS.rmsize, dimO, dimA)
        # start tf session
        self.sess = tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=FLAGS.thread,
            log_device_placement=False,
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True)))

        # create tf computational graph
        #
        self.theta_p = nets.theta_p(dimO, dimA, FLAGS.l1size, FLAGS.l2size)
        self.theta_pt, update_pt = exponential_moving_averages(self.theta_p, tau)
        #self.theta_qt, update_qt = exponential_moving_averages(self.theta_q, tau)

        obs = tf.placeholder(tf.float32, [None] + dimO, "obs")
        act_test = nets.policy(obs, self.theta_p)

        # explore
        noise_init = tf.zeros([1] + dimA)
        noise_var = tf.Variable(noise_init)
        self.ou_reset = noise_var.assign(noise_init)
        noise = noise_var.assign_sub((outheta) * noise_var - tf.random_normal(dimA, stddev=ousigma))
        act_expl = act_test + noise

        # training
        
        # q optimization
        act_train = tf.placeholder(tf.float32, [FLAGS.bsize] + dimA, "act_train")
        rew = tf.placeholder(tf.float32, [FLAGS.bsize], "rew")
        obs2 = tf.placeholder(tf.float32, [FLAGS.bsize] + dimO, "obs2")
        term2 = tf.placeholder(tf.bool, [FLAGS.bsize], "term2")
        # policy loss
        act_train = nets.policy(obs, self.theta_p)
        with tf.variable_scope('q'):
            negQ = self.negQ(obs, act_train)
        negQ_entr = negQ - entropy(act_train)
        q = -negQ
        q_entr = -negQ_entr
        meanq = tf.reduce_mean(q, 0)
        wd_p = tf.add_n([pl2norm * tf.nn.l2_loss(var) for var in self.theta_p])  # weight decay
        loss_p = -meanq + wd_p
        # policy optimization
        optim_p = tf.train.AdamOptimizer(learning_rate=plearning_rate, epsilon=1e-4)
        grads_and_vars_p = optim_p.compute_gradients(loss_p, var_list=self.theta_p)
        optimize_p = optim_p.apply_gradients(grads_and_vars_p)
        with tf.control_dependencies([optimize_p]):
            train_p = tf.group(update_pt)
        
        act2 = nets.policy(obs2, theta=self.theta_pt)
        with tf.variable_scope('q_target'):
            negQ_target = self.negQ(obs2, act2)
        negQ_entr_target = negQ_target - entropy(act2)
        act_target_grad, = tf.gradients(negQ_target, act2)
        act_entr_target_grad, = tf.gradients(negQ_entr_target, act2)
        q_target = -negQ_target
        q_target_entr = -negQ_entr_target

        if FLAGS.icnn_opt == 'adam':
            y = tf.where(term2, rew, rew + discount * q_target_entr)
            y = tf.maximum(q_entr - 1., y)
            y = tf.minimum(q_entr + 1., y)
            y = tf.stop_gradient(y)
            td_error = q_entr - y
        elif FLAGS.icnn_opt == 'bundle_entropy':
            q_target = tf.select(term2, rew, rew + discount * q2_entropy)
            q_target = tf.maximum(q_entropy - 1., q_target)
            q_target = tf.minimum(q_entropy + 1., q_target)
            q_target = tf.stop_gradient(q_target)
            td_error = q_entropy - q_target
        else:
            raise RuntimeError("Needs checking.")
        ms_td_error = tf.reduce_mean(tf.square(td_error), 0)

        regLosses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='q/')
        loss_q = ms_td_error + l2norm*tf.reduce_sum(regLosses)

        self.theta_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q/')
        self.theta_cvx_ = [v for v in self.theta_
                           if 'proj' in v.name]
        self.makeCvx = [v.assign(tf.abs(v)) for v in self.theta_cvx_]
        self.proj = [v.assign(tf.maximum(v, 0)) for v in self.theta_cvx_]
        # self.proj = [v.assign(tf.abs(v)) for v in self.theta_cvx_]

        self.theta_target_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope='q_target/')
        update_target = [theta_target_i.assign_sub(tau*(theta_target_i-theta_i))
                    for theta_i, theta_target_i in zip(self.theta_, self.theta_target_)]

        optim_q = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_and_vars_q = optim_q.compute_gradients(loss_q,var_list = self.theta_)
        optimize_q = optim_q.apply_gradients(grads_and_vars_q)
      

        summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'board'), self.sess.graph)
        if FLAGS.icnn_opt == 'adam':
            tf.summary.scalar('Qvalue', tf.reduce_mean(q))
        elif FLAGS.icnn_opt == 'bundle_entropy':
            tf.summary.scalar('Qvalue', tf.reduce_mean(q_entr))
        tf.summary.scalar('loss', ms_td_error)
        tf.summary.scalar('reward', tf.reduce_mean(rew))
        merged = tf.summary.merge_all
        # tf functions
        with self.sess.as_default():
            self._act_test = Fun(obs, act_test)
            self._act_expl = Fun(obs, act_expl)
            self._reset = Fun([], self.ou_reset)
            self._train = Fun([obs, act_train, rew, obs2, term2], [train_p,optimize_q, update_target, loss_q], merged, summary_writer)

        # initialize tf variables
        self.saver = tf.train.Saver(max_to_keep=1)
        ckpt = tf.train.latest_checkpoint(FLAGS.outdir + "/tf3")
        if ckpt:
            self.saver.restore(self.sess, ckpt)
        else:
            self.sess.run(tf.initialize_all_variables())
            self.sess.run(self.makeCvx)
            self.sess.run([theta_target_i.assign(theta_i)
                    for theta_i, theta_target_i in zip(self.theta_, self.theta_target_)])

        self.sess.graph.finalize()

        self.t = 0  # global training time (number of observations)

    def reset(self, obs):
        self._reset()
        self.noise = np.zeros(self.dimA)
        self.observation = obs  # initial observation

    def act(self, test=False):
        obs = np.expand_dims(self.observation, axis=0)
        action = self._act_test(obs) if test else self._act_expl(obs)
        action = np.clip(action, -1, 1)
        self.action = np.atleast_1d(np.squeeze(action, axis=0))  # TODO: remove this hack
        return self.action

    def observe(self, rew, term, obs2, test=False):
        obs1 = self.observation
        self.observation = obs2

        # train
        if not test:
            self.t = self.t + 1

            self.rm.enqueue(obs1, term, self.action, rew)

            if self.t > FLAGS.warmup:
                for i in range(FLAGS.iter):
                    loss = self.train()

    def train(self):
        obs, act, rew, ob2, term2, info = self.rm.minibatch(size=FLAGS.bsize)
        _,_, _, loss = self._train(obs, act, rew, ob2, term2, log=FLAGS.summary, global_step=self.t)
        self.sess.run(self.proj)
        return loss

    def adam(self, func, obs, plot=False):
        # if npr.random() < 1./20:
        #     plot = True
        b1 = 0.9
        b2 = 0.999
        lam = 0.5
        eps = 1e-8
        alpha = 0.01
        nBatch = obs.shape[0]
        act = np.zeros((nBatch, self.dimA))
        m = np.zeros_like(act)
        v = np.zeros_like(act)

        b1t, b2t = 1., 1.
        act_best, a_diff, f_best = [None]*3
        hist = {'act': [], 'f': [], 'g': []}
        for i in range(1000):
            f, g = func(obs, act)
            if plot:
                hist['act'].append(act.copy())
                hist['f'].append(f)
                hist['g'].append(g)

            if i == 0:
                act_best = act.copy()
                f_best = f.copy()
            else:
                prev_act_best = act_best.copy()
                I = (f < f_best)
                act_best[I] = act[I]
                f_best[I] = f[I]
                a_diff_i = np.mean(np.linalg.norm(act_best - prev_act_best, axis=1))
                a_diff = a_diff_i if a_diff is None \
                         else lam*a_diff + (1.-lam)*a_diff_i
                # print(a_diff_i, a_diff, np.sum(f))
                if a_diff < 1e-3 and i > 5:
                    #print('  + Adam took {} iterations'.format(i))
                    if plot:
                        self.adam_plot(func, obs, hist)
                    return act_best

            m = b1 * m + (1. - b1) * g
            v = b2 * v + (1. - b2) * (g * g)
            b1t *= b1
            b2t *= b2
            mhat = m/(1.-b1t)
            vhat = v/(1.-b2t)

            act -= alpha * mhat / (np.sqrt(v) + eps)
            # act = np.clip(act, -1, 1)
            act = np.clip(act, -1.+1e-8, 1.-1e-8)

        #print('  + Warning: Adam did not converge.')
        if plot:
            self.adam_plot(func, obs, hist)
        return act_best

    def adam_plot(self, func, obs, hist):
        hist['act'] = np.array(hist['act']).T
        hist['f'] = np.array(hist['f']).T
        hist['g'] = np.array(hist['g']).T
        if self.dimA == 1:
            xs = np.linspace(-1.+1e-8, 1.-1e-8, 100)
            ys = [func(obs[[0],:], [[xi]])[0] for xi in xs]
            fig = plt.figure()
            plt.plot(xs, ys)
            plt.plot(hist['act'][0,0,:], hist['f'][0,:], label='Adam')
            plt.legend()
            fname = os.path.join(FLAGS.outdir, 'adamPlt.png')
            #print("Saving Adam plot to {}".format(fname))
            plt.savefig(fname)
            plt.close(fig)
        elif self.dimA == 2:
            assert(False)
        else:
            xs = npr.uniform(-1., 1., (5000, self.dimA))
            ys = np.array([func(obs[[0],:], [xi])[0] for xi in xs])
            epi = np.hstack((xs, ys))
            pca = PCA(n_components=2).fit(epi)
            W = pca.components_[:,:-1]
            xs_proj = xs.dot(W.T)
            fig = plt.figure()

            X = Y = np.linspace(xs_proj.min(), xs_proj.max(), 100)
            Z = griddata(xs_proj[:,0], xs_proj[:,1], ys.ravel(),
                         X, Y, interp='linear')

            plt.contourf(X, Y, Z, 15)
            plt.colorbar()

            adam_x = hist['act'][:,0,:].T
            adam_x = adam_x.dot(W.T)
            plt.plot(adam_x[:,0], adam_x[:,1], label='Adam', color='k')
            plt.legend()

            fname = os.path.join(FLAGS.outdir, 'adamPlt.png')
            #print("Saving Adam plot to {}".format(fname))
            plt.savefig(fname)
            plt.close(fig)

    def bundle_entropy(self, func, obs):
        act = np.ones((obs.shape[0], self.dimA)) * 0.5
        def fg(x):
            value, grad = func(obs, 2 * x - 1)
            grad *= 2
            return value, grad

        act = bundle_entropy.solveBatch(fg, act)[0]
        act = 2 * act - 1

        return act
    def negQ(self, x, y, reuse=False):
        szs = [FLAGS.l1size, FLAGS.l2size]
        assert(len(szs) >= 1)
        fc = tflearn.fully_connected
        bn = tflearn.batch_normalization
        lrelu = tflearn.activations.leaky_relu

        if reuse:
            tf.get_variable_scope().reuse_variables()

        nLayers = len(szs)
        us = []
        zs = []
        z_zs = []
        z_ys = []
        z_us = []

        reg = 'L2'

        prevU = x
        for i in range(nLayers):
            with tf.variable_scope('u'+str(i)) as s:
                u = fc(prevU, szs[i], reuse=reuse, scope=s, regularizer=reg)
                if i < nLayers-1:
                    u = tf.nn.relu(u)
                    if FLAGS.icnn_bn:
                        u = bn(u, reuse=reuse, scope=s, name='bn')
            variable_summaries(u, suffix='u{}'.format(i))
            us.append(u)
            prevU = u

        prevU, prevZ = x, y
        for i in range(nLayers+1):
            sz = szs[i] if i < nLayers else 1
            z_add = []
            if i > 0:
                with tf.variable_scope('z{}_zu_u'.format(i)) as s:
                    zu_u = fc(prevU, szs[i-1], reuse=reuse, scope=s,
                              activation='relu', bias=True,
                              regularizer=reg, bias_init=tf.constant_initializer(1.))
                    variable_summaries(zu_u, suffix='zu_u{}'.format(i))
                with tf.variable_scope('z{}_zu_proj'.format(i)) as s:
                    z_zu = fc(tf.multiply(prevZ, zu_u), sz, reuse=reuse, scope=s,
                              bias=False, regularizer=reg)
                    variable_summaries(z_zu, suffix='z_zu{}'.format(i))
                z_zs.append(z_zu)
                z_add.append(z_zu)

            with tf.variable_scope('z{}_yu_u'.format(i)) as s:
                yu_u = fc(prevU, self.dimA, reuse=reuse, scope=s, bias=True,
                          regularizer=reg, bias_init=tf.constant_initializer(1.))
                variable_summaries(yu_u, suffix='yu_u{}'.format(i))
            with tf.variable_scope('z{}_yu'.format(i)) as s:
                z_yu = fc(tf.multiply(y, yu_u), sz, reuse=reuse, scope=s, bias=False,
                          regularizer=reg)
                z_ys.append(z_yu)
                variable_summaries(z_yu, suffix='z_yu{}'.format(i))
            z_add.append(z_yu)

            with tf.variable_scope('z{}_u'.format(i)) as s:
                z_u = fc(prevU, sz, reuse=reuse, scope=s,
                         bias=True, regularizer=reg,
                         bias_init=tf.constant_initializer(0.))
                variable_summaries(z_u, suffix='z_u{}'.format(i))
            z_us.append(z_u)
            z_add.append(z_u)

            z = tf.add_n(z_add)
            variable_summaries(z, suffix='z{}_preact'.format(i))
            if i < nLayers:
                # z = tf.nn.relu(z)
                z = lrelu(z, alpha=FLAGS.lrelu)
                variable_summaries(z, suffix='z{}_act'.format(i))

            zs.append(z)
            prevU = us[i] if i < nLayers else None
            prevZ = z

        z = tf.reshape(z, [-1], name='energies')
        return z



# Tensorflow utils
#
class Fun:
    """ Creates a python function that maps between inputs and outputs in the computational graph. """

    def __init__(self, inputs, outputs, summary_ops=None, summary_writer=None, session=None):
        self._inputs = inputs if type(inputs) == list else [inputs]
        self._outputs = outputs
        self._summary_op = tf.summary.merge(summary_ops) if type(summary_ops) == list else summary_ops
        self._session = session or tf.get_default_session()
        self._writer = summary_writer

    def __call__(self, *args, **kwargs):
        """
        Arguments:
          **kwargs: input values
          log: if True write summary_ops to summary_writer
          global_step: global_step for summary_writer
        """
        log = kwargs.get('log', False)

        feeds = {}
        for (argpos, arg) in enumerate(args):
            feeds[self._inputs[argpos]] = arg

        out = self._outputs + [self._summary_op] if log else self._outputs
        res = self._session.run(out, feeds)

        if log:
            i = kwargs['global_step']
            self._writer.add_summary(res[-1], global_step=i)
            res = res[:-1]

        return res

def exponential_moving_averages(theta, tau=0.001):
    ema = tf.train.ExponentialMovingAverage(decay=1 - tau)
    update = ema.apply(theta)  # also creates shadow vars
    averages = [ema.average(x) for x in theta]
    return averages, update

def entropy(x): #the real concave entropy function
    x_move_reg = tf.clip_by_value((x + 1) / 2, 0.0001, 0.9999)
    pen = x_move_reg * tf.log(x_move_reg) + (1 - x_move_reg) * tf.log(1 - x_move_reg)
    return -tf.reduce_sum(pen, 1)
