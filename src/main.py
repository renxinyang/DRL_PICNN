#!/usr/bin/env python3
# Code from Repo SimonRamstedt/ddpg
# Heavily modified

import os
import pprint
import json

import gym,logging
import numpy as np
import tensorflow as tf

import agent
import normalized_env
import runtime_env

import time

import setproctitle
import matplotlib.pyplot as plt

srcDir = os.path.dirname(os.path.realpath(__file__))
rlDir = os.path.join(srcDir, '..')
plotScr = os.path.join(rlDir, 'plot-single.py')

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('env', '', 'gym environment')
flags.DEFINE_string('outdir', 'output', 'output directory')
flags.DEFINE_boolean('force', False, 'overwrite existing results')
flags.DEFINE_integer('train', 1000, 'training timesteps between testing episodes')
flags.DEFINE_integer('test', 100, 'testing episodes between training timesteps')
flags.DEFINE_integer('tmax', 1000, 'maxium timesteps each episode')
flags.DEFINE_integer('total', 5000000, 'total training timesteps')
flags.DEFINE_float('monitor', 0.01, 'probability of monitoring a test episode')
flags.DEFINE_string('model', 'ICNN', 'reinforcement learning model[DDPG, NAF, ICNN]')
flags.DEFINE_integer('tfseed', 0, 'random seed for tensorflow')
flags.DEFINE_integer('gymseed', 0, 'random seed for openai gym')
flags.DEFINE_integer('npseed', 0, 'random seed for numpy')
flags.DEFINE_float('ymin', 0, 'random seed for numpy')
flags.DEFINE_float('ymax', 1000, 'random seed for numpy')

setproctitle.setproctitle('ICNN.RL.{}.{}.{}'.format(
    FLAGS.env,FLAGS.model,FLAGS.tfseed))

os.makedirs(FLAGS.outdir, exist_ok=True)
with open(os.path.join(FLAGS.outdir, 'flags.json'), 'w') as f:
    json.dump(FLAGS.__flags, f, indent=2, sort_keys=True)

if FLAGS.model == 'DDPG':
    import ddpg
    Agent = ddpg.Agent
elif FLAGS.model == 'DDPG_CON':
    import ddpg_convex
    Agent = ddpg_convex.Agent
elif FLAGS.model == 'DDPG_ARCH':
    import ddpg_arch
    Agent = ddpg_arch.Agent
elif FLAGS.model == 'ICNN':
    import icnn
    Agent = icnn.Agent
elif FLAGS.model == 'ICNN_ARCH':
    import icnn_arch
    Agent = icnn_arch.Agent



class Experiment(object):

    def run(self):
        self.train_timestep = 0
        self.test_timestep = 0

        # create normal
        self.env = normalized_env.make_normalized_env(gym.make(FLAGS.env))
        tf.set_random_seed(FLAGS.tfseed)
        np.random.seed(FLAGS.npseed)
        #self.env.monitor.start(os.path.join(FLAGS.outdir, 'monitor'), force=FLAGS.force)
        self.env = gym.wrappers.Monitor(self.env,os.path.join(FLAGS.outdir, 'monitor'),force=True)
        self.env.seed(FLAGS.gymseed)
        gym.logger.set_level(logging.WARNING)

        dimO = self.env.observation_space.shape
        dimA = self.env.action_space.shape
        pprint.pprint(self.env.spec.__dict__)

        self.agent = Agent(dimO, dimA=dimA)
        test_log = open(os.path.join(FLAGS.outdir, 'test.log'), 'w')
        train_log = open(os.path.join(FLAGS.outdir, 'train.log'), 'w')
        x_offline = []
        y_offline = []
        x_online = []
        y_online = []
        while self.train_timestep < FLAGS.total:
            # test
            reward_list = []
            for _ in range(FLAGS.test):
                reward, timestep = self.run_episode(
                    test=True, monitor=np.random.rand() < FLAGS.monitor)
                reward_list.append(reward)
                self.test_timestep += timestep
            avg_reward = np.mean(reward_list)
            print('Average test return {} after {} timestep of training.'.format(
                avg_reward, self.train_timestep))
            x_offline.append(self.train_timestep)
            y_offline.append(avg_reward)
            #test_log.write("{}\t{}\n".format(self.train_timestep, avg_reward))
            #test_log.flush()
            if avg_reward > -7:
                break
            if self.train_timestep % 5000 == 0:
                print('Saving ckpt at {} timesteps.'.format(self.train_timestep))
        #os.makedirs(os.path.join(FLAGS.outdir, "tf3"))
        #ckpt = os.path.join(FLAGS.outdir, "tf3/model_ddpgcon.ckpt")
        #self.agent.saver.save(self.agent.sess, ckpt)
                address_online = FLAGS.outdir+'/plots/Reacher/'+FLAGS.model+'_on.png'
                address_offline = FLAGS.outdir+'/plots/Reacher/'+FLAGS.model+'_off.png'
                np.save(FLAGS.outdir+'/plots/Reacher/'+FLAGS.model+'_xon.np',x_online)
                np.save(FLAGS.outdir+'/plots/Reacher/'+FLAGS.model+'_xoff.np',x_offline)
                np.save(FLAGS.outdir+'/plots/Reacher/'+FLAGS.model+'_yon.np',y_online)
                np.save(FLAGS.outdir+'/plots/Reacher/'+FLAGS.model+'_yoff.np',y_offline)
                plt.clf()
                plt.plot(x_online,y_online)
                plt.xlabel('Learning Steps')
                plt.ylabel('Online Average Rewards')
                plt.savefig(address_online)
                plt.clf()
                plt.plot(x_offline,y_offline)
                plt.xlabel('Learning Steps')
                plt.ylabel('Offline Average Rewards')
                plt.savefig(address_offline)
            # train
            reward_list = []
            last_checkpoint = np.floor(self.train_timestep / FLAGS.train)
            while np.floor(self.train_timestep / FLAGS.train) == last_checkpoint:
                #print('=== Running episode')
                reward, timestep = self.run_episode(test=False, monitor=False)
                reward_list.append(reward)
                self.train_timestep += timestep
                #train_log.write("{}\t{}\n".format(self.train_timestep, reward))
                #train_log.flush()
            avg_reward = np.mean(reward_list)
            print('Average train return {} after {} timestep of training.'.format(
                avg_reward, self.train_timestep))
            x_online.append(self.train_timestep)
            y_online.append(avg_reward)
            #os.system('{} {}'.format(plotScr, FLAGS.outdir))

        self.env.close()
        print('Saving ckpt at {} timesteps.'.format(self.train_timestep))
        #os.makedirs(os.path.join(FLAGS.outdir, "tf3"))
        #ckpt = os.path.join(FLAGS.outdir, "tf3/model_ddpgcon.ckpt")
        #self.agent.saver.save(self.agent.sess, ckpt)
        address_online = FLAGS.outdir+'/plots/Reacher/'+FLAGS.model+'_on.png'
        address_offline = FLAGS.outdir+'/plots/Reacher/'+FLAGS.model+'_off.png'
        np.save(FLAGS.outdir+'/plots/Reacher/'+FLAGS.model+'_xon.np',x_online)
        np.save(FLAGS.outdir+'/plots/Reacher/'+FLAGS.model+'_xoff.np',x_offline)
        np.save(FLAGS.outdir+'/plots/Reacher/'+FLAGS.model+'_yon.np',y_online)
        np.save(FLAGS.outdir+'/plots/Reacher/'+FLAGS.model+'_yoff.np',y_offline)
        plt.clf()
        plt.plot(x_online,y_online)
        plt.xlabel('Learning Steps')
        plt.ylabel('Online Average Rewards')
        plt.savefig(address_online)
        plt.clf()
        plt.plot(x_offline,y_offline)
        plt.xlabel('Learning Steps')
        plt.ylabel('Offline Average Rewards')
        plt.savefig(address_offline)

    def run_episode(self, test=True, monitor=False):
        #self.env.configure(lambda _: monitor)
        observation = self.env.reset()
        self.agent.reset(observation)
        sum_reward = 0
        timestep = 0
        term = False
        times = {'act': [], 'envStep': [], 'obs': []}
        while not term:
            start = time.clock()
            action = self.agent.act(test=test)
            times['act'].append(time.clock()-start)

            start = time.clock()
            observation, reward, term, info = self.env.step(action)
            times['envStep'].append(time.clock()-start)
            term = (not test and timestep + 1 >= FLAGS.tmax) or term

            filtered_reward = self.env.filter_reward(reward)

            start = time.clock()
            self.agent.observe(filtered_reward, term, observation, test=test)
            times['obs'].append(time.clock()-start)

            sum_reward += reward
            timestep += 1
        '''
        print('=== Episode stats:')
        for k,v in sorted(times.items()):
            print('  + Total {} time: {:.4f} seconds'.format(k, np.mean(v)))

        print('  + Reward: {}'.format(sum_reward))
        '''
        return sum_reward, timestep


def main():
    Experiment().run()

if __name__ == '__main__':
    runtime_env.run(main, FLAGS.outdir)
