
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
flags.DEFINE_string('arch','2x200_','architecture of convex neural network')
flags.DEFINE_boolean('force', False, 'overwrite existing results')
flags.DEFINE_integer('train', 1000, 'training timesteps between testing episodes')
flags.DEFINE_integer('test', 100, 'testing episodes between training timesteps')
flags.DEFINE_integer('tmax', 1000, 'maxium timesteps each episode')
flags.DEFINE_integer('total',100000, 'total training timesteps')
flags.DEFINE_float('monitor', 0.01, 'probability of monitoring a test episode')
flags.DEFINE_string('model', 'ICNN', 'reinforcement learning model[DDPG, NAF, ICNN]')
flags.DEFINE_integer('tfseed', 0, 'random seed for tensorflow')
flags.DEFINE_integer('gymseed', 0, 'random seed for openai gym')
flags.DEFINE_integer('npseed', 0, 'random seed for numpy')
flags.DEFINE_integer('i',0,'iteration round')
flags.DEFINE_integer('num_layer',2,'number of layers')
flags.DEFINE_integer('num_nodes',2,'number of hundreds of nodes per layer')
flags.DEFINE_float('ymin', 0, 'random seed for numpy')
flags.DEFINE_float('ymax', 1000, 'random seed for numpy')
flags.DEFINE_integer('parseq',0,'No, of hyperparameter to test.')

setproctitle.setproctitle('ICNN.RL.{}.{}.{}'.format(
    FLAGS.env,FLAGS.model,FLAGS.tfseed))

os.makedirs(FLAGS.outdir, exist_ok=True)
#with open(os.path.join(FLAGS.outdir, 'flags.json'), 'w') as f:
   # json.dump(FLAGS.__flags, f, indent=2, sort_keys=True)

if FLAGS.model == 'DDPG':
    import ddpg
    Agent1 = ddpg.Agent
    Agent2 = ddpg.Agent
    Agent3 = ddpg.Agent
    Agent4 = ddpg.Agent
elif FLAGS.model == 'DDPG_CON':
    import ddpg_convex
    Agent1 = ddpg_convex.Agent
    Agent2 = ddpg_convex.Agent
    Agent3 = ddpg_convex.Agent
    Agent4 = ddpg_convex.Agent
    Agent5 = ddpg_convex.Agent
    Agent6 = ddpg_convex.Agent
    Agent7 = ddpg_convex.Agent
    Agent8 = ddpg_convex.Agent
    Agent9 = ddpg_convex.Agent
    Agent10 = ddpg_convex.Agent
elif FLAGS.model == 'DDPG_ARCH':
    import ddpg_arch
    Agent1 = ddpg_arch.Agent
    Agent2 = ddpg_arch.Agent
    Agent3 = ddpg_arch.Agent
    Agent4 = ddpg_arch.Agent
    Agent5 = ddpg_arch.Agent
    Agent6 = ddpg_arch.Agent
    Agent7 = ddpg_arch.Agent
    Agent8 = ddpg_arch.Agent
    Agent9 = ddpg_arch.Agent
    Agent10 = ddpg_arch.Agent
elif FLAGS.model == 'ICNN':
    import icnn
    Agent1 = icnn.Agent
    Agent2 = icnn.Agent
    Agent3 = icnn.Agent
    Agent4 = icnn.Agent
    Agent5 = icnn.Agent
elif FLAGS.model == 'ICNN_ARCH':
    import icnn_arch
    Agent1 = icnn_arch.Agent
    Agent2 = icnn_arch.Agent
    Agent3 = icnn_arch.Agent
    Agent4 = icnn_arch.Agent
    Agent5 = icnn_arch.Agent

class Experiment(object):
    def run(self):
        Agents = [Agent1,Agent2,Agent3,Agent4,Agent5]
        rd_seeds = [8,15,20,35]
        for i in range(4):
            if FLAGS.i !=0:
                i = FLAGS.i
            self.train_timestep = 0
            self.test_timestep = 0

        # create normal
            self.env = normalized_env.make_normalized_env(gym.make(FLAGS.env))
            tf.set_random_seed(rd_seeds[i])
            np.random.seed(rd_seeds[i])
        #self.env.monitor.start(os.path.join(FLAGS.outdir, 'monitor'), force=FLAGS.force)
           # self.env = gym.wrappers.Monitor(self.env,os.path.join(FLAGS.outdir, 'monitor'),force=True)
            self.env.seed(rd_seeds[i])
            gym.logger.set_level(logging.WARNING)

            dimO = self.env.observation_space.shape
            dimA = self.env.action_space.shape
            pprint.pprint(self.env.spec.__dict__)
            if FLAGS.model == "ICNN" or FLAGS.model=="ICNN_ARCH":
                print("Yes,ICNN!")
                self.agent = Agents[i](dimO=dimO,dimA=dimA)
            else:
                self.agent = Agents[i](dimO=dimO, dimA=dimA,num_layer=FLAGS.num_layer,num_nodes=FLAGS.num_nodes)
            test_log = open(os.path.join(FLAGS.outdir, 'test.log'), 'w')
            train_log = open(os.path.join(FLAGS.outdir, 'train.log'), 'w')
            x_offline = []
            y_offline = []
            x_online = []
            y_online = []
            reward_best = -2000
            while self.train_timestep <= FLAGS.total:
            # test
                reward_list = []
                for _ in range(FLAGS.test):
                    reward, timestep = self.run_episode(
                        test=True, monitor=np.random.rand() < FLAGS.monitor)
                    reward_list.append(reward)
                    self.test_timestep += timestep
                    if reward > reward_best:
                        reward_best = reward
                avg_reward = np.mean(reward_list)
                print('Average test return {} after {} timestep of training.'.format(
                    avg_reward, self.train_timestep))
                x_offline.append(self.train_timestep)
                y_offline.append(avg_reward)
            #test_log.write("{}\t{}\n".format(self.train_timestep, avg_reward))
            #test_log.flush()
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
            if FLAGS.parseq == 0:
                subadr = "pl2norm/"
                parval = FLAGS.pl2norm
            elif FLAGS.parseq == 1:
                subadr = "rate/"
                parval = FLAGS.rate
            elif FLAGS.parseq == 2:
                subadr = "prate/"
                parval = FLAGS.prate
            elif FLAGS.parseq == 4:
                subadr = "mix/"
                if FLAGS.rate == 0.0005 and FLAGS.prate == 0.00005:
                    parval=1
                elif FLAGS.rate == 0.0001 and FLAGS.prate == 0.00001:
                    parval=2
                elif FLAGS.rate == 0.001 and FLAGS.prate == 0.0001:
                    parval=3
                elif FLAGS.l2norm == 0.0001 and FLAGS.pl2norm == 0: 
                    parval=4 
                elif FLAGS.l2norm == 0.00001 and FLAGS.pl2norm == 0.0005:
                    parval=5
                elif FLAGS.l2norm == 0.00005 and FLAGS.pl2norm == 0.001:
                    parval=6
            if FLAGS.env=="HalfCheetah-v2":
                env_addr="HalfCheetah/"+subadr
            elif FLAGS.env=="Pendulum-v0":
                env_addr="Pendulum/"+subadr
            elif FLAGS.env =="Reacher-v2":
                env_addr="Reacher/"
            elif FLAGS.env == "MountainCarContinuous-v0":
                env_addr="MCContinuous/"
            #ckpt=FLAGS.outdir+"/"+env_addr+"tf/"+FLAGS.model+"_"+FLAGS.arch+str(parval)+"_"+str(i)+".ckpt"
           # os.makedirs(os.path.join(ckpt_addr, "tf"))
            #self.agent.saver.save(self.agent.sess, ckpt)
            x_offline.append(reward_best)
            print('Saving ckpt at {} timesteps.'.format(self.train_timestep))
            print("Best Reward:{}".format(reward_best))
            np.save(FLAGS.outdir+'/plots/'+env_addr+FLAGS.model+'_xon'+FLAGS.arch+str(parval)+"_"+str(i),x_online)
            np.save(FLAGS.outdir+'/plots/'+env_addr+FLAGS.model+'_xoff'+FLAGS.arch+str(parval)+"_"+str(i),x_offline)
            np.save(FLAGS.outdir+'/plots/'+env_addr+FLAGS.model+'_yon'+FLAGS.arch+str(parval)+"_"+str(i),y_online)
            np.save(FLAGS.outdir+'/plots/'+env_addr+FLAGS.model+'_yoff'+FLAGS.arch+str(parval)+"_"+str(i),y_offline)
            tf.reset_default_graph()
            tf.Graph().as_default()
        

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
            time.sleep(.002)
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
