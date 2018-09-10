import os
import time
from collections import deque
import pickle

from baselines.ddpg.ddpg import DDPG
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI


def scale_range(x, x_min, x_max, y_min, y_max):
    """ Scales the entries in x which have a range between x_min and x_max
    to the range defined between y_min and y_max. """
    # y = a*x + b
    # a = deltaY/deltaX
    # b = y_min - a*x_min (or b = y_max - a*x_max)
    y = (y_max - y_min) / (x_max - x_min) * x + (y_min*x_max - y_max*x_min) / (x_max - x_min)
    return y

def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50, restore=True):
    rank = MPI.COMM_WORLD.Get_rank()

    # assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    # max_action = env.action_space.high
    # logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, (env.action_space.shape[0],),
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, observation_range=(env.observation_space.low[0], env.observation_space.high[0]),
        action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Set up saving stuff only for a single worker.
    savingModelPath = "/home/joel/Documents/saved_models_OpenAI_gym/"
    if rank == 0:
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    else:
        saver = None

    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    with U.single_threaded_session() as sess:
        # Prepare everything.

        # from https://github.com/openai/baselines/issues/162#issuecomment-397356482 and
        # https://www.tensorflow.org/api_docs/python/tf/train/import_meta_graph
        
        if restore == True:
            # restoring doesn't actually work
            logger.info("Restoring from saved model")
            saver = tf.train.import_meta_graph(savingModelPath + "ddpg_test_model.meta")
            saver.restore(sess, tf.train.latest_checkpoint(savingModelPath))
        else:
            logger.info("Starting from scratch!")
            sess.run(tf.global_variables_initializer()) # this should happen here and not in the agent right?


        agent.initialize(sess)
        sess.graph.finalize()

        agent.reset()
        obs = env.reset()
        if eval_env is not None:
            eval_obs = eval_env.reset()
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        epoch = 0
        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        for epoch in range(nb_epochs):
            start_time_epoch = time.time()
            for cycle in range(nb_epoch_cycles):
                start_time_cycle = time.time()
                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    if(t_rollout == nb_rollout_steps - 2):
                        print("break here")
                    start_time_rollout = time.time()
                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    # e.g. action = array([ 0.02667301,  0.9654905 , -0.5694418 , -0.40275186], dtype=float32)

                    np.set_printoptions(precision=3)
                    print("selected (unscaled) action: " + str(action)) # e.g. [ 0.04  -0.662 -0.538  0.324]
                    # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    target = scale_range(action, -1, 1, env.action_space.low, env.action_space.high)
                    
                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    assert target.shape == env.action_space.shape
                    new_obs, r, done, info = env.step(target)
                    t += 1
                    if rank == 0 and render:
                        env.render()
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs

                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        agent.reset()
                        obs = env.reset()
                    
                    logger.info('runtime rollout-step {0}.{1}.{2}: {3}s'.format(epoch, cycle, t_rollout, time.time() - start_time_rollout))
                # for rollout_steps

                # Train.
                print("Training the Agent")
                start_time_train = time.time()
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps): # 50 iterations
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise() # e.g. 0.7446093559265137
                        epoch_adaptive_distances.append(distance)

                    cl, al = agent.train()
                    epoch_critic_losses.append(cl) # e.g. 25.988863
                    epoch_actor_losses.append(al) # e.g. -0.008966461
                    agent.update_target_net()

                # Evaluate.
                eval_episode_rewards = []
                eval_qs = []
                if eval_env is not None:
                    eval_episode_reward = 0.
                    for t_rollout in range(nb_eval_steps):
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        if render_eval:
                            eval_env.render()
                        eval_episode_reward += eval_r

                        eval_qs.append(eval_q)
                        if eval_done:
                            eval_obs = eval_env.reset()
                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_reward = 0.
                logger.info('runtime training actor & critic: {}s'.format(time.time() - start_time_train))

                # Saving the trained model
                if(saver is not None):
                    logger.info("saving the trained model")
                    start_time_save = time.time()
                    saver.save(sess, savingModelPath + "ddpg_test_model")
                    logger.info('runtime saving: {}s'.format(time.time() - start_time_save))

                logger.info('runtime epoch-cycle {0}: {1}s'.format(cycle, time.time() - start_time_cycle))
            # for epoch_cycles

            mpi_size = MPI.COMM_WORLD.Get_size()
            # Log stats.
            # XXX shouldn't call np.mean on variable length lists
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = stats.copy()
            combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
            combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
            combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
            combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
            combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
            combined_stats['total/duration'] = duration
            combined_stats['total/steps_per_second'] = float(t) / float(duration)
            combined_stats['total/episodes'] = episodes
            combined_stats['rollout/episodes'] = epoch_episodes
            combined_stats['rollout/actions_std'] = np.std(epoch_actions)
            # Evaluation statistics.
            if eval_env is not None:
                combined_stats['eval/return'] = eval_episode_rewards
                combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                combined_stats['eval/Q'] = eval_qs
                combined_stats['eval/episodes'] = len(eval_episode_rewards)
            def as_scalar(x):
                if isinstance(x, np.ndarray):
                    assert x.size == 1
                    return x[0]
                elif np.isscalar(x):
                    return x
                else:
                    raise ValueError('expected scalar, got %s'%x)
            combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
            combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

            # Total statistics.
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)
            
            # Saving the trained model
            if(saver is not None):
                logger.info("saving the trained model")
                start_time_save = time.time()
                saver.save(sess, savingModelPath + "ddpg_model_epochSave", global_step=epoch)
                logger.info('runtime saving: {}s'.format(time.time() - start_time_save))

            logger.info('runtime epoch {0}: {1}s'.format(epoch, time.time() - start_time_epoch))
        # for epochs
    # with session
# train
