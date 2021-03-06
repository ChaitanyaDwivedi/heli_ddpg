from rdpg import *
import opensim as osim
from osim.http.client import Client
from osim.env import *
from history import History
ENV_NAME = 'learning_to_run'
PATH = 'models/'
EPISODES = 100000
TEST = 5

def main():
    env = RunEnv(visualize=False)
    env.reset(difficulty = 0)
    agent = RDPG(env)

    returns = []
    rewards = []

    for episode in xrange(EPISODES):
        state = env.reset(difficulty = 0)
        reward_episode = []
        print ("episode:",episode)
        #Initializing empty history
        history = History(state)
        # Train
        for step in xrange(env.spec.timestep_limit):
            action = agent.noise_action(history)
            next_state,reward,done,_ = env.step(action)
            # appending to history
            history.append(next_state,action,reward)
            reward_episode.append(reward)
            if done:
                break
        # storing the history into replay buffer and if the number of histories sequence is above the threshod, start training
        agent.perceive(history)
        # Testing:
        #if episode % 1 == 0:
        # if episode % 1000 == 0 and episode > 50:
        #     agent.save_model(PATH, episode)

        #     total_return = 0
        #     ave_reward = 0
        #     for i in xrange(TEST):
        #         state = env.reset()
        #         reward_per_step = 0
        #         for j in xrange(env.spec.timestep_limit):
        #             action = agent.action(state) # direct action for test
        #             state,reward,done,_ = env.step(action)
        #             total_return += reward
        #             if done:
        #                 break
        #             reward_per_step += (reward - reward_per_step)/(j+1)
        #         ave_reward += reward_per_step

        #     ave_return = total_return/TEST
        #     ave_reward = ave_reward/TEST
        #     returns.append(ave_return)
        #     rewards.append(ave_reward)

        #     print 'episode: ',episode,'Evaluation Average Return:',ave_return, '  Evaluation Average Reward: ', ave_reward

if __name__ == '__main__':
    main()
