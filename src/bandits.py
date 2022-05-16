#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from src.utils import StatsSimulation


class TLMMAB(object):
    """
        Structure of Multi tasking MAB
    """

    def __init__(self,
                 means,
                 optimal_g,
                 strategy,
                 reward,
                 offline=False,
                 tasks_pmf=None,
                 **kwargs):

        # Average of each arm
        self.means = means

        # Spaces dimensionality
        self.n_X = means.shape[0]
        self.n_G = means.shape[1]
        self.n_H = means.shape[2]

        # Type of reward
        self.reward_type = reward(self.means)

        # Shared structure
        self.optimal_g = optimal_g

        # Online or offline
        self.offline = offline

        self.tasks_pmf = tasks_pmf

        # Create agent
        self.agent = strategy(self.n_X, self.n_G, self.n_H, **kwargs)
        self.play_history = []

    def sample_reward(self, x, g, h):
        """ Sample reward from (x,g,h) """
        return self.reward_type.sample_reward(x, g, h)

    def simulate_offline(self, exit_condition=None):
        """ Routine used to simulate sample complexity """
        if self.offline is False:
            return None

        stats = StatsSimulation(self.n_G, save_history=False)
        self.agent.stats = stats
        while True:
            # Fetch next action
            x, g, h = self.agent.play()
            # Sample reward
            reward = self.sample_reward(x, g, h)
            # Update agent statistics
            done = self.agent.update((x, g, h), reward)

            # Update logging
            stats.update(x, (g, h), reward,
                         np.max(self.means[x][self.optimal_g]))
            stats.update_sample_complexity(self.agent.emp_means, self.agent.w,
                                           self.agent.last_lb)
            if done:
                break

            # Check exit condition
            if exit_condition is not None:
                if exit_condition(self.agent):
                    break
        stats.final_update(self.agent.N_x_g_h)
        stats.best_arms_update(self.agent.best_arms())

        return stats

    def simulate_online(self,
                        horizon=None,
                        exit_condition=None,
                        task_list=None,
                        save_history=True):
        """ Routine used to simulate regret minimization """
        if self.offline is True:
            return None

        if self.tasks_pmf is None and task_list is None:
            print("Can't execute online simulation without"
                  "a pmf or list of tasks!")
        stats = StatsSimulation(self.n_G, horizon, save_history=save_history)
        self.agent.stats = stats
        for t in range(horizon):
            # Simulate environment

            # Fetch tasks
            if task_list is None:
                task = np.random.choice(self.n_X, p=self.tasks_pmf)
            else:
                task = task_list[t]
            # Fetch agent action
            action = self.agent.play(task)
            # Sample reward
            reward = self.sample_reward(task, action[0], action[1])

            # Update agent statistics/strategies
            self.agent.update(task, action, reward)

            # Update logs
            stats.update(task, action, reward,
                         np.max(self.means[task][self.optimal_g]))

            # Check exit condition
            if exit_condition is not None:
                if exit_condition(self.agent):
                    break

        # Compute regret and save number of pulls
        stats.final_update(self.agent.N_x_g_h)
        stats.best_arms_update(self.agent.best_arms())

        return stats

    def simulate(self,
                 horizon=500,
                 exit_condition=None,
                 task_list=None,
                 save_history=True):
        """
            Routine used to simulate both the
            regret minimization/sample complexity settings
        """

        if self.offline is True:
            return self.simulate_offline(exit_condition=exit_condition)
        else:
            return self.simulate_online(
                horizon=horizon,
                exit_condition=exit_condition,
                task_list=task_list,
                save_history=save_history)
