#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class RewardDistribution(object):
    def __init__(self, means, std):
        self.means = means
        self.std = std
        self.distribution = None

    def sample_reward(self, x, g, h):
        if self.distribution is None:
            raise ValueError("Invalid distribution")

        try:
            m = self.means[x][g][h]
        except Exception as _:
            raise ValueError(
                "Invalid reward indexes ({},{},{}). Shape of rewards is {}".
                format(x, g, h, self.means.shape))

        return self.distribution(m, self.std)


class RewardGaussianDistribution(RewardDistribution):
    def __init__(self, means, std=1.0):
        RewardDistribution.__init__(self, means, std)
        self.distribution = lambda m, s: np.random.normal(m, s, size=1)


class RewardBernoulliDistribution(RewardDistribution):
    def __init__(self, means, std=None):
        RewardDistribution.__init__(self, means, std)
        self.distribution = lambda m, _: np.random.binomial(1, m)


class StatsSimulation(object):
    def __init__(self, G, horizon=0, save_history=True):
        self.rewards = []
        self.best_case_reward = [] if save_history else 0.0
        self.play_history = []
        self.regret = 0.0
        self.cumulated_reward = 0.0
        self.npulls = None
        self.horizon = horizon
        self.best_arms = None
        self.save_history = save_history
        self.emp_means_history = []
        self.w_history = []
        self.lb_history = []
        self.phaseskey = {'estimation': 0, 'exploration': 1, 'exploitation': 2}

        self.phases = [[0] * 3]
        self.T = 0

        self.selected_g = [[0] * G]

    def update_sample_complexity(self, emp_means, w, lb):
        self.w_history.append(np.copy(w))
        self.lb_history.append(lb)
        self.emp_means_history.append(np.copy(emp_means))

    def update(self, task, action, reward, best_reward):
        if self.save_history:
            self.update_g(action[0])
            self.play_history.append((task, action, reward, best_reward))
            self.best_case_reward.append(best_reward)
            self.rewards.append(reward)
        else:
            self.cumulated_reward += reward
            self.best_case_reward += best_reward

    def final_update(self, pulls):
        if self.save_history:
            self.cumulated_reward = np.cumsum(self.rewards)
            self.best_case_reward = np.cumsum(self.best_case_reward)
        self.regret = self.best_case_reward - self.cumulated_reward
        self.npulls = pulls
        if not self.save_history:
            self.regret = np.array([self.regret])

    def best_arms_update(self, arms):
        self.best_arms = arms
        self.T = np.sum(self.npulls)

    def update_phase(self, phase):
        if self.save_history:
            self.phases.append(list(self.phases[-1]))
        self.phases[-1][self.phaseskey[phase]] += 1

    def update_g(self, g):
        if self.save_history:
            self.selected_g.append(list(self.selected_g[-1]))
        self.selected_g[-1][g] += 1
