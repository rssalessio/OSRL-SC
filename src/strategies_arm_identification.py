#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import math

import pyximport
_ = pyximport.install()
from src.math_func_cython import solveFBern, klBern, beta_threshold, Tconf
from src.mt_g_lower_bound import MTArmIdentificationLB_GTerm

debug = False


class PlayerStrategyIdentification(object):
    def __init__(self, X, G, H, flatten=False):
        self.t = 0  # current round
        self.n_X = X
        self.n_G = G
        self.n_H = H
        self.emp_means = np.zeros(
            (X, G, H) if not flatten else X * G * H,
            order='C',
            dtype=np.float64)  # empirical means
        self.N_x_g_h = np.zeros(
            (X, G, H) if not flatten else X * G * H,
            order='C',
            dtype=np.float64)  # number of pulls for each arm (x,g,h)
        self.N_x_g = np.zeros(
            (X, G), order='C',
            dtype=np.float64)  # number of pulls for each arm (x,g)
        self.N_x = np.zeros(
            X, order='C', dtype=np.float64)  # number of pulls per task
        self.done = False
        self.flatten = flatten

    def is_done(self):
        return self.done

    def _convert_index_flatten(self, x, g, h):
        return self.n_H * (self.n_G * x + g) + h

    def update_means(self, action, obs):
        self.t += 1
        x, g, h = action
        #print(self.flatten)
        if self.flatten:
            # Convert (x,g,h) to k
            # Example: (0, 1, 0) with (X,G,H) = (3,4,5) becomes
            # H * 1 + 0
            # Example: (2, 1, 3) becomes
            # G * H * 2 + H * 1 + 3 =  40 + 5 + 3=  48
            k = self._convert_index_flatten(x, g, h)
            self.N_x_g_h[k] += 1
            self.emp_means[k] = ((self.N_x_g_h[k] - 1) * self.emp_means[k] +
                                 obs) / self.N_x_g_h[k]
        else:
            self.N_x_g_h[x][g][h] += 1
            self.emp_means[x][g][h] = (
                (self.N_x_g_h[x][g][h] - 1) * self.emp_means[x][g][h] +
                obs) / self.N_x_g_h[x][g][h]

        self.N_x[x] += 1
        self.N_x_g[x][g] += 1

    def uniform_play(self):
        return (np.random.choice(self.n_X), np.random.choice(self.n_G),
                np.random.choice(self.n_H))

    @property
    def annealing_probability(self):
        return np.log(self.t + 1) / (self.t + 1)

    def update(self, action, obs):
        self.update_means(action, obs)
        return self.is_done()

    def best_arms(self, m=1):

        if m <= 0:
            raise Exception(
                "m should be a positive number (number of best arms).")
        if self.flatten:
            self.emp_means = np.reshape(self.emp_means,
                                        (self.n_X, self.n_G, self.n_H))
        _, goptimal, _ = np.unravel_index(
            np.argmax(self.emp_means, axis=None), self.emp_means.shape)
        arms = [
            np.argmax(self.emp_means[x][goptimal]) for x in range(self.n_X)
        ]
        arms.insert(0, goptimal)
        return arms


class TrackAndStopD(PlayerStrategyIdentification):
    def __init__(self, X, G, H, delta, challenger=False, stopping_rule=0):
        PlayerStrategyIdentification.__init__(self, X, G, H, flatten=True)
        self.name = 'TrackAndStop - D Tracking'
        self.delta = delta
        self.K = self.n_X * self.n_H * self.n_G
        self.w = np.zeros(self.K)
        self.threshold = 0
        self.opt_arms = np.array([0])
        self.challenger = challenger
        self.challenger_id = 0
        self.update_maximum_arms()
        self._stoppingrule = stopping_rule
        self.last_lb = 0.

    def build_exploration_set(self):
        return np.argwhere(self.N_x_g_h < self.threshold)

    def get_index(self, x):
        return np.unravel_index(x, (self.n_X, self.n_G, self.n_H))

    def update_maximum_arms(self):
        self.mask_opt_arm = np.isclose(self.emp_means, np.max(self.emp_means))
        self.opt_arms = np.argwhere(self.mask_opt_arm)
        self.non_opt_arms = np.argwhere(~self.mask_opt_arm)

    def _compute_score(self):
        # compute the stopping statistic
        idx_other_arms = ~self.mask_opt_arm

        # Best arms stats
        muBest = self.emp_means[self.opt_arms]
        nBest = self.N_x_g_h[self.opt_arms]

        # other arms stats
        nOther = self.N_x_g_h[idx_other_arms]
        muOther = self.emp_means[idx_other_arms]

        muMid = (
            (nBest * muBest + nOther * muOther) / (nBest + nOther)).flatten()
        _min = np.inf

        if self.challenger:
            self.challenger_id = 0

        for i in range(self.N_x_g_h.shape[0] - 1):
            Z = (nBest * klBern(muBest, muMid[i]) +
                 nOther[i] * klBern(muOther[i], muMid[i]))
            if Z < _min:
                _min = Z
                if self.challenger:
                    self.challenger_id = self.non_opt_arms[i]
        return _min

    def is_done(self):
        # Update set of maximum
        self.update_maximum_arms()
        self.done = False
        if self.opt_arms.shape[0] == 1:
            if self._compute_score() > self.rate():
                self.done = True

        return self.done

    def rate(self):
        t = self.t + 1
        #import pdb
        #pdb.set_trace()
        if self._stoppingrule == 0:  # Original stopping rule
            return np.log(2 * t * (self.K - 1) / self.delta)
        elif self._stoppingrule == 1:  # Unproved
            return np.log((np.log(t) + 1) / self.delta)
        else:  # new stopping rule
            t1 = np.log(np.log(t / 2) + 1)
            t2 = np.log((self.K - 1) / self.delta) / 2
            return 6 * t1 + 2 * Tconf(t2)

    def play(self):

        # Initialization phase
        Lopt = self.opt_arms.shape[0]
        if self.t < self.K:
            action = self.t
        # If multiple optimal arms pick one uniformly
        elif Lopt > 1:
            #print(self.opt_arms)
            action = np.random.choice(self.opt_arms.flatten())
        # Exploration/Tracking phases
        else:
            # Build exploration set
            self.threshold = max(np.sqrt(self.t) - self.K / 2, 0)
            U_t = self.build_exploration_set()

            if len(U_t) > 0:
                # Forced exploration
                action = np.argmin(self.N_x_g_h, axis=None)
            else:
                # Exploitation
                self.update_optimal_weights()
                if self.challenger:
                    # Best arms stats
                    nBest = self.N_x_g_h[self.opt_arms]
                    nChallenger = self.N_x_g_h[self.challenger_id]
                    wBest = self.w[self.opt_arms]
                    wChallenger = self.w[self.challenger_id]
                    condition = nBest / (nChallenger + nBest) < wBest / (
                        wBest + wChallenger)
                    if nBest / (nChallenger + nBest) < wBest / (
                            wBest + wChallenger):
                        action = self.opt_arms.flatten()[0]
                    else:
                        action = self.challenger_id[0]
                else:
                    action = np.argmax(self.t * self.w - self.N_x_g_h)
        return self.get_index(action)

    def update_optimal_weights(self):
        """ Returns T*(mu) and w*(mu) """
        # Check if the optimal arm is unique, if not we select uniformly
        # amongst the optimal
        optimal_arms = self.opt_arms
        Lopt = self.opt_arms.shape[0]
        if Lopt > 1:
            self.w *= 0.0
            self.w[self.opt_arms] = 1 / Lopt
        else:
            indexes_sort = np.argsort(self.emp_means)[::-1]
            invert_index = np.zeros(self.K, dtype=int)
            invert_index[indexes_sort] = np.arange(self.K)
            dist, T = solveFBern(
                np.ascontiguousarray(np.sort(self.emp_means)[::-1]))
            self.w = dist[invert_index]
            self.last_lb = T


class MTTrackAndStopD(PlayerStrategyIdentification):
    # Type 0 applies Track and stop to every task,
    # Type 1 applies track and stop by identifying a g first in a random task
    # and then use this knowledge
    def __init__(self, X, G, H, delta, challenger=False, stopping_rule=0):
        PlayerStrategyIdentification.__init__(self, X, G, H, flatten=True)
        self.name = 'MT - TrackAndStop - D Tracking'
        self.delta = delta
        self.K = self.n_X * self.n_H * self.n_G
        self.challenger = challenger
        self.stopping_rule = stopping_rule

        initial_task = np.random.choice(X)
        self.task_list = [initial_task]
        for i in range(X):
            if i != initial_task:
                self.task_list.append(i)

        self.make_agent = lambda nG: TrackAndStopD(1, nG, H, delta, challenger,
                                                   stopping_rule)
        self.agent = self.make_agent(G)
        self.current_task = self.task_list.pop(0)
        self.estimated_g = 0
        self.first_task = True
        self._best_arms = [-1] * (X + 1)
        self.last_lb = 0.
        self.w = None

    def update(self, action, obs):
        self.update_means(action, obs)
        x, g, h = action
        self.agent.update((0, g, h), obs)
        self.last_lb = self.agent.last_lb
        self.w = self.agent.w
        _done = self.agent.is_done()
        if _done is True:
            if self.first_task is True:
                self.estimated_g = self.agent.best_arms()[0]
                self.first_task = False
                self._best_arms[0] = self.estimated_g
            self._best_arms[self.current_task + 1] = self.agent.best_arms()[1]
            return True

        return False

    def best_arms(self):
        return self._best_arms

    def is_done(self):
        return not self.task_list

    def play(self):
        _, g, h = self.agent.play()
        if not self.first_task:
            return (self.current_task, self.estimated_g, h)
        return (self.current_task, g, h)


class OSRLSC(PlayerStrategyIdentification):
    def __init__(self, X, G, H, delta_g):
        PlayerStrategyIdentification.__init__(self, X, G, H, flatten=False)
        self.name = 'OSRL-SC'
        self.delta_g = delta_g
        self.K = self.n_X * self.n_H * self.n_G
        self.w = np.zeros((X, G, H), order='C', dtype=np.float64)
        self.g_global_leader = np.uint16(0)
        self.g_task_leader = -1 * np.ones(self.n_X, order='C', dtype=np.uint16)
        self.task_best_h = np.zeros(self.n_X, order='C', dtype=np.uint16)
        self.mu_xg = np.zeros((self.n_X, self.n_G),
                              order='C',
                              dtype=np.float64)
        self.mu_g = np.zeros(self.n_G, order='C', dtype=np.float64)
        self.unique_leader = False

        self.threshold = np.float64(0.)
        self.compute_allocation = MTArmIdentificationLB_GTerm((X, G, H),
                                                              discount=0.5,
                                                              sigma=1e5)

        self.nq = 40
        self.lr = 5
        self.last_lb = 0.

    def build_exploration_set(self):
        return np.argwhere(self.N_x_g_h < self.threshold)

    def get_index(self, x):
        return np.unravel_index(x, (self.n_X, self.n_G, self.n_H))

    def is_done(self):
        self.done = False
        if self.unique_leader:
            # score = self.t * self.last_lb

            score = self.compute_allocation.evaluate_vector(
                self.emp_means,
                self.g_global_leader,
                q=self.N_x_g_h.flatten() / self.t) * self.t
            threshold = beta_threshold(self.N_x_g_h, self.n_X, self.n_G,
                                       self.n_H, self.K, self.n_G,
                                       self.delta_g)
            # print('[{}] SCORE: {} - th:{}'.format(self.t, score, threshold))
            if score > threshold:
                self.done = True
        if self.t > 20000:
            self.done = True
        return self.done

    def rate(self):
        return beta_threshold(self.N_x_g_h, self.n_X, self.n_G, self.n_H,
                              self.n_G * self.n_H, self.n_G, self.delta_g)

    def play(self):
        # Initialization phase
        if self.t < self.K:
            action = self.t
        # Exploration/Tracking phases
        else:
            # Build exploration set
            self.threshold = max(np.sqrt(self.t) - self.K / 2, 0.)
            U_t = self.build_exploration_set()
            if len(U_t) > 0 or not self.unique_leader:
                # Forced exploration
                action = np.argmin(self.N_x_g_h, axis=None)
            else:
                if self.nq > self.K:
                    self.nq = 0
                    lr = klBern(np.max(self.emp_means), np.min(self.emp_means))
                    Q, q, t = self.compute_allocation(
                        self.emp_means, self.g_global_leader, lr=self.lr)
                    self.lr = max(5., 1 / lr)  # Increase in speed
                    self.last_lb = Q

                    self.w = np.reshape(q, (self.n_X, self.n_G, self.n_H))
                self.nq += 1

                action = np.argmax(self.t * self.w - self.N_x_g_h)
        return self.get_index(action)

    def _amax(self, a):
        return np.argwhere(a == np.amax(a)).flatten()

    def update(self, action, obs):
        self.update_means(action, obs)

        # Compute hat mu(x,g)
        if self.t > self.K:
            self.unique_leader = True

            for x in range(self.n_X):
                for g in range(self.n_G):
                    task_best_gh = self._amax(self.emp_means[x, g, :])
                    if len(task_best_gh) > 1:
                        self.unique_leader = False
                        task_best_gh = [0]
                    self.mu_xg[x, g] = self.emp_means[x, g, task_best_gh[0]]

                # Compute g*(x)
                task_best_g = self._amax(self.mu_xg[x])
                if len(task_best_g) > 1:
                    self.unique_leader = False
                    task_best_g = [0]
                self.g_task_leader[x] = task_best_g[0]

            # Compute g*
            for g in range(self.n_G):
                self.mu_g[g] = np.sum(self.mu_xg[:, g])
            best_g = self._amax(self.mu_g)
            if len(best_g) > 1:
                self.unique_leader = False
                self.g_global_leader = 0
            else:
                self.g_global_leader = best_g[0]

            if self.unique_leader:
                if (self.g_global_leader == self.g_task_leader).all():
                    self.unique_leader = True
                else:
                    self.unique_leader = False

        return self.is_done()
