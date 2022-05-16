#! /usr/bin/env python
# -*- coding: utf-8 -*-

from math import log
import numpy as np
from src.utils import RewardBernoulliDistribution


def klBern(x, y):
    """ KL divergence for bernoulli distributions """
    eps = 1e-15
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)

    return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))


class Scenario(object):
    """ Scenarios ID """
    ScenarioA = 1
    ScenarioB = 2
    ScenarioC = 3


class GenerateTasks(object):
    def __init__(self, X=2, G=4, H=9, scenario=Scenario.ScenarioA, x_tilde=0, goptimal=0):
        # Number of arms
        self.X = X
        self.G = G
        self.H = H

        self.gap = (1 - 0.1) / H
        # Scenario considered
        self.scenario = scenario
        self.x_tilde = x_tilde

        self.generate_pmf()
        self.generate_means()
        self.reward = RewardBernoulliDistribution
        self.goptimal = goptimal
        self.opt_arms = [
            np.argmax(self.means[x][self.goptimal]) for x in range(self.X)
        ]
        self.opt_arms.insert(0, self.goptimal)

    def generate_pmf(self):
        ''' Function used to generate alpha(x) '''
        self.pmf = np.array([0.] * self.X)

        if self.scenario != Scenario.ScenarioA:
            self.pmf[
                self.
                x_tilde] = 0.9 if self.scenario == Scenario.ScenarioB else 0.1
        else:
            self.pmf[self.x_tilde] = 1 / self.X

        self.pmf[:self.x_tilde] = (1 - self.pmf[self.x_tilde]) / (self.X - 1)
        self.pmf[self.x_tilde +
                 1:] = (1 - self.pmf[self.x_tilde]) / (self.X - 1)
        return self.pmf

    def generate_means(self):
        """ Compute means for each (x,g,h) """
        self.means = np.zeros((self.X, self.G, self.H))
        for x in range(self.X):
            if self.scenario == Scenario.ScenarioA:
                self.means[x] = self.generate_means_tasks(x_tilde=False)
            else:
                self.means[x] = self.generate_means_tasks(
                    x_tilde=True if x == self.x_tilde else False)
        return self.means

    def generate_means_tasks(self, x_tilde):
        if not x_tilde:
            means = np.zeros((self.G, self.H))
            means[0] = np.linspace(0.9, 0.1, self.H)
            for g in range(1, self.G):
                means[g] = np.clip(means[0] - self.H * self.gap / 2, 1e-2, 1.)
        else:
            means = 1e-5 * np.ones((self.G, self.H))
            means[0][0] = 0.9
        return means

    def lower_bound(self):
        """ Compute asymptotic lower bound """
        first_term = 0.
        second_term = 0
        for x in range(self.X):
            for h in range(self.H):
                if h == self.opt_arms[x + 1]:
                    continue
                delta = (self.means[x][self.goptimal][self.opt_arms[x + 1]] -
                         self.means[x][self.goptimal][h])
                first_term += delta / klBern(
                    self.means[x][self.goptimal][h],
                    self.means[x][self.goptimal][self.opt_arms[x + 1]])

        for g in range(self.G):
            if g == self.goptimal:
                continue
            temp = [0.] * self.X
            for x in range(self.X):
                for h in range(self.H):
                    delta = (self.means[x][self.goptimal][self.opt_arms[x + 1]]
                             - self.means[x][g][h])
                    temp[x] += delta / klBern(
                        self.means[x][g][h],
                        self.means[x][self.goptimal][self.opt_arms[x + 1]])
            second_term += min(temp)
        return first_term + second_term
