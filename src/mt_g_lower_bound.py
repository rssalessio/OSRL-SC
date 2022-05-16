#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import pyximport
_ = pyximport.install()
from src.math_func_cython import klBern

eps = torch.tensor(1e-10, dtype=torch.float64, requires_grad=False)


def kltorch(x, y):
    a = x * ((x / y) + eps).log()
    b = (1 - x) * (eps + (1 - x) / (1 - y)).log()
    return a + b


zero = torch.tensor(0, dtype=torch.float64, requires_grad=False)


class FindAlternativeModel(object):
    def __init__(self, X, G, H, torch=False):
        self.X = X
        self.G = G
        self.H = H
        self.Gq = np.diag([1.] * (self.G * self.H))
        self.get_id = lambda x, g, h: x * self.G * self.H + g * self.H + h

    def __call__(self, mu, q, x0, gbar, hbar):
        U = self.build_U(mu, q, x0, gbar, hbar)
        return self.mix(S=U, mu=mu, q=q, x=x0), U

    def build_U(self, mu, q, x, gbar, hbar):
        U = set()
        U.add((gbar, hbar))
        for g in range(self.G):
            for h in range(self.H):
                M = self._build_M(mu, x, g, h, gbar, hbar)
                rhs = self.mix(M, mu, q, x)
                if mu[self.get_id(x, g, h)] + eps >= rhs:
                    U.add((g, h))
        return U

    def mix(self, S, mu, q, x):
        N = 0.
        D = 0.
        for el in S:
            g, h = el
            _id = self.get_id(x, g, h)
            N += q[_id] * mu[_id]
            D += q[_id]
        return N / D

    def _build_M(self, mu, x, g, h, gbar, hbar):
        M = set()
        M.add((gbar, hbar))
        _id = self.get_id(x, g, h)
        for gp in range(self.G):
            for hp in range(self.H):
                if mu[self.get_id(x, gp, hp)] >= mu[_id]:
                    M.add((gp, hp))
        return M


class MTArmIdentificationLB_GTerm(object):
    def __init__(self, shape, lr=5, T=2000, discount=0.5, sigma=None):
        self.X = shape[0]
        self.G = shape[1]
        self.H = shape[2]
        self.K = self.X * self.G * self.H
        self.T = T
        self.lr = lr
        self.calls = 1
        self.lr_lmbd = lambda t: self.lr / 7 * np.sqrt(2 * np.log(self.G) / (
            t + 1))
        self.lr_theta = lambda t: self.lr / 7 * np.sqrt(2 * np.log(self.H) / (
            t + 1))
        self.lr_q = lambda t: self.lr * np.sqrt(2 * np.log(self.K) / (t + 1))
        self.get_index = lambda x, g, h: x * self.G * self.H + g * self.H + h
        self.alternative_problem = FindAlternativeModel(self.X, self.G, self.H)
        self._reset()
        self.discount = discount
        self.sigma = None

    def _reset(self, soft=False):
        # If number of G arms is less than 2 we don't need to reset
        if self.G < 2:
            return

        if soft:
            discount = self.discount
            self.q = torch.tensor(
                discount * self.q.detach().numpy() + (1 - discount) / self.K,
                dtype=torch.float64,
                requires_grad=True)
            self.lmbd = torch.tensor(
                discount * self.lmbd.detach().numpy() +
                (1 - discount) / (self.G - 1),
                dtype=torch.float64,
                requires_grad=True)
            self.theta = torch.tensor([[
                discount * self.theta[x][g].detach().numpy() +
                (1 - discount) / self.H for g in range(self.G - 1)
            ] for x in range(self.X)],
                                      dtype=torch.float64,
                                      requires_grad=True)

        else:
            # Reinitialize tensors
            self.q = torch.tensor(
                [1 / self.K] * self.K, dtype=torch.float64, requires_grad=True)

            self.lmbd = torch.tensor(
                [1 / (self.G - 1)] * (self.G - 1),
                dtype=torch.float64,
                requires_grad=True)

            if self.H > 1:
                self.theta = torch.tensor([[[1 / (self.H)] * (self.H)
                                            for g in range(self.G - 1)]
                                           for x in range(self.X)],
                                          dtype=torch.float64,
                                          requires_grad=True)

    def compute_loss(self, gstar):
        L = 0.
        glambda = 0
        for g in range(self.G):
            if g == gstar:
                continue
            inner_sum = 0.
            for x in range(self.X):
                for h in range(self.H):
                    with torch.no_grad():
                        U = self.alternative_problem.build_U(
                            self.mu, self.q, x, g, h)
                    mbar = self.alternative_problem.mix(
                        S=U, mu=self.mu, q=self.q, x=x)
                    for el in U:
                        gu, hu = el[0], el[1]
                        a = self.get_index(x, gu, hu)
                        inner_sum += (self.theta[x][glambda][h] * self.q[a] *
                                      kltorch(self.mu[a], mbar))

                    # Add check if H < 2 and if not torch.isnan(ltemp) and not torch.isinf(ltemp):
            L += inner_sum * self.lmbd[glambda]
            glambda += 1

        return L

    def _compute_rho(self, means, gstar):
        # Compute rho term
        Q = 0.
        qq = None
        t = 0

        if self.G < 2:
            return Q, qq, t

        self.mu = torch.tensor(
            means.flatten(), dtype=torch.float64, requires_grad=False)
        self.mu = torch.min(torch.max(self.mu, eps), 1 - eps)
        if self.calls > 1:
            self._reset(soft=True)
        prev_Q = 0.
        self.prev_q = None
        while t < self.T:
            Q = self.compute_loss(gstar)
            if self.sigma:
                Q += torch.sum(self.q**2) / (2 * self.sigma)
            Q.backward()
            if t > 1 and torch.isclose(Q, prev_Q):  #, rtol=1e-6):
                break
            prev_Q = Q
            with torch.no_grad():
                self.prev_q = self.q.detach().numpy()
                if self.H > 1:
                    for x in range(self.X):
                        for g in range(self.G - 1):
                            self.theta[x][g] *= (-self.lr_theta(t) *
                                                 self.theta.grad[x][g]).exp()
                            self.theta[x][g] /= self.theta[x][g].sum()
                    self.theta.grad.zero_()

                # Update lambda
                self.lmbd *= (-self.lr_lmbd(t) * self.lmbd.grad).exp()
                self.lmbd /= self.lmbd.sum()
                self.lmbd.grad.zero_()

                # Update q
                qqgrad = self.q.grad.clamp(max=6)
                qqgrad[qqgrad != qqgrad] = 0.
                self.q *= (self.lr_q(t) * qqgrad).exp()
                self.q /= self.q.sum()
                self.q.grad.zero_()
            t += 1
        return Q.detach().numpy(), self.q.detach().numpy(), t

    def __call__(self, means, gstar, lr=5):
        self.lr = lr
        self.calls += 1
        return self._compute_rho(means, gstar)

    def evaluate_vector(self, means, gstar, q):
        mu = np.minimum(np.maximum(means.flatten(), 1e-10), 1 - 1e-10)
        best_g = np.inf

        for g in range(self.G):
            if g == gstar:
                continue
            outer_sum = 0.
            for x in range(self.X):
                best_h = np.inf
                for h in range(self.H):
                    U = self.alternative_problem.build_U(mu, q, x, g, h)
                    mbar = self.alternative_problem.mix(S=U, mu=mu, q=q, x=x)
                    inner_sum = 0.
                    for el in U:
                        gu, hu = el[0], el[1]
                        a = self.get_index(x, gu, hu)
                        inner_sum += q[a] * klBern(mu[a], mbar)
                    if inner_sum < best_h:
                        best_h = inner_sum

                if best_h < np.inf:
                    outer_sum += best_h
            if outer_sum < best_g:
                best_g = outer_sum
        return best_g
