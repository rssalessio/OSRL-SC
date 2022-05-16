#! /usr/bin/env python
# -*- coding: utf-8 -*-

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

from __future__ import division, print_function  # Python 2 compatibility
from libc.math cimport log, sqrt, exp, fabs
import numpy as np
from scipy.special import lambertw
cimport numpy as np

cdef double eps = 1e-8 #: Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]



# --- Generic routines

cpdef double KL(double[::1] a, double[::1] b):
    cdef Py_ssize_t n = a.shape[0]
    cdef double ret = 0.
    for i in range(n):
        if a[i] > eps:
            ret += a[i] * log(a[i] / b[i])
    return ret

cdef inline double bisection(f, double xMin, double xMax):
    cdef double l = xMin
    cdef double u = xMax
    cdef double sgn = f(xMin)
    cdef double m = (u + l) / 2.
    while u - l > eps:
        if f(m) * sgn > 0.:
            l = m
        else:
            u = m
        m = (u + l) / 2.
    return m

# --- Simple Kullback-Leibler divergence for known distributions
cpdef double[:] klBern_vec_c(double[::1] x, double[::1] y):
    cdef Py_ssize_t n = x.shape[0]
    cdef double[:] v = x
    for i in range(n):
        a = min(max(x[i], eps), 1 - eps)
        b = min(max(y[i], eps), 1 - eps)
        v[i] = a * log(a / b) + (1 - a) * log((1 - a) / (1 - b))
    return v


cpdef double klBern(double x, double y) nogil:
    """ Kullback-Leibler divergence for Bernoulli distributions. """
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))


cpdef double klGauss(double x, double y, double sig2x=0.25, double sig2y=0.25):
    """ Kullback-Leibler divergence for Gaussian distributions """
    if - eps < (sig2y - sig2x) < eps:
        return (x - y) ** 2 / (2. * sig2x)
    else:
        return (x - y) ** 2 / (2. * sig2y) + 0.5 * ((sig2x/sig2y)**2 - 1 - log(sig2x/sig2y))

# --- KL functions, for the KL-UCB policy

cpdef double klucb(double x, double d, kl,
        double upperbound,
        double lowerbound=float('-inf'),
        double precision=1e-6,
        int max_iterations=50
    ):
    """ The generic KL-UCB index computation. """
    cdef double value = max(x, lowerbound)
    cdef double u = upperbound
    cdef int _count_iteration = 0
    while _count_iteration < max_iterations and u - value > precision:
        _count_iteration += 1
        m = (value + u) / 2.
        if kl(x, m) > d:
            u = m
        else:
            value = m
    return (value + u) / 2.


cpdef double klucbBern(double x, double d, double precision=1e-6):
    """ KL-UCB index computation for Bernoulli distributions, using :func:`klucb`.
    """
    cdef double upperbound = min(1., klucbGauss(x, d, sig2x=0.25))  # variance 1/4 for [0,1] bounded distributions
    # upperbound = min(1., klucbPoisson(x, d))  # also safe, and better ?
    return klucb(x, d, klBern, upperbound, precision)


cpdef double klucbGauss(double x, double d, double sig2x=0.25, double precision=0.):
    """ KL-UCB index computation for Gaussian distributions. """
    return x + sqrt(2 * sig2x * d)


# --- Best arm identification routines

cdef inline double IBern(const  double alpha, const double mu_1, const double mu_2):
    if alpha <= 0. or alpha >= 1.:
        return 0.
    else:
        mid_mu = alpha * mu_1 + (1 - alpha) * mu_2
        return alpha * klBern(mu_1, mid_mu) + (1 - alpha) * klBern(mu_2, mid_mu)


cdef inline double gBern(const double x, const double mu_1, const double mu_k):
    if abs(x) < eps or x == np.inf:
        return 0.
    return ((1. + x) * IBern(1. / (1. + x), mu_1, mu_k))


cdef inline double mux(const double a, const double b, const double c, const double d):
    return (a * c + b * d)/(c + d)


cdef double xOfy(double y, Py_ssize_t k, const double *mu):
    """ finds x such that g_k(x)=y """
    if abs(mu[0]-mu[k]) < eps:
        return -y
    g = lambda x: gBern(x, mu[0], mu[k]) - y
    cdef double x_max = 1.0

    while g(x_max) < 0.:
        x_max *= 2.0
        if x_max == np.inf:
            return klBern(mu[0], mu[k]) - y
    return bisection(g, 0., x_max)


cdef double solveFy(const double y, const double y_max, const double *mu, Py_ssize_t K):
    """ Computes F(y) - 1 """
    if abs(y) < eps:
        return 0.
    elif abs(y-y_max) < eps:
        return np.inf

    cdef double c = 0.
    cdef double klb = 0.
    for k in range(1, K):
        c = mux(mu[0], mu[k], 1, xOfy(y, k-1, mu))
        klb += klBern(mu[0], c) / klBern(mu[k], c)
    return klb


def solveFBern(double[::1] mu):
    """ mu is  avector of means, sorted such that mu_1 > mu_2 """
    cdef double y_max = klBern(mu[0], mu[1])
    F = lambda y: solveFy(y, y_max, &mu[0], mu.shape[0]) - 1.0

    y = bisection(F, 0., y_max)
    cdef np.ndarray[double, ndim=1, mode="c"] x = np.zeros(mu.shape[0])
    cdef double s = 1.0
    x[0] = 1.0
    for k in range(1, mu.shape[0]):
        x[k] = xOfy(y, k, &mu[0])
        s += x[k]
    return x / s, x[0] * y


# Vectorized numpy is as fast
cdef void c_expkl(Py_ssize_t n, double *y, double *val, double *gradval, double lmbd):
    cdef int i = 0
    cdef double x = 0.
    for i in range(n):
        x = exp(lmbd * y[i])
        gradval[i] = -lmbd * (1 - x / (1 - x) + 2 * x * (log(1 - x) - y[i]))
        val[i] = (1 - 2 * x) * (np.log(1 - x) - lmbd * y[i])

cpdef expkl(double[::1] y, double lmbd):
    cdef np.ndarray[double, ndim=1] val = np.zeros(y.shape[0], dtype=np.float64)
    cdef np.ndarray[double, ndim=1] gradval = np.zeros(y.shape[0], dtype=np.float64)
    c_expkl(y.shape[0], &y[0], &val[0], &gradval[0], lmbd)
    return val, gradval


# Compute DX C

cpdef double _compute_rho(const double best_mean, const double[:] mean_vector, const short nH) nogil:
    cdef double rho = 0.
    cdef double delta = 0.
    for h in range(nH):
        delta = (best_mean - mean_vector[h])
        if fabs(mean_vector[h] - best_mean) > eps:
            rho += delta / klBern(mean_vector[h], best_mean)
    return rho

cdef double _const1 = 3.8017702851374455  # hinv(1/log(3/2))
cdef double _const2 = 1.1908474830306905  # ln(2 zeta(2)) = ln(pi^3/3)
cdef double _const3 = -0.90272045571788   # ln(ln(3/2))

cpdef double Tconf(double x):
    # T(x):R+ -> R+
    # T(x)=2\tilde h_{3/2} ((h^{-1}(1+x)+ln(2pi^2/6)/2)
    return 2 * h_3_2((hinv(1+x) + _const2)/2)

cdef double hinv(double x):
    # hinv(x) = -W_{-1}(-e^(-x))
    return -lambertw(z=-exp(-x), k=-1).real

cdef double h_3_2(double x):
    # h^{-1}(1/ln(3/2)) = -W_{-1}(-e^(-1/ln(3/2)))
    # Which is approx 3.8018
    if x >= _const1:
        h = hinv(x)
        return h * exp(1/h)
    else:
        return 3 * (x - _const3) / 2

cdef double beta_1(double *N_t, int Nx, int Ng, int Nh):
    cdef double b1 = 0.
    for i in range(Nx * Ng * Nh):
        b1 += 3 * log(1+log(N_t[i]))

cdef double beta_2(double S, double G, double delta_G):
    return S * Tconf(log((G-1) / delta_G) / S)

cpdef double beta_threshold(double[:,:,:] N_t, int Nx, int Ng, int Nh, double S, double G, double delta_G):
    return beta_1(&N_t[0,0,0], Nx, Ng, Nh) + beta_2(S, G, delta_G)
