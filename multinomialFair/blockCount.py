import random
import scipy.stats as stats
import pandas as pd
import numpy
import copy

from scipy.stats import poisson
import math

def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

@timer_func
def multinomCDF_log(G, k, p, tau_p):
    s = float(k);
    log_cdf = -poisson.logpmf(k,s);
    gamma1 = 0.0;
    gamma2 = 0.0;
    sum_s2 = 0.0;
    sum_mu = 0.0;

    # P(W=k)
    for i in range(0,G):
        sp = s*p[i];

        pcdf = poisson.cdf(tau_p[i],sp);
        log_cdf += numpy.log(pcdf);

        mu = sp*(1-poisson.pmf(tau_p[i],sp)/pcdf);
        s2 = mu-(tau_p[i]-mu)*(sp-mu);

        mr = tau_p[i];
        mf2 = sp*mu-mr*(sp-mu);

        mr *= tau_p[i]-1;
        mf3 = sp*mf2-mr*(sp-mu);

        mr *= tau_p[i]-2;
        mf4 = sp*mf3-mr*(sp-mu);

        mu2 = mf2+mu*(1-mu);
        mu3 = mf3+mf2*(3-3*mu)+mu*(1+mu*(-3+2*mu));
        mu4 = mf4+mf3*(6-4*mu)+mf2*(7+mu*(-12+6*mu))+mu*(1+mu*(-4+mu*(6-3*mu)));

        gamma1 += mu3;
        gamma2 += mu4-3*s2*s2;
        sum_mu += mu;
        sum_s2 += s2;
    sp = numpy.sqrt(sum_s2);
    gamma1 /= sum_s2*sp;
    gamma2 /= sum_s2*sum_s2;

    x = (k-sum_mu)/sp;
    x2 = x*x;

    PWN = (-x2/2
    +numpy.log(1+gamma1/6*x*(x2-3)+gamma2/24*(x2*x2-6*x2+3)
    +gamma1*gamma1/72*(((x2-15)*x2+45)*x2-15))
    -numpy.log(2*math.pi)/2 -numpy.log(sp));

    log_cdf += PWN;
    return log_cdf;
@timer_func
def multinomCDF(G, k, p, tau_p):
    return numpy.exp(multinomCDF_log(G, k, p, tau_p ));

@timer_func
def multinomial_icdf_continuous(G, k, p, a, tau):
    tau_p = [k] + list(tau);
    temp = copy.copy(tau_p)
    cdf = multinomCDF(G, k, p, tau_p)
    new_cdf = 0;
    initial = 1;

    if(cdf > a):
        return tau_p;
    for i in range(len(tau_p)-1):
        temp[i+1] = temp[i+1]+1;
        if(initial == 1):
            tau_p = copy.copy(temp);
            cdf = multinomCDF(G, k, p, tau_p);
            initial = 0;
        else:
            new_cdf = multinomCDF(G, k, p, temp)
            if(new_cdf >= a and new_cdf >= cdf):
                tau_p = copy.copy(temp);
                cdf = multinomCDF(G, k, p, tau_p);
        if(new_cdf >= a or cdf >= a):
            temp[i+1] = temp[i+1]-1
    return tau_p

@timer_func
def get_minimum_targets(categories, p, alpha, k):
    positions = numpy.array(list(range(k))) + 1;
    minimum_targets = [];
    tau = numpy.zeros(len(categories)-1);
    block_sizes = []
    count=0

    for i in positions:
        tau_p = multinomial_icdf_continuous(len(p), i, p , alpha, tau)[1:]

        if(all(tau_p[j]==tau[j] for j in range(len(tau)))):
            count = count + 1
        else:
            block_sizes.append(count)
            count =1
        minimum_targets.append(numpy.array(tau_p));
        print ("tau_p: ",tau_p, " cdf: ",multinomCDF(len(categories),i, p, [i]+tau_p))
        tau = copy.copy(tau_p);

    block_sizes.append(count)
    return minimum_targets, block_sizes, len(block_sizes)
