import pandas as pd
import random
import scipy.stats as stats
import numpy
import copy
from scipy.stats import poisson
import math

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

def multinomCDF(G, k, p, tau_p):
    return numpy.exp(multinomCDF_log(G, k, p, tau_p ));

def multinomial_icdf_most_likely(G, k, p, a, tau):
    tau_p = [k] + list(tau);
    temp = copy.copy(tau_p)
    cdf = multinomCDF(G, k, p, tau_p)
    new_cdf = 0;
    initial = 1;
    not_fulfilled = 0;

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


def multinomial_icdf_most_unlikely(G, k, p, a, tau):
    tau_p = [k] + list(tau);
    temp = copy.copy(tau_p)
    cdf = multinomCDF(G, k, p, tau_p)
    new_cdf = 0;
    initial = 1;
    not_fulfilled = 0;

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
            if(new_cdf >= a and new_cdf <= cdf):
                tau_p = copy.copy(temp);
                cdf = multinomCDF(G, k, p, tau_p);
        if(new_cdf >= a or cdf >= a):
            if(not_fulfilled == 1):
                tau_p = copy.copy(temp);
                not_fulfilled = 0;
            else:
                temp[i+1] = temp[i+1]-1
                not_fulfilled = 0
        else:
            not_fulfilled = 1
    return tau_p

"""
example: try multinomial_icdf_most_most_likely and safe it as table in html
"""

p = [0.4, 0.3, 0.2, 0.1];
a = 0.1;
k = 100;
positions = numpy.array(list(range(k))) + 1;
least_items = [];
tau = numpy.zeros(len(p)-1);

for i in positions:
    tau_p = multinomial_icdf_most_likely(len(p), i, p , a, tau)[1:]
    least_items.append(numpy.array(tau_p));
    tau = copy.copy(tau_p);
for i in range (k):
    print( i+1,"  : ",least_items[i])
    test = numpy.append([i+1],least_items[i])
    print ("CDF: ",multinomCDF(4, i+1, p, test))
    test = []
df = pd.DataFrame(data=(numpy.array(least_items)).astype(int))
df.columns = p[1:]
df.index = numpy.array(range(k))+1
df.to_html("Minimum target_tables/minimum_target_table_most_likely.html")
