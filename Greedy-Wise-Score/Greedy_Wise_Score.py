import matplotlib
import operator
import random
import scipy.stats as stats
import pandas as pd
import numpy
import copy
import json
import ast
from anytree import PreOrderIter
import math
import itertools
import scipy.stats as stats
import matplotlib.pyplot as pl
from operator import itemgetter, attrgetter, methodcaller
from scipy.stats import poisson
from docx import Document
from docx.text.run import Font, Run
from docx.dml.color import ColorFormat
from docx.shared import RGBColor
from time import time
from anytree import AnyNode
from anytree.exporter import JsonExporter



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


def find_target(minimum_targets, count, categories):
    for i in range(len(minimum_targets)):
        if(minimum_targets[i] > count[i+1]):
                return categories[i+1];

def find_achieved_target(minimum_targets, count, categories):
    for i in range(len(minimum_targets)):
        if(minimum_targets[i] < count[i+1]):
                return categories[i+1];

def get_num_categories(attributeNamesAndCategories):
    num_categories = 1
    for i in attributeNamesAndCategories.items():
        num_categories *= i[1]
    return num_categories

def determineGroups(attributeNamesAndCategories):
    elementSets = []
    groups = []
    for attr, cardinality in attributeNamesAndCategories.items():
        elementSets.append(list(range(0, cardinality)))

    groups = list(itertools.product(*elementSets))
    return groups


def separate_groups(data_set, categories, attributeItems):
    num_categories = len(categories)
    separateByGroups = [[] for _ in range(num_categories)]

    for i in data_set:
        categorieList = []
        for j in attributeItems:
            categorieList.append(i[j[0]])
        separateByGroups[categories.index(tuple(categorieList))].append(i)
        categorieList = []
    return separateByGroups




'''
The multinomial CDF function is the implementation of Levin's "Representaion of Multinomial Cumulative Distribution Function"

G:= Number of total groups (including non-protected)
k:= Position
p:= Array of probabilities of each group to be selected
tau_p:= Array of number of protected items

EXAMPLE:
Groups: ["White"(NP), "Black"(P), "Asian"(P), "Hispanic"(P)]
k: 30
p: [0.4, 0.3, 0.2, 0.1]
tau_p: [k, 10, 5, 3]
The ranking is fair if multinomCDF(4, 30, p, tau_p) > a = 0.1
'''

def multinomCDF_log(G, k, p, tau_p):
    s = float(k);
    log_cdf = -poisson.logpmf(k,s);
    gamma1 = 0.0;
    gamma2 = 0.0;
    sum_s2 = 0.0;
    sum_mu = 0.0;

    # P(W=k)
    for i in range(0,G):
        i = i-1
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


def child_generator_4(G, k, p ,a, tau):
    tau_p = [k] + list(tau) ;
    temp = copy.copy(tau_p)
    old_cdf = multinomCDF(G, k, p, tau_p)
    new_cdf = 0;
    initial = 1;
    not_fulfilled = 0;
    diz = {}
    ls = []
    i = 0
    if old_cdf > a:


        return  tau

    for i in range(len(tau_p)-1):
        temp[i+1] = temp[i+1]+1;
        if(initial == 1):
            tau_p = copy.copy(temp);
            cdf = multinomCDF(G, k, p, tau_p);
            initial = 0;
            if (cdf > a):

                ls.append(tau_p[1:])


        else:
            new_cdf = multinomCDF(G, k, p, temp)
            if(new_cdf > a):
                tau_p = copy.copy(temp);
                cdf = multinomCDF(G, k, p, tau_p)

                ls.append(tau_p[1:])

        if(new_cdf >= a or cdf >= a or cdf <= a or new_cdf< a):
            temp[i+1] = temp[i+1]-1


    return ls

# Il dataset in entrata sarà un dict-like e avrà come attributi il punteggio,
# la k posizione pre re_rank, e a quale gruppo protetto appartiene
@timer_func
def GWS(data_set, p, alpha, k_th, attributeQuality, attributes):
    protected_item = [key for key,value in attributes.items() ][0]
    categories = determineGroups(attributes)
    num_categories = len(categories)
    parent = list(numpy.zeros(num_categories-1))
    separateByGroups = [[] for _ in range(num_categories)]
    attributeItems = attributes.items()
    dataset =[]
    separateByGroups = separate_groups(data_set, categories, attributeItems)

    for i in range(num_categories):
        separateByGroups[i] = sorted(separateByGroups[i], key=lambda item: item[attributeQuality],reverse=True)

    for k in range(1,k_th):
        child = child_generator_4(num_categories, k, p, alpha, parent)
        child_ls = []

        if numpy.array_equal(child, parent)== True:
            dataset.append(separateByGroups[0].pop(0))

        else:

            for i in range(len(child)):
                diz = {}
                diz['Index_Group'] = child[i]
                diz['Item'] = separateByGroups[numpy.nonzero(list(map(operator.sub, child[i], parent)))[0][0]+1][0]

                child_ls.append( diz)


            dataset.append( separateByGroups[max(child_ls, key=lambda x: x['Item'][attributeQuality])['Item'][protected_item]].pop(0))
            parent = max(child_ls, key=lambda x: x['Item'][attributeQuality])['Index_Group']


    return dataset
