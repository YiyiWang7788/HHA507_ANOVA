#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 21:08:06 2021

@author: yiyiwang
"""

import pandas as pd

### This A Stroke Prediction Dataset 
df = pd.read_csv('/Users/yiyiwang/Downloads/healthcare-dataset-stroke-data.csv')

### List of all variables in the dataset

listdf=list(df)
listdf

### Variable to be used for 1-way ANOVA test
""" the dependent variable (continous value) = 'stroke'
#Independent variable 1 (categorical value) = 'age'
#Independent variable 2 (categorical value) = 'bmi'
#Independent variable 3 (categorical value) = 'avg_glucose_level'
"""

### Rename the variables columns
df = df.rename(columns={'avg_glucose_level':'glucose'})

## create a new dataframe with variables used only
strokelevel = df[['stroke','age','bmi','glucose']]

### value counts to determine if the variables are balanced or unbalanced
age_counts = strokelevel['age'].value_counts().reset_index()
bmi_counts = strokelevel['bmi'].value_counts().reset_index()
glucose_counts = strokelevel['glucose'].value_counts().reset_index()
""" all variables are unblanced data"""

####Let's do one-way ANOVA test and assumption test to see the differences between each variables###
#import packages for ANOVA test
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns 
from statsmodels.formula.api import ols
import statsmodels.api as sm

### One-way ANOVA test
""" 1. testing the relationship between age and stroke"""
model = smf.ols("stroke ~ C(age)", data = strokelevel).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table
### p = 1.85822e-55, p< o.05 shows there is a significant difference between age and stroke.

"""2. testing the relationship between bmi and stroke"""
model = smf.ols("stroke ~ C(bmi)", data = strokelevel).fit()
anova_table1 = sm.stats.anova_lm(model, typ=2)
anova_table1
### p = 0.650096, p > 0.05 shows there is NOT a significant difference between bmi and stroke.

"""3. testing teh relationshiop betweeen glucose and stroke"""
model = smf.ols("stroke ~ C(glucose)", data = strokelevel).fit()
anova_table2 = sm.stats.anova_lm(model, typ=2)
anova_table2
### p = 8.10149e-07, p < 0.05 shows there is a significant difference between glucose and stroke.

### Kurtosis testing
"""perform Kurtosis check the outliers present in distribution"""
from scipy.stats import norm, kurtosis

kut1 = stats.kruskal(df['stroke'], df['age'])
### kurtosis = o, mesokurtic distributions, data has a normal distribution

kut2 = stats.kruskal(df['stroke'], df['bmi'])
### kurtosis = nan,  Returns NaN if data has less than three entries or if any entry is NaN.

kut3 = stats.kruskal(df['stroke'], df['glucose'])
### kurtosis = o, mesokurtic distributions, data has a normal distribution

data = norm.rvs(size=1000, random_state=3)
kurtosis(data)



### Post-hoc analysis for significant differences between groups
import statsmodels.stats.multicomp as mc
comp = mc.MultiComparison(df['stroke'], df['age'])
post_hoc_res = comp.tukeyhsd()
post_hoc_res.summary()

"""graphs and visuals to see relationships between variables"""
###Boxplots
stroke_age_boxplot = sns.boxplot(x='age', y= 'stroke', data=strokelevel, palette="Set3")
stroke_bmi_boxplot = sns.boxplot(x='bmi', y= 'stroke', data=strokelevel, palette="Set3")
stroke_glucose_boxplot = sns.boxplot(x='glucose', y= 'stroke', data=strokelevel, palette="Set3")

###Barplots
stroke_bar_age = sns.barplot(x='age', y= 'stroke', data=strokelevel, palette="Set3") 
stroke_bar_bmi = sns.barplot(x='bmi', y= 'stroke', data=strokelevel, palette="Set3") 
stroke_bar_glucose = sns.barplot(x='glucose', y= 'stroke', data=strokelevel, palette="Set3") 

