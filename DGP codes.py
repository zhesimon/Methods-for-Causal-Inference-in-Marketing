'''
For regression of treatment effects
Assume a firm is interested in testing the impact of a new television ad compared to its existing television ad, and the firm will be airing the current ad in TV areas 1, …, 20 and airing the new add in TV areas 21, …, 40. We define each area in terms of artificial area codes i, where i = 1, …, 40, and denote each outcome measure as Y(i, t) for t = 1, …, 10, where t is month.
The variables include average age of people in the area, average income per household, percent of females in the area,  and percentage of days in the period the brand was sold on promotion. 
The outcome variables are percent of households buying the brand during the period, and percentage of buyers (households) buying for the first time during the period.
Treatment effect is 2 for purchase rate, and 3 for % buyers (households) buying for the first time.
'''

import numpy as np
import pandas as pd
import random
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

random.seed(1)

TV_areas, n_periods = 40, 10

age_mean, age_std = 38, 5
income_household_mean, income_household_std = 70000, 10000
female_rate, female_std = 50, 10
pct_days_promo_mean, pct_days_promo_std = 50, 15

data = []
for i in range(TV_areas):
    TV_area = i+1
    age = int(np.random.normal(age_mean, age_std))
    income = int(np.random.normal(income_household_mean, income_household_std))
    female = round(np.random.normal(female_rate, female_std),2)
    pct_days_promo = round(np.random.normal(pct_days_promo_mean, pct_days_promo_std),2)

    for j in range(n_periods):
        TV_area_data = { 
            'period': j+1,
            'TV_areas': TV_area,
            'avg_age': age,
            'avg_income': income,
            '%female': female,
            'pct_days_promo': pct_days_promo,
        }
        data.append(TV_area_data)

df = pd.DataFrame(data)

#outcome var, %households buying the brand
df['purchase_rate'] = (0.02 * df['avg_age'] + 0.5/10000 * df['avg_income'] + 0.1 * df['%female'] 
                       + 0.05 * df['pct_days_promo'] + np.random.uniform(-2, 2, size=len(df)))

#outcome var, %buyers (households) buying for the first time.
df['pct_buyer_1sttime'] = (0.01 * df['avg_age'] + 0.3/10000 * df['avg_income'] + 0.05 * df['%female'] 
                           + 0.1 * df['pct_days_promo'] + np.random.uniform(-2, 2, size=len(df)))

df[['purchase_rate','pct_buyer_1sttime']]=df[['purchase_rate','pct_buyer_1sttime']].round(2).clip(0, 100) #Trim values to between 0 and 100

#treatment group: TV_areas 21-40
df['treatment'] = [0 if i < TV_areas/2*10 else 1 for i in range(df.shape[0])]
df['purchase_rate_t'] = df['purchase_rate'] + 2*df['treatment'] #treatment effect = 2
df['pct_buyer_1sttime_t'] = df['pct_buyer_1sttime'] + 3*df['treatment'] #treatment effect = 3

df.to_csv('simulated_TV_areas.csv', index=False)



'''
For nearest neighbor matching and propensity score matching
We generated the TV areas 21, …, 40 as close neighbors of TV areas 1, …, 20
duplicate the first 200 rows (thus, duplicate TV_areas 1-20 that are controls) and add noise to avg_age, avg_income, %female, pct_days_promo
covariates are the same plus noise, so the duplicated rows are close neighbors of control areas.
'''

new_df1 = df.iloc[:200, :6] #controls
new_df2=new_df1.copy() #new_df2 to be modified to close neighbors of controls

new_df2['TV_areas']=new_df2['TV_areas']+20
new_df2['avg_age'] = df.groupby('TV_areas')['avg_age'].transform(lambda x: x + np.random.normal(0, 2)).round(2)
new_df2['avg_income'] = df.groupby('TV_areas')['avg_income'].transform(lambda x: x + np.random.normal(0, 1000)).round(2)
new_df2['%female'] = df.groupby('TV_areas')['%female'].transform(lambda x: x + np.random.normal(0, 2)).round(2)
new_df2['pct_days_promo'] = df.groupby('TV_areas')['pct_days_promo'].transform(lambda x: x + np.random.normal(0, 2)).round(2)

df = pd.concat([new_df1, new_df2]).reset_index(drop=True)

#outcome var, %households buying the brand
df['purchase_rate'] = (0.02 * df['avg_age'] + 0.5/10000 * df['avg_income'] + 0.1 * df['%female'] 
                       + 0.05 * df['pct_days_promo'] + np.random.uniform(-2, 2, size=len(df)))

#outcome var, %buyers (households) buying for the first time.
df['pct_buyer_1sttime'] = (0.01 * df['avg_age'] + 0.3/10000 * df['avg_income'] + 0.05 * df['%female'] 
                           + 0.1 * df['pct_days_promo'] + np.random.uniform(-2, 2, size=len(df)))

df[['purchase_rate','pct_buyer_1sttime']]=df[['purchase_rate','pct_buyer_1sttime']].round(2).clip(0, 100) #Trim values to between 0 and 100

#treatment for TV_areas 21-40
df['treatment'] = [0 if i < TV_areas/2*10 else 1 for i in range(df.shape[0])]

df['purchase_rate_t'] = df['purchase_rate'] + 2*df['treatment'] #treatment effect = 2
df['pct_buyer_1sttime_t'] = df['pct_buyer_1sttime'] + 3*df['treatment'] #treatment effect = 3


df.to_csv('simulated_TV_areas_nnmatch.csv', index=False)



'''
For instrumental variable
We use IV to estimate the causal effect of advertising on sales, we suspect that there may be unobserved variables that affect both advertising expenditure and sales at the same time.
Instrumental variable: advertising costs. This variable is assumed to affect advertising expenditure but not sales directly.
Assume linear relationship between advertising costs and advertising expenditure, and between advertising expenditure and sales.
We generate data for 100 periods, for each period there are different advertising costs, advertising expenditure, and sales.
'''

np.random.seed(42)
n = 100 #periods
pi0 = 30  # Intercept1
pi1 = -2  # effect of advertising costs on advertising expenditure
beta0 = 35  # Intercept2
beta1 = 2.5  # effect of advertising expenditure on sales

omit_exp=2 #effect of omitted variable on advertising expenditure
omit_sales=3 #effect of omitted variable on sales

# IV: advertising costs
advertising_costs = np.random.randint(1, 11, size=n)

omitted_variable = np.random.randint(1, 3, size=n)

# X: advertising expenditure
advertising_expenditure = pi0 + pi1 * advertising_costs + omit_exp * omitted_variable + np.random.uniform(-2, 2, size=n)

# Y: sales
sales = beta0 + beta1 * advertising_expenditure + omit_sales * omitted_variable + np.random.uniform(-5, 5, size=n)

data = pd.DataFrame({
    'Ad_Costs': advertising_costs,
    'Ad_Expenditure': advertising_expenditure,
    'Omitted_variable': omitted_variable,
    'Sales': sales
})

data.to_csv('IV_Data.csv',index=False)
data['Ad_Costs'].min(), data['Ad_Costs'].max(), data['Ad_Expenditure'].min(), data['Ad_Expenditure'].max(), data['Sales'].min(), data['Sales'].max()



'''
For regression discontinuity design
Assume a firm owns an online platform where people give breakdown (brandimage, price, service) ratings after staying at a hotel.
Based on the aggregate-level feedback of brand image of the hotel, the average perceived price level, and average service performance feedback, an aggregate rating of each product is calculated.

The firm is planning to assign symbols to products based on a cutoff of the ratings, where if the rating is below 3, Symbol B is assigned, if the rating is equal or above 3, Symbol A is assigned.
Treatment effect of Symbol A is 8.
The firm is interested in estimating the impact of symbols on product sales based on 1000 products. We define each product in terms of product identification code i, where i = 1, …, 1000, and denote each Sales measure as Y(i).
'''

random.seed(0)

num_samples = 1000
beta1 = 8  # weight of symbols
beta2 = 5  # weight of ratings

data = []
for _ in range(num_samples):
    brandimage = random.uniform(1, 5)
    price = random.uniform(1, 5)
    service = random.uniform(1, 5)

    rating = brandimage * 0.3 + price * -0.2 + service * 0.4

    data.append({
        'Brandimage': brandimage,
        'Price': price,
        'Service': service,
        'Rating_': rating})

df = pd.DataFrame(data)

# Scale ratings to between 0 and 5
scaler = MinMaxScaler(feature_range=(1, 5))
df['Rating'] = scaler.fit_transform(df[['Rating_']])

df['Symbol'] = df['Rating'].apply(lambda x: 'B' if x < 3 else 'A')

df['Sales'] = beta1 * (df['Symbol'] == 'A') + beta2 * df['Rating'] + df.apply(lambda row: random.uniform(-2, 2), axis=1)

df.to_csv('RDD_ratings.csv', index=False)



'''
For synthetic control method
Assume for a category of product, one state reduced tax recently while other states did not. We are interested in estimating the effect of tax reduction on sales of the category data. We define each state in terms of index i, where i = 0, 1, …, 50, where State0 is the treated unit, and State1-50 are control states. We denote each outcome measure as Y(i, t) for t = 1, …, 200, where t is period. Periods 1-100 are pre-treatment periods, treatment happens in period 100, and periods 101-200 are post-treatment periods. 
Treatment effect is 10.
We generated a synthetic control using  State1 (weight 0.2) and State2 (weight 0.8) and random noise. In our simulation, we tried performing synthetic control methods based on all 50 states, States 1-25, States 1-10, States 1-5, and All 100 pre-treatment periods, pre-treatment periods 51-100, pre-treatment periods 81-100, pre-treatment periods 91-100.
After obtaining the synthetic control, we use diff-in-diff to estimate the treatment effect.
'''

random.seed(40)

betas = [0.2, 0.8] + [0] * 48 #weights used for synthetic control
#creates variables beta1, ..., beta50 and assigns them values from betas.
for i in range(1, 51):
    globals()[f'beta{i}'] = betas[i - 1]

print(beta1)
print(beta2)
print(betas)


#generates a dictionary containing 50 key-value pairs, keys "mu1", "mu2", ..., "mu50", values are random integers in range
mu_gen = {}
for i in range(1, 51):
    mu_gen[f"mu{i}"] = random.randint(5, 15)
print(mu_gen)

mus = list(mu_gen.values()) #a list containing values of all mu's
print(mus)


#create 50 variables mu1,..., mu50, corresponding to the values stored in dictionary mu_gen
for i in range(1, 51):
    globals()[f"mu{i}"] = mu_gen[f"mu{i}"]
#print(mu1, mu2, mu3)
    
control_units = ['Y' + str(i) for i in range(1, 51)]

#for i, Y in enumerate (control_units): 
#    print(control_units[i],mus[i],betas[i])


n=200
treated_period=int(n/2)
random_state=100

data= pd.DataFrame(index=range(n))

#control unit Y[i] ~ N (mu[i], 5)
for i, Y in enumerate (control_units):
    random_state += 1
    data[control_units[i]]=norm.rvs(loc=mus[i], scale=5, size=data.shape[0], random_state=random_state)


#error term for the synthetic control ~ N (0, 1)
random_state=random_state+1
data['error']=norm.rvs(loc=0, scale=1, size=data.shape[0], random_state=random_state)

#data_w holds the weighted control units Y[i]w, weighted by their corresponding beta values
data_w=pd.DataFrame(index=range(n))
for i, Y in enumerate (control_units):
    data_w[control_units[i]+'w']=betas[i]*data[control_units[i]]

data_w['error']=data['error']

print(data_w['Y1w'][0]+data_w['Y2w'][0]+data_w['error'][0]) #checking the number
data_w.to_csv('Data_w.csv',index=False)

#treated outcome in the absence of treatment: weighted sum plus noise
data['Y0_treated'] = data_w[list(data_w.columns)].sum(axis=1) 

#treatment effect for the treated units (post-treatment periods 101-200), 10 + eps ~Uniform (-0.1,+0.1)
data['eps_te']  = np.concatenate((np.zeros(treated_period,dtype=int), np.random.uniform(-0.1,0.1,treated_period)+10))

#Treated outcome in the presence of treatment 
data['Y0_te'] =data['Y0_treated']+data['eps_te'] 

data['period'] = np.arange(1,n+1) 

#check that the weighted Ys are Y*beta
for i in range(data.shape[0]):
    assert(data['Y1'][i]*betas[0]==data_w['Y1w'][i])
    
data.to_csv('Synthetic_Control_Data.csv',index=False)
