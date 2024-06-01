'''
Zezhen (Dawn) He and Vithala R. Rao (2024), “Methods for Causal Inference in Marketing”, 
Foundations and Trends® in Marketing: Vol. 18, No. 3, pp 176–309.

Appendix A: Python Code for Generating Simulated Data
'''



'''
For Regression of Treatment Effects, Section 3.4.1

Example Context: Assume a firm is interested in testing the impact of
a new television ad compared to its existing television ad, and the firm
will be airing the current ad in TV areas 1, . . . , 20 and airing the new
add in TV areas 21, . . . , 40. We define each area in terms of artificial
area codes i, where i = 1, . . . , 40, and denote each outcome measure as
Y (i, t) for t = 1, . . . , 10, where t is month.
The variables describing the TV areas are the average age of people
in the area, average income per household, the percentage of females in
the area, and the percentage of days in the period the brand was sold
on promotion.
The two outcome variables are the percentage of households buying
the brand during the period, and the percentage of buyers (households)
buying for the first time during the period.
The data is generated with the assumed treatment effects of 2 for
the percentage of households buying the brand during the period and
3 for the percentage of buyers (households) buying for the first time
during the period.

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

#Trim values to between 0 and 100
df[['purchase_rate','pct_buyer_1sttime']]=df[['purchase_rate','pct_buyer_1sttime']].round(2).clip(0, 100)

#treatment group: TV_areas 21-40
df['treatment'] = [0 if i < TV_areas/2*10 else 1 for i in range(df.shape[0])]
df['purchase_rate_t'] = df['purchase_rate'] + 2*df['treatment'] #treatment effect = 2
df['pct_buyer_1sttime_t'] = df['pct_buyer_1sttime'] + 3*df['treatment'] #treatment effect = 3

df.to_csv('simulated_TV_areas.csv', index=False)



'''
For Nearest Neighbor Matching, Section 3.4.7 and Propensity Score Matching, Section 3.4.8

Example Context: Here we used a similar dataset except that we generated
the TV areas 21, . . . , 40 as close neighbors of TV areas 1, . . . , 20.
To do that, we duplicated the first 200 rows (thus, duplicated
TV_areas 1–20 that are control units) and added noise to the average
age of people living in the area, the average income per household, the
percentage of female in the area, and the percentage of days in the
period the brand was sold on promotion.
The covariates are the same with noise added, so the areas from the
duplicated rows are close neighbors of control areas.
'''

new_df1 = df.iloc[:200, :6] #controls
new_df2=new_df1.copy() #new_df2 to be modified to close neighbors of controls

new_df2['TV_areas']=new_df2['TV_areas']+20

# error ~ N(0, 2) for age
new_df2['avg_age'] = df.groupby('TV_areas')['avg_age'].transform(lambda x: x + np.random.normal(0, 2)).round(2)
# error ~ N(0, 1000) for income
new_df2['avg_income'] = df.groupby('TV_areas')['avg_income'].transform(lambda x: x + np.random.normal(0, 1000)).round(2)
# error ~ N(0, 2) for percent female
new_df2['%female'] = df.groupby('TV_areas')['%female'].transform(lambda x: x + np.random.normal(0, 2)).round(2)
# error ~ N(0, 2) for percent of days the brand was sold on promotion
new_df2['pct_days_promo'] = df.groupby('TV_areas')['pct_days_promo'].transform(lambda x: x + np.random.normal(0, 2)).round(2)

df = pd.concat([new_df1, new_df2]).reset_index(drop=True)

#outcome var, %households buying the brand
df['purchase_rate'] = (0.02 * df['avg_age'] + 0.5/10000 * df['avg_income'] + 0.1 * df['%female'] 
                       + 0.05 * df['pct_days_promo'] + np.random.uniform(-2, 2, size=len(df)))

#outcome var, %buyers (households) buying for the first time.
df['pct_buyer_1sttime'] = (0.01 * df['avg_age'] + 0.3/10000 * df['avg_income'] + 0.05 * df['%female'] 
                           + 0.1 * df['pct_days_promo'] + np.random.uniform(-2, 2, size=len(df)))

#Trim values to between 0 and 100
df[['purchase_rate','pct_buyer_1sttime']]=df[['purchase_rate','pct_buyer_1sttime']].round(2).clip(0, 100)

#treatment for TV_areas 21-40
df['treatment'] = [0 if i < TV_areas/2*10 else 1 for i in range(df.shape[0])]

df['purchase_rate_t'] = df['purchase_rate'] + 2*df['treatment'] #treatment effect = 2
df['pct_buyer_1sttime_t'] = df['pct_buyer_1sttime'] + 3*df['treatment'] #treatment effect = 3


df.to_csv('simulated_TV_areas_nnmatch.csv', index=False)



'''
For Instrumental Variable, Section 3.4.3

Example Context: We used the IV method to estimate the causal effect
of advertising on sales. For this purpose, we generated one unobserved
variable that affects both advertising expenditure and sales at the same
time, so there is endogeneity if we directly regress sales on advertising
expenditure. The instrumental variable is set as advertising costs. This
variable is generated to affect advertising expenditure but not sales
directly. We also assumed a linear relationship between advertising costs
and advertising expenditure, and between advertising expenditure and
sales.
We generated data for 100 periods, for each period there are different
advertising costs, advertising expenditure, and sales. The treatment
effect of advertising expenditure on sales is set to be 2.5.
'''

np.random.seed(42)
n = 100 #periods
pi0 = 30  # Intercept1
pi1 = -2  # effect of advertising costs on advertising expenditure
beta0 = 35  # Intercept2
beta1 = 2.5  # effect of advertising expenditure on sales

omit_exp=2 #effect of omitted variable on advertising expenditure
omit_sales=3 #effect of omitted variable on sales
omitted_variable = np.random.randint(1, 3, size=n)

# IV (Z): advertising costs
advertising_costs = np.random.randint(1, 11, size=n)


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



'''
For Regression Discontinuity Design, Section 3.4.4

Example Context: Assume a firm owns an online platform where hotel
guests provide ratings on three variables (brand image, price, and
service) after staying at a hotel.
Based on the overall feedback on the three variables (i.e., the average
score of brand image of the hotel, the average perceived price level, and
the average score of service performance), the firm develops a rating of
each hotel listed on the platform.
Further, the firm assigns symbols to the hotels based on a threshold.
If the rating is below 3, the platform assigns a Symbol B, and if the
rating is equal or above 3, Symbol A is assigned. Relative to Symbol B,
the treatment effect of being assigned Symbol A is 8.
The firm is interested in estimating the impact of being assigned to
Symbol A compared to Symbol B on hotel sales based on 1000 hotels.
We define each hotel in terms of hotel identification code i, where
i = 1, . . . , 1000, and denote each Sales measure as Y (i).
'''

random.seed(0)

num_samples = 1000
beta1 = 8  # effect of symbol A on sales
beta2 = 5  # effect of ratings on sales

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

# if rating < 3, assign symbol B, else assign Symbol A
df['Symbol'] = df['Rating'].apply(lambda x: 'B' if x < 3 else 'A')

df['Sales'] = beta1 * (df['Symbol'] == 'A') + beta2 * df['Rating'] + df.apply(lambda row: random.uniform(-2, 2), axis=1)

df.to_csv('RD_ratings.csv', index=False)



'''
For Synthetic Control Method, Section 3.4.5 (and Differences-In-Differences, Section 3.4.2)

Example Context: Assume that a country consists of 51 geographically
identified States and that these States have a sales tax for products sold
within the State. Assume further that for a specific product category,
one State reduced tax in period 100 while other States did not. We are
interested in estimating the effect of the tax reduction on the State’s
product category sales using data available for 200 units of time (e.g.,
weeks). We define each State in terms of index i, where i = 0, 1, . . . , 50,
where State 0 is the treated unit, and States 1–50 are control States.
We denote each outcome measure as Y (i, t) for t = 1, . . . , 200, where
t is period. Periods 1–100 are pre-treatment periods, the treatment
happened in period 100, and periods 101–200 are post-treatment periods.
We assumed that the treatment effect is 10.
Initially, we generated the treatment unit using only two States i.e.,
State 1 (weight 0.2) and State 2 (weight 0.8) and random noise. Later,
we performed the Synthetic Control Method based on all 50 States,
States 1–25, States 1–10, and States 1–5. The pre-treatment periods
in our simulation were set as periods 1–100, periods 51–100, periods
81–100, or periods 91–100.
After obtaining the synthetic control, we used DID to estimate the
treatment effect.
'''

random.seed(40)

# weights used for synthetic control (correspond to the 50 control states)
betas = [0.2, 0.8] + [0] * 48

#creates variables beta1, ..., beta50 and assigns them values from betas.
for i in range(1, 51):
    globals()[f'beta{i}'] = betas[i - 1]

print(beta1, beta2)
print(betas)


#generates a dictionary containing 50 key-value pairs, keys are "mu1", "mu2", ..., "mu50", values are random integers in range (5, 15)
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

#treatment effect for the treated units (post-treatment periods 101-200), which equals 10 + eps ~ Uniform (-0.1,+0.1)
data['eps_te']  = np.concatenate((np.zeros(treated_period,dtype=int), np.random.uniform(-0.1,0.1,treated_period)+10))

#Treated outcome in the presence of treatment 
data['Y0_te'] =data['Y0_treated']+data['eps_te'] 

data['period'] = np.arange(1,n+1) 

#check that the weighted Ys are Y*beta
for i in range(data.shape[0]):
    assert(data['Y1'][i]*betas[0]==data_w['Y1w'][i])
    
data.to_csv('Synthetic_Control_Data.csv',index=False)
