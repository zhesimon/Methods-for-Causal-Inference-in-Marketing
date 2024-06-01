*This file should be run after the Python generation code.
*Please specify the directory path to the files on your system in place of <YourDirectoryPath>.

******************************
***Regression of Treatment Effects
******************************

import delimited "<YourDirectoryPath>/simulated_TV_areas.csv", clear 
rename female pct_female
*Dependent var: purchase rate
reg purchase_rate_t avg_age avg_income pct_female pct_days_promo treatment
*Dependent var: percent of first time buyer
reg pct_buyer_1sttime_t avg_age avg_income pct_female pct_days_promo treatment



******************************
***Nearest-Neighbor Matching
******************************

import delimited "<YourDirectoryPath>/simulated_TV_areas_nnmatch.csv", clear 
rename female pct_female

*Dependent var: purchase rate
teffects nnmatch (purchase_rate_t avg_age avg_income pct_female pct_days_promo)(treatment),nneighbor(1)
*Dependent var: percent of first time buyer
teffects nnmatch (pct_buyer_1sttime_t avg_age avg_income pct_female pct_days_promo)(treatment),nneighbor(1)



******************************
***Propensity Score Matching
******************************

*Dependent var: purchase rate
teffects psmatch (purchase_rate_t) (treatment avg_age avg_income pct_female pct_days_promo)
*Dependent var: percent of first time buyer
teffects psmatch (pct_buyer_1sttime_t) (treatment avg_age avg_income pct_female pct_days_promo)



******************************
***Instrumental Variable
******************************

import delimited "<YourDirectoryPath>/IV_Data.csv", clear

*manually run 2sls
reg ad_expenditure ad_costs
gen constructed_ad_expenditure = _b[_cons] + _b[ad_costs] * ad_costs
reg sales constructed_ad_expenditure

*run IV
ivregress 2sls sales (ad_expenditure = ad_costs)

*Tests of endogeneity: check whether ad_expenditure is endogenous
estat endog

*check whether the instrument is weak
estat firststage



******************************
***Regression Discontinuity Method
******************************

net install rdrobust, from(https://raw.githubusercontent.com/rdpackages/rdrobust/master/stata) replace

import delimited "<YourDirectoryPath>/RD_ratings.csv", clear 
*RD plot, Cut-off c = 3
rdplot sales rating, c(3) graph_options(title(RD Plot) xtitle("Rating") ytitle("Sales"))
*Sharp RD estimates
rdrobust sales rating, c(3) all



******************************
***Synthetic Control Method
******************************

*Convert time series to panel data
import delimited "<YourDirectoryPath>/Synthetic_Control_Data.csv", clear 
reshape long y, i(period) j(State)
rename period Period
rename y Sales
save "<YourDirectoryPath>/Synth_Panel.dta", replace


*Performing synthetic control method using all 50 control states and all pre-treatment periods
use "<YourDirectoryPath>/Synth_Panel.dta", clear
tsset State Period
synth Sales Sales(1(1)100), trunit(0) trperiod(101) fig keep(100period_50state)
*graph export "/Graph_100period_50state.png", replace


*Performing synthetic control method using partial (25) control states and all pretreatment periods
use "<YourDirectoryPath>/Synth_Panel.dta", clear
keep if State<26
tsset State Period
synth Sales Sales(1(1)100), trunit(0) trperiod(101) fig keep(100period_25state)


*Performing synthetic control method using all 50 control states and 50 pre-treatment periods
use "<YourDirectoryPath>/Synth_Panel.dta", clear
keep if Period>50
tsset State Period
synth Sales Sales(51(1)100), trunit(0) trperiod(101) fig keep(50period_50state)


* Calculating treatment effects using DID (reshape data from wide format to long format)
use "<YourDirectoryPath>/100period_50state.dta", clear
drop _Co_Number _W_Weight
reshape long _Y_, i(_time) j(State, string)
gen treatment = 0
replace treatment = 1 if _time >=101 & State == "treated"
encode State, generate(nState)
xtset nState
xtdidregress (_Y_)(treatment), group(nState) time(_time)



******************************
***Differences-in-Differences
******************************

use "<YourDirectoryPath>/Synth_Panel.dta", clear

*DID result using State2 as control (which has weight of beta 0.8 when generating the treatment unit in our simulation). 
keep if State ==2 | State ==0
gen treatment = 0
replace treatment = 1 if Period >=101 & State == 0
xtset State
xtdidregress (Sales)(treatment), group(State) time(Period)
