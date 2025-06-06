clear
cd F:\FF三因子stata
import delimited data.csv,clear
rename mktcap me
save data.dta,replace

*去年12月的账面市值比
use data.dta,clear
keep if mod(month,100)==12
gen bmyear=int(month/100)
keep stkcd bmyear bm
save bm.dta,replace

*6月末的市值
use data.dta,clear
keep if mod(month,100)==6
gen meyear=int(month/100)
keep stkcd meyear me
save me.dta,replace

*上月末市值，用于计算权重
use data.dta,clear
rename me weight
keep stkcd month weight
gen month2=month+1
replace month2=month+100-11 if mod(month,100)==12
drop month
rename month2 month
save weight.dta,replace

use data.dta,clear
*每年7月初使用今年6月末的市值和去年12月的账面市值比进行分组
gen bmyear=int(month/100)-1
replace bmyear=int(month/100)-2 if mod(month,100)<=6
gen meyear=int(month/100)
replace meyear=int(month/100)-1 if mod(month,100)<=6
sort stkcd month
keep stkcd month bmyear meyear exret

merge 1:1 stkcd month using weight.dta,keep(match) nogen
merge m:1 stkcd bmyear using bm.dta,keep(match) nogen
merge m:1 stkcd meyear using me.dta,keep(match) nogen

drop if exret==.
drop if me==.
drop if bm==.
drop if weight==.

*最终需要的月份，共计240个月
keep if month>200100 & month<202100 
sort stkcd month
egen id=group(month)
save month.dta,replace

forvalues i=1(1)240{
use month.dta,clear
keep if id==`i'

*等权重市场因子
egen mkt_rf_equal=mean(exret)

*市值加权市场因子
gen vw=exret*weight
egen mkt_rf1=sum(vw)
egen mkt_rf2=sum(weight)
gen mkt_rf=mkt_rf1/mkt_rf2

*根据市值分组
cumul me, g(group_me) eq 
recode group_me (min/0.5=1)(0.5/max=2) 

*根据账面市值比分组
cumul bm, g(group_bm) eq 
recode group_bm (min/0.3=1)(0.3/0.7=2)(0.7/max=3) 

keep stkcd month exret vw weight me mkt_rf mkt_rf_equal group_me group_bm

*等权重组合收益率
bys group_me group_bm: egen rew=mean(exret)
*市值加权组合收益率
bys group_me group_bm: egen r1=sum(vw)
bys group_me group_bm: egen r2=sum(weight)
gen r=r1/r2
duplicates drop group_me group_bm,force 
sort group_me group_bm

gen smb_equal=(rew[1]+rew[2]+rew[3]-rew[4]-rew[5]-rew[6])/3
gen hml_equal=(rew[3]+rew[6]-rew[1]-rew[4])/2

gen smb=(r[1]+r[2]+r[3]-r[4]-r[5]-r[6])/3
gen hml=(r[3]+r[6]-r[1]-r[4])/2

duplicates drop month,force
keep month mkt_rf_equal smb_equal hml_equal mkt_rf smb hml 
order month mkt_rf_equal smb_equal hml_equal mkt_rf smb hml 
save m`i'.dta,replace
}

*拼接数据
use m1.dta,clear
forvalues i=2(1)240{
append using m`i'.dta
}
save ff3factor.dta,replace

forvalues i=1(1)240{
cap erase m`i'.dta
}

export delimited FF3因子stata.csv,replace

dir *.dta
local filelist: dir . files "*.dta"
foreach files of local filelist {
cap erase `files'
}
