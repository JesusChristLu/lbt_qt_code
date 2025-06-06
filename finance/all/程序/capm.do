clear

import excel "F:\BaiduNetdiskWorkspace\vs experiment\finance\combineChina.xls", sheet("Sheet1") firstrow
save "F:\BaiduNetdiskWorkspace\vs experiment\finance\combineChina.dta"
use "F:\BaiduNetdiskWorkspace\vs experiment\finance\combineChina.dta"

import excel "F:\BaiduNetdiskWorkspace\vs experiment\finance\chinaindex.xls", sheet("Sheet1") firstrow
save "F:\BaiduNetdiskWorkspace\vs experiment\finance\chinaindex.dta"
use "F:\BaiduNetdiskWorkspace\vs experiment\finance\chinaindex.dta"

merge m:1 date using "F:\BaiduNetdiskWorkspace\vs experiment\finance\chinaindex.dta"
drop if _merge == 2
drop _merge

gen date2 = date(date,"DMY")
clonevar date3 = date2 
format date %td  
drop date2
rename date3 date   
drop date

destring mktret, replace

rename id enterprise

egen id=group(enterprise)
xtset id date

xtdes
xtsum

*xtunitroot fisher price, dfuller drift lags(2) demean

*xtunitroot fisher open, dfuller drift lags(2)

*xtunitroot fisher close, dfuller drift lags(0)

gen t1=-5
gen t2=5

sort id date 
by id: gen date_p1 = _n	
by id: gen date_p2 = date_p1 if date == event_date1
by id: replace date_p2 = date_p1 if date-event_date1<=t2 & date-event_date1>=t1

egen date_p3 = mean(date_p2), by(id)
gen date_p4 = round(date_p3)
gen date_new = date_p1 - date_p4
drop date_p1 - date_p4					//至此对数据的初步划分全部结束
keep if date_new <= t2	
keep if abs(date_new) < 145


 *划分窗口期
gen event_window = 0 
replace event_window = 1 if date_new >= t1 & date_new <= t2
											//选取包括事件前后各一天共计三天为窗口期 
  *划分估计期
gen event_estimate = 1
replace event_estimate = 0 if date_new >= t1 & date_new <= t2
//把事件日以前除前一日以外的事件全部设置为估计期
*分组回归计算预测值
egen N = max(id)  //这里的N代表上一步创建的id的最大值
gen predict_return=.
forvalues i = 1/80{
display `i'
*得到估计窗口中的正常回报率
reg ret mktret if event_estimate==1 & id==`i'
*估计在事件窗口内计算正常回报率
predict p if id==`i'
replace predict_return = p if event_window==1 & id==`i'
drop p
}


*计算超额收益AR
gen AR=ret-predict_return if event_window==1 

by id:egen CAR=sum(AR) //计算每个公司在事件窗口内的累计非正常报酬率（固定值）
gen car_date=.
forvalues i=1/80{
replace car_date=sum(AR) if (id==`i' & event_window==1)
} //每个公司在事件窗口内逐日累加得到的累积回报率

sort id date //按id、交易日排序
by id:egen ar_sd=sd(AR) //对每个公司求超额收益的标准差
gen test=(1/sqrt(80))*(CAR/ar_sd) //生成test统计量

reg CAR if date_new==0, robust


list id CAR test if date_new==0 //列出发生事件的公司代码，累计超额收益，test统计量
*outsheet id date CAR test using car_test.csv if date_new==0,comma names

preserve
keep if event_window==1
bysort date_new:egen car_t=mean(car_date)
keep date_new car_t //计算每天的平均累积超常回报率
duplicates drop //仅保留事件期的观察值
twoway connect car_t date_new //绘制时序图
restore

bysort date_new:egen car_t=mean(car_date)

list date_new car_t
duplicates drop
