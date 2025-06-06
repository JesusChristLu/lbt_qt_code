// use https://gitee.com/arlionn/data/raw/master/data01/eventdates.dta
// use https://gitee.com/arlionn/data/raw/master/data01/stockdata.dta
// https://www.lianxh.cn/news/3820f71099fd9.html

**************1.数据准备工作**************
use "E:\B站视频数据\10.事件研究法\eventdates.dta", clear
format event_date %td

*排序
sort company_id

*分组计算
by company_id: gen eventcount=_N
by company_id: keep if _n==1
*数据删除/保留
keep company_id eventcount
*保存处理后数据 
save eventcount.dta, replace

use "E:\B站视频数据\10.事件研究法\stockdata.dta", clear
*排序
sort company_id
*数据合并(通过id合并上面保存的eventcount数据)
merge company_id using eventcount
*tab查看频率
tab _merge
*数据保存和删除
keep if _merge==3
drop _merge
expand eventcount	
drop eventcount
*排序和分组计算
sort company_id date
by company_id date: gen set=_n
sort company_id set
*保存数据
save stockdata2.dta, replace


use "E:\B站视频数据\10.事件研究法\eventdates.dta", clear
*排序和分组计算
sort company_id
by company_id: gen set=_n
sort company_id set
save eventdates2.dta, replace

use "E:\B站视频数据\10.事件研究法\stockdata2.dta", clear
merge company_id set using eventdates2
tab _merge	
list company_id if _merge==2 
keep if _merge==3
drop _merge
egen group_id = group(company_id set)	



**************2.计算距离事件发生的天数***************
*首先对面板数据按照id和日期进行排序
sort company_id date

/*产生一个新变量datenum，
在每个id内部对日期进行编号(比如1Jan2020是1，2Jan2020是2，3Jan2020是3)
这一步有利于接下来进行事件窗口的定义*/
by company_id: gen datenum = _n 

/*产生一个新变量target，代表了事件发生的日期，
因为在上一步我们已经进行了编号，因此这一步中，
如果事件发生的日期是2Jan2020，那么target产生的值就会是2*/
by company_id: gen target=datenum if date==event_date 

*产生一个新变量td，填充了每个id内事件发生的日期编号
egen td=min(target), by(company_id)

/*最终dif变量代表了距离事件发生的天数，0代表事件发生当天，
负值代表事件发生前，正值代表事件发生后*/
gen dif=datenum-td 


**************3.定义事件和估计窗口**************

*我们选择事件发生的前两天和后两天作为事件发生的窗口
by company_id: gen event_window=1 if dif>=-2 & dif<=2 
*计算每个id内部属于事件窗口内的个数
egen count_event_obs=count(event_window), by(company_id) 

*选择估计窗口样本，原则是选择事件发生前且不与事件窗口重合的样本
by company_id: gen estimation_window=1 if dif<-30 & dif>=-60 
*同理，计算每个id内部属于估计窗口的个数
egen count_est_obs=count(estimation_window), by(company_id) 

*以下两步将事件窗口和估计窗口的缺失值替换成0值
replace event_window=0 if event_window==.
replace estimation_window=0 if estimation_window==. 

/*这里我们需要在做回归之前，
将不满足我们所定义的事件窗口长度(5天)的样本去掉*/
drop if count_event_obs < 5
*同理，我们也需要将不符合估计窗口长度(30天)的样本去掉
drop if count_est_obs < 30

**************4.估计正常表现**************
*创建一个新的变量储存我们估计的正常表现值
gen predicted_return=. 
*为每个公司创建一个新的变量
egen id=group(company_id) 

/*在这一步，我们利用估计窗口，循环估计每个id正常表现值，
该正常变现值是通过回归在估计窗口内的Y与X得到，
并将其进一步在事件窗口中进行预测*/
local N = 354  //这里的N代表上一步创建的id的最大值
forvalues i=1(1)`N'{
	l id company_id if id==`i' & dif==0
	*得到估计窗口中的正常回报率
	reg ret market_ret if id==`i' & estimation_window==1 
	*估计在事件窗口内计算正常回报率
	predict p if id==`i'
	replace predicted_return = p if id==`i' & event_window==1 
	drop p
}


**************5.估计异常表现**************
sort id date
*异常收益率是实际收益率与不发生事件的估计收益率的差值
gen abnormal_return=ret-predicted_return if event_window==1 
*另一个指标是累计异常收益率
by id: egen cumulative_abnormal_return = total(abnormal_return)



**************6.显著性检验**************
sort id date
*计算公式中所需要的AR_SD
by id: egen ar_sd = sd(abnormal_return)
local event_window_days = 5
gen test =(1/sqrt(`event_window_days'))*(cumulative_abnormal_return/ar_sd) 


**************7.全部事件的稳健性检验**************
reg cumulative_abnormal_return if dif==0, robust 















 