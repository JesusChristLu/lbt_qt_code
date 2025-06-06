Copyright © 2021 Shutter_Z. All rights reserved
email: Shutter_Z@outlook.com / 1339895993@qq.com
*导入数据
	import excel "D:\研究生\经典文献\data.xlsx", firstrow
*日期标红（文字型变量转日期变量）
	gen date_date = date(date_str, "MDY")
	format date_date %td					//实际到这里就可以直接开始转换
	
*对数据日期的基本处理
	sort code date 
	by code: gen date_p1 = _n	
	by code: gen date_p2 = date_p1 if date == event_date
	by code: replace date_p2 = date_p1 if date-event_date<=1 ///
					 & date-event_date>=-1
	egen date_p3 = mean(date_p2), by(code)
	gen date_p4 = round(date_p3)
	gen date_new = date_p1 - date_p4
	drop date_p1 - date_p4					//至此对数据的初步划分全部结束
	keep if date_new <= 1					
	keep if abs(date_new) < 210
	*keep if date_new > -210				//至此对数据的完全划分全部结束
  
  *划分窗口期
	gen event_window = 0 
	replace event_window = 1 if date_new >= (-1) & date_new <= 1
											//选取包括事件前后各一天共计三天为窗口期 
  *划分估计期
	gen event_estimate = 1
	replace event_estimate = 0 if date_new >= (-1) & date_new <= 1
											//把事件日以前除前一日以外的事件全部设置为估计期

*分组回归计算预测值
	egen id = group(code)
	egen max_id = max(id)
	*sum id	
	gen predict_return = .
	forvalues i = 1/11 {
	reg share_earn market_earn if id == `i' ///
		& event_estimate == 1 
	predict p if id == `i'
	replace predict_return = p if id == `i'	///
		& event_window == 1
	drop p
	}

*计算超额收益AR
	gen AR= share_earn - predict_return

*计算个股累计超额收益CAR
	bysort code(date): gen CAR_initial = sum(AR)	//计算CAR
	by code: gen CAR = CAR_initial[_N]				//保留最后的累计CAR
	preserve
		keep code CAR 
		duplicates drop code CAR, force
		list code CAR
	restore											//不破坏原数据的情况下显示个股CAR
*t检验
	by code: ttest CAR_initial == 0 if event_window == 1