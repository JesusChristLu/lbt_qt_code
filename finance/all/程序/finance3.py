import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xlrd as xd
import xlwt as xt
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from scipy.interpolate import make_interp_spline

def read_xlsx(fname, sheet):
    data = xd.open_workbook(fname)
    sheet = data.sheet_by_name(sheet)
    d = []
    for r in range(sheet.nrows):
        data1 = []
        for c in range(sheet.ncols):
            data1.append(sheet.cell_value(r, c))
        d.append(list(data1))
    return d

def linear_regu(x, y):
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    results = model.fit()
    const, slope = results.params[0], results.params[1]
    constStd, slopestd = results.bse[0], results.bse[1]
    tvalues = results.tvalues
    pvalues = results.pvalues
    fvalue = results.fvalue
    fpvalue = results.f_pvalue
    R = results.rsquared_adj
    return const, slope, constStd, slopestd, tvalues, pvalues, fvalue, fpvalue, R

def multi_linear_regu(x, y):
    data_x = pd.DataFrame({'rm' : x[0], 'smb' : x[1], 'hml' : x[2]})
    data_y = pd.Series(y)
    data_y.name = 'Rit'
    data = pd.concat([data_x, data_y], axis=1)
    mod = smf.ols(formula='Rit~rm+smb+hml', data=data)
    res = mod.fit()
    const, c1, c2, c3 = res.params[0], res.params[1], res.params[2], res.params[3]
    return const, c1, c2, c3


data = read_xlsx('能源.xls', '中国')
# 写数据
workbook = xt.Workbook(encoding='ascii')
worksheet = workbook.add_sheet('Sheet1')

worksheet.write(0, 0, 'id')
worksheet.write(0, 1, 'num_id')
worksheet.write(0, 2, 'date')
worksheet.write(0, 3, 'ret')
worksheet.write(0, 4, 'price')
for raw in range(1, len(data)):
    if '' == data[raw][3]:
        continue
    for i in range(0, len(data[0]) - 3):
        worksheet.write(1 + (raw - 1) * (len(data[0]) - 3) + i, 0, data[raw][1])
        worksheet.write(1 + (raw - 1) * (len(data[0]) - 3) + i, 1, data[raw][2])
        worksheet.write(1 + (raw - 1) * (len(data[0]) - 3) + i, 2, data[0][i + 3])
        if i == 0:
            worksheet.write(1 + (raw - 1) * (len(data[0]) - 3) + i, 3, 0)    
            worksheet.write(1 + (raw - 1) * (len(data[0]) - 3) + i, 4, float(data[raw][i + 3]))    
        else:
            worksheet.write(1 + (raw - 1) * (len(data[0]) - 3) + i, 3, (float(data[raw][i + 3]) - float(data[raw][i + 2])) / float(data[raw][i + 3]))
            worksheet.write(1 + (raw - 1) * (len(data[0]) - 3) + i, 4, (float(data[raw][i + 3])))
workbook.save('combineChina.xls')
