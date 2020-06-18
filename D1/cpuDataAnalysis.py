import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from konlpy import Okt

cpuData_csv = pd.read_csv('./cpuDataSet.csv',  encoding = 'CP949')
#cpuData_excel = pd.read_excel('./cpuData.xlsx', encoding = 'CP949')


cpuData_csv['명사'] = cpuData_csv['발명의 명칭'] + ' ' + cpuData_csv['요약'] + ' ' + cpuData_csv['대표청구항']

cpuDict = Okt()



