import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from konlpy.tag import Okt
from tqdm import tqdm
from konlpy.tag import Kkma

cpuData_csv = pd.read_csv('./cpuDataSet.csv',  encoding = 'CP949')
#cpuData_excel = pd.read_excel('./cpuData.xlsx', encoding = 'CP949')


cpuData_csv['명사'] = cpuData_csv['발명의 명칭'] + ' ' + cpuData_csv['요약'] + ' ' + cpuData_csv['대표청구항']
cpuData_csv = cpuData_csv.drop(['발명의 명칭', '요약', '대표청구항'], 1)
cpuData_csv = cpuData_csv.astype('object')

cpuDict = Okt()
kkmaCPU = Kkma()

def morph(input_data):
    preprcessed = kkmaCPU.pos(input_data)
    print(preprcessed)

morph(cpuData_csv['명사'])
# with tqdm(total = len(list(cpuData_csv.iterrows()))) as tProgressBar:
#     for i, row in cpuData_csv.iterrows():
#         if isinstance(row['명사'], str):
#             cpuData_csv.loc[i, '명사'] = cpuDict.nouns(row['명사'])
#         tProgressBar.update(1)

print('writing okt noun')
writer = pd.ExcelWriter('cpu_noun_kkma.xlsx') # pylint: disable=abstract-class-instantiated
#writer.book.use_zip64()
cpuData_csv.to_excel(writer, sheet_name = 'sheet1', index = False, header = True)
writer.save()
print('end writing')

# cpuData_csv = pd.read_excel('./cpuDataSet.csv', sheet_name = 'sheet1', header = 0)





