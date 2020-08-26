'''
해당 소스코드는 다음의 환경에서 이루어짐
Local 컴퓨터: 
    OS : windows 64bit
    Python : 3.7.7.final.0
    pandas : 1.0.4
    numpy : 1.18.1

Google Colaboratory:
    Python3 버전
    하드웨어 가속기 : None
'''
''' 밑의 주석된 코드는 Google Colaboratory에서 사용함 
import matplotlib as mpl
import matplotlib.pyplot as plt
!apt -qq -y install fonts-nanum
import matplotlib.font_manager as fm
fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)
plt.rc('font', family='NanumBarunGothic') 
%config InlineBackend.figure_format = 'retina'
mpl.font_manager._rebuild()
!pip install konlpy
!pip install nltk
'''

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import openpyxl

from konlpy.tag import Kkma
import nltk
import re

# 데이터 명사 추출
cpuData = pd.read_excel('./drive/My Drive/Colab_Notebooks/filtering_Data/cpuValidData.xlsx') # 유효 데이터
invalid_KR = pd.read_excel('./drive/My Drive/Colab_Notebooks/excel_Data/Real_invalid_Data_KR.xlsx') # 비교 데이터(한국어)
invalid_ENG = pd.read_excel('./drive/My Drive/Colab_Notebooks/excel_Data/Real_invalid_Data_ENG.xlsx') # 비교 데이터(영어)

# 유효 데이터 전처리
for i in range(len(cpuData['대표청구항'])):
    cpuData['대표청구항'][i] = cpuData['대표청구항'][i][8:]

for i in range(len(cpuData['발명의 명칭'])):
    cpuData['발명의 명칭'][i] = re.sub(r'\([^)]*\)', '', cpuData['발명의 명칭'][i])

cpuData_KR = cpuData[cpuData['국가코드'].isin(['KR']) | cpuData['국가코드'].isin(['JP'])]
cpuData_ENG = cpuData[cpuData['국가코드'].isin(['CN']) | cpuData['국가코드'].isin(['EP']) | cpuData['국가코드'].isin(['US'])]

cpuData_KR['kkk'] = cpuData_KR['발명의 명칭'] + ' ' + cpuData_KR['요약'] + cpuData_KR['대표청구항']
cpuData_ENG['nltk'] = cpuData_ENG['발명의 명칭'] + ' ' + cpuData_ENG['요약'] + cpuData_ENG['대표청구항']

# 비교 데이터 전처리
for i in range(len(invalid_ENG['대표청구항'])):
  invalid_ENG['대표청구항'][i] = invalid_ENG['대표청구항'][0][3:]

invalid_KR['kkk'] = invalid_KR['발명의 명칭'] + ' ' + invalid_KR['요약'] + ' ' + invalid_KR['대표청구항']
invalid_ENG['nltk'] = invalid_ENG['발명의 명칭'] + ' ' + invalid_ENG['요약'] + ' ' + invalid_ENG['대표청구항']

kkmaCPU = Kkma() # 꼬꼬마 객체

def morph(input_data):  # pos(모든 형태소 분석)
    preprcessed = kkmaCPU.pos(input_data)
    return preprcessed

def get_wordData_KR(morpheme_pos):
    wordData = []
    for words, tags in morpheme_pos:
        #if (tags == 'NNG') or (tags == 'NNP') or (tags == 'OL'):
        if (("NN") or ("OL")) in tags:
            wordData.append(words) # NNG, NNP, OL만 뽑음

    return wordData

def get_wordData_ENG(morpheme_pos):
    wordData = []
    for words, tags in morpheme_pos:
        #if (tags == 'NNG') or (tags == 'NNP') or (tags == 'OL'):
        if (("NN")) in tags:
            wordData.append(words) # NN만 뽑음

    return wordData

# 유효 데이터 한국어 형태소 분석
morph_KR = [morph(x) for x in cpuData_KR['꼬꼬마'].values]
wordMaterialList_KR = [get_wordData_KR(kk) for kk in morph_KR]

# 유효 데이터 영어 형태소 분석 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
morph_ENG = [nltk.pos_tag(nltk.word_tokenize(x)) for x in cpuData_ENG['nltk'].values]
wordMaterialList_ENG = [get_wordData_ENG(ee) for ee in morph_ENG]

# 비교 데이터 한국어 형태소 분석
Invalid_morph_KR = [morph(x) for x in invalid_KR['kkk'].values]
Invalid_wordMaterialList_KR = [get_wordData_KR(kk) for kk in Invalid_morph_KR]

# 비교 데이터 영어 형태소 분석 
invalid_morph_ENG = [nltk.pos_tag(nltk.word_tokenize(x)) for x in invalid_ENG['nltk'].values]
invalid_wordMaterialList_ENG = [get_wordData_ENG(ee) for ee in invalid_morph_ENG]

# 유효 데이터 csv 형태로 저장
df_KR_valid = pd.DataFrame.from_records(wordMaterialList_KR)
df_KR_valid.to_csv('nounKR_csv.csv')
df_ENG_valid = pd.DataFrame.from_records(wordMaterialList_ENG)
df_ENG_valid.to_csv('nounKR_csv.csv')

# 비교 데이터 csv 형태로 저장
df_KR_invalid = pd.DataFrame.from_records(Invalid_wordMaterialList_KR)
df_KR_invalid.to_csv('invalid_nounKR.csv')
df_ENG_invalid = pd.DataFrame.from_records(invalid_wordMaterialList_ENG)
df_ENG_invalid.to_csv('invalid_nounENG.csv')

#데이터 불러오기
valid_KR = pd.read_csv('nounKR_csv.csv',) # 유효 한국
valid_ENG = pd.read_csv('nounENG_csv.csv') # 유효 영어
invalid_KR = pd.read_csv('invalid_nounKR.csv') # 비유효 한국어
invalid_ENG = pd.read_csv('invalid_nounENG.csv') # 비유효 영어

# 유효 데이터 전처리
stopwords = ['무엇', '정보', '시스템', '데이터', '도'] # 불용어 처리: 불용어는 실행때마다 수정 가능
cpuData_morph_list_KR = []
valid_list_KR = []
for i in range(len(valid_KR)):
    cpuData_morph_list_KR.append(valid_KR.iloc[i][1:].tolist())
for i in range(len(cpuData_morph_list_KR)):
    valid_list_KR.append([x for x in cpuData_morph_list_KR[i] if str(x) != 'nan'])

cpuData_morph_list_ENG = []
valid_list_ENG = []
for i in range(len(valid_ENG)):
    cpuData_morph_list_ENG.append(valid_ENG.iloc[i][1:].tolist())
for i in range(len(cpuData_morph_list_ENG)):
    valid_list_ENG.append([x for x in cpuData_morph_list_ENG[i] if str(x) != 'nan'])

for s in stopwords:
    valid_list_KR.remove(s)

# 비교 데이터 전처리
cpuData_morph_list_KR = []
invalid_list_KR = []
for i in range(len(invalid_KR)):
    cpuData_morph_list_KR.append(invalid_KR.iloc[i][1:].tolist())
for i in range(len(cpuData_morph_list_KR)):
    invalid_list_KR.append([x for x in cpuData_morph_list_KR[i] if str(x) != 'nan'])

cpuData_morph_list_ENG = []
invalid_list_ENG = []
for i in range(len(invalid_ENG)):
    cpuData_morph_list_ENG.append(invalid_ENG.iloc[i][1:].tolist())
for i in range(len(cpuData_morph_list_ENG)):
    invalid_list_ENG.append([x for x in cpuData_morph_list_ENG[i] if str(x) != 'nan'])

# 데이터 조절
# import random
# random.shuffle(invalid_list_KR)
# invalid_list_KR = invalid_list_KR[:1600]

# 유효 데이터 중복제거 한국
Deduplication_of_valid_KR = [] 
for kk in  valid_list_KR:
    for k in kk:
        Deduplication_of_valid_KR.append(k)
Deduplication_of_valid_KR = list(set(Deduplication_of_valid_KR))

# 유효데이터 중복제거 영어
Deduplication_of_valid_ENG = []
for ee in  valid_list_ENG:
    for e in ee:
        Deduplication_of_valid_ENG.append(e)
Deduplication_of_valid_ENG = list(set(Deduplication_of_valid_ENG))



# Doucment Frequency
def get_DF(valid_list, invalid_list, Deduplication_of_valid):
    valid = []
    for list_valid in valid_list:
        valid += list_valid

    invalid = []
    for list_invalid in invalid_list:
        invalid += list_invalid

    result = []
    result.append(("단어","DF","유효집단 단어 빈도 수","비교집단 단어 빈도 수", "유효집단 단어 수", "비교집단 단어 수"))

    for w in Deduplication_of_valid:
        valid_c = valid.count(w)
        valid_whole = len(valid)
        invalid_c = invalid.count(w)
        invalid_whole = len(invalid)
        result.append((w, (valid_c/valid_whole) -  (invalid_c/invalid_whole), valid_c, invalid_c, valid_whole, invalid_whole))

    return result

KR_result = get_DF(valid_list_KR, invalid_list_KR, Deduplication_of_valid_KR)
ENG_result = get_DF(valid_list_ENG, invalid_list_ENG, Deduplication_of_valid_ENG)

# DF 저장
df_KR_DF = pd.DataFrame.from_records(KR_result)
df_KR_DF.to_csv('filter_DF_KR.csv', encoding='utf-8-sig')
df_ENG_DF = pd.DataFrame.from_records(ENG_result)
df_ENG_DF.to_csv('filter_DF_ENG.csv', encoding='euc_kr')

# chi-squre
def get_chi_Number(Deduplication_List, Valid_List):
    chi_Number = []
    for x in Deduplication_List:
      x_count = 0
      for i in range(len(Valid_List)):
          if x in Valid_List[i]:
              x_count += 1
      chi_Number.append((x, x_count))

    return chi_Number

def get_chi_substract_Number(chi_List, count_of_Document):
    chi_substract_Number = []
    for nn, count in chi_List:
        chi_substract_Number.append((nn, count_of_Document - count))
    return chi_substract_Number

def get_chi_square(Data_T, N, A,B,C,D):
    chi_square = []
    
    for i in range(len(Data_T)):
        # N_test = A[i][1]+B[i][1]+C[i][1]+D[i][1]
        ad_bc = (A[i][1] * D[i][1]) - (C[i][1] * B[i][1])
        up_expression = N * (ad_bc ** 2)
        down_expression = (A[i][1] + C[i][1]) * (B[i][1] + D[i][1]) * (A[i][1] + B[i][1]) * (C[i][1] + D[i][1])
        chi_square.append((Data_T[i], up_expression / down_expression))
    return chi_square

# chi 계산(한국어)
chi_A_KR = get_chi_Number(Deduplication_of_valid_KR, valid_list_KR) # 부류에 속한 문서 중에서, 단어 t를 포함한 문서의 수 / 한국 
chi_B_KR = get_chi_Number(Deduplication_of_valid_KR, invalid_list_KR) # 부류에 속하지 않은 문서 중에서, 단어 t를 포함한 문서의 수 / 한국
chi_C_KR = get_chi_substract_Number(chi_A_KR, len(valid_list_KR)) # 부류에 속한 문서 중에서, 단어 t를 포함하지 않은 문서의 수
chi_D_KR = get_chi_substract_Number(chi_B_KR, len(invalid_list_KR))
N_KR = len(valid_list_KR) + len(invalid_list_KR)
chi_square_KR = get_chi_square(Deduplication_of_valid_KR, N_KR, chi_A_KR, chi_B_KR, chi_C_KR, chi_D_KR)

# chi 계산(영어)
chi_A_ENG = get_chi_Number(Deduplication_of_valid_ENG, valid_list_ENG) # 부류에 속한 문서 중에서, 단어 t를 포함한 문서의 수 / 미국
chi_B_ENG = get_chi_Number(Deduplication_of_valid_ENG, invalid_list_ENG) # 부류에 속하지 않은 문서 중에서, 단어 t를 포함한 문서의 수 / 미국 
chi_C_ENG = get_chi_substract_Number(chi_A_ENG, len(valid_list_ENG)) # 부류에 속한 문서 중에서, 단어 t를 포함하지 않은 문서의 수
chi_D_ENG = get_chi_substract_Number(chi_B_ENG, len(invalid_list_ENG))
N_ENG = len(valid_list_ENG) + len(invalid_list_ENG)
chi_square_ENG = get_chi_square(Deduplication_of_valid_ENG, N_ENG, chi_A_ENG, chi_B_ENG, chi_C_ENG, chi_D_ENG)

# Chi-square 저장
df_KR_CHI = pd.DataFrame.from_records(chi_square_KR)
df_KR_CHI.to_csv('filter_chi_square_KR.csv',encoding='utf-8-sig')
df_ENG_CHI = pd.DataFrame.from_records(chi_square_ENG)
df_ENG_CHI.to_csv('test_filter_chi_square_ENG.csv')


# 정보 이득
IPC_Number = [ipc[0:4] for ipc in cpuData['메인 IPC'].values]
Unique_IPC_Number = list(set(IPC_Number))
 
# IPC get
IPC_Probability = []
for u in Unique_IPC_Number:
    IPC_Probability.append((u, (IPC_Number.count(u) / len(IPC_Number))))

# 정보이득 get
All_nouns_KR = 0
for i in range(len(valid_list_KR)):
    All_nouns_KR += len(valid_list_KR[i])

cpuDF_KR = pd.DataFrame()
cpuDF_KR['Data'] = cpuData_KR['꼬꼬마']
cpuDF_KR['IPC'] = cpuData_KR['메인 IPC']

result_KR = []
for rows, AllDocument in enumerate(valid_list_KR):
    for t in AllDocument:
        # for t in Document:
        count_t_of_all = str(valid_list_KR).count(t) # 전체 문서에서 단어 t의 수
        if count_t_of_all == 0:
            count_t_of_all = 1
        P_t = count_t_of_all / All_nouns_KR # (len(valid_list_KR) + len(valid_list_ENG)) # 전체 문서에서 단어 t의 확률
        p_not_t = 1 - P_t # 전체 문서에서 단어 t가 아닌 확률
        sum_of_p_conditional_c_and_t = 0 # 조건부확률 P(c|t)의 합을 담을 공간
        sum_of_p_conditional_c_and_not_t = 0 # 조건부확률 P(c| not t)의 합을 담을 공간
        first_term = 0 # 첫째 항
        for ipc, P_c in IPC_Probability:
            if ipc == cpuDF_KR.iloc[rows,1][0:4]: # IPC가 같으면

                c_and_t = cpuDF_KR.iloc[rows,0].count(t) # 해당하는 IPC에 대한 단어 t의 수
                p_conditional_c_and_t = c_and_t / count_t_of_all # 해당하는 IPC에 대한 문서에서 단어 t의 수 / 전체 문서에서 단어 t의 수
                sum_of_p_conditional_c_and_t += (p_conditional_c_and_t * np.log(p_conditional_c_and_t)) # 조건부확률의 합 P(c|t) * log(P(c|t))

                c_and_not_t = len(AllDocument) - cpuDF_KR.iloc[rows,0].count(t)   # 문서 전체 단어 수에서 단어 t의 수를 뺌
                p_conditional_c_and_not_t = c_and_not_t / count_t_of_all    # P(c| not t)를 구함
                sum_of_p_conditional_c_and_not_t += (p_conditional_c_and_not_t * np.log(p_conditional_c_and_not_t)) # P(c| not t) * log(P(c| not t))

            first_term = first_term + (P_c * np.log(P_c))
        result_KR.append((t, (-1 * first_term) + (P_t * sum_of_p_conditional_c_and_t) + (p_not_t * sum_of_p_conditional_c_and_not_t)))

# 정보 이득(한국어) 저장
if len(result_KR) == 0:
    pass
else:
    result_df_KR_IG = pd.DataFrame.from_records(result_KR)
    result_df_KR_IG.to_csv('filter_log10_IG_KR.csv', encoding='utf-8-sig')
    result_df_KR_IG.to_excel('filter_log10_IG_KR.xlsx', encoding='utf-8-sig')


cpuDF_ENG = pd.DataFrame()
cpuDF_ENG['Data'] = cpuData_ENG['nltk']
cpuDF_ENG['IPC'] = cpuData_ENG['메인 IPC']

All_nouns_ENG = 0
for i in range(len(valid_list_ENG)):
    All_nouns_ENG += len(valid_list_ENG[i])

result_ENG = []
for rows, Document in enumerate(valid_list_ENG):
    for t in Document:
        count_t_of_all = str(valid_list_ENG).count(t) # 전체 문서에서 단어 t의 수
        if count_t_of_all == 0:
            count_t_of_all = 1
        P_t = count_t_of_all / All_nouns_ENG # 전체 문서에서 단어 t의 확률
        p_not_t = 1 - P_t # 전체 문서에서 단어 t가 아닌 확률
        sum_of_p_conditional_c_and_t = 0 # 조건부확률 P(c|t)의 합을 담을 공간
        sum_of_p_conditional_c_and_not_t = 0 # 조건부확률 P(c| not t)의 합을 담을 공간
        first_term = 0 # 첫째 항
        for ipc, P_c in IPC_Probability:
            if ipc == cpuDF_ENG.iloc[rows,1][0:4]: # IPC가 같으면

                c_and_t = cpuDF_ENG.iloc[rows,0].count(t) # 해당하는 IPC에 대한 단어 t의 수
                p_conditional_c_and_t = c_and_t / count_t_of_all # 해당하는 IPC에 대한 문서에서 단어 t의 수 / 전체 문서에서 단어 t의 수
                sum_of_p_conditional_c_and_t += (p_conditional_c_and_t * np.log(p_conditional_c_and_t)) # 조건부확률의 합 P(c|t) * log(P(c|t))

                c_and_not_t = len(Document) - cpuDF_ENG.iloc[rows,0].count(t)   # 문서 전체 단어 수에서 단어 t의 수를 뺌
                p_conditional_c_and_not_t = c_and_not_t / count_t_of_all    # P(c| not t)를 구함
                sum_of_p_conditional_c_and_not_t += (p_conditional_c_and_not_t * np.log(p_conditional_c_and_not_t)) # P(c| not t) * log(P(c| not t))

            first_term = first_term + (P_c * np.log(P_c))
        result_ENG.append((t, (-1 * first_term) + (P_t * sum_of_p_conditional_c_and_t) + (p_not_t * sum_of_p_conditional_c_and_not_t)))

# 정보 이득(영어) 저장
if len(result_KR) == 0:
    pass
else:
    result_df_ENG_IG = pd.DataFrame.from_records(result_ENG)
    result_df_ENG_IG.to_csv('filter_log10_IG_ENG.csv')
    result_df_ENG_IG.to_excel('filter_log10_IG_ENG.xlsx')

# 키워드 동향 그래프
# Graph_nounKR_excel은 nounKR_csv, nounENG_csv 파일에 각각 국가코드, 출원연도, 대표출원인, 중분류, 소분류 열을 붙여 엑셀파일로 변환한 파일
word_KR = pd.read_excel('Graph_nounKR_excel.xlsx')
word_ENG = pd.read_excel('Graph_nounENG_excel.xlsx', encoding='CP949')
year = list(range(2004,2018,1))

Middle_Classification = ["내장기능 대용기","마취호흡 기기","생체계측 기기",
        "수술치료 기기" ,"영상진단 기기","의료용경","의료용품/기구",
        "의료정보/관리기기","재활보조기기","정형용품","진료장치",
        "체외진단 기기","치과용기기","치료용보 조장치"]

Small_Classification_Biomeasurement = ["검안장치","근전도검사장치","기타 기기(키, 피부)",
        "뇌파검사장치","생체진단장치(임피던스, 한방진단)","심박측정장치(혈압, 심박, 맥박 측정장치)",
        "심전계","청력검사장치","청진기","체온측정장치","혈류계측장치(산소포화도, 혈류,심박출)",
        "호흡기능 검사장치(폐활량계, 호흡 측정기)"]

company_Classification = ['KONINKLIJKE PHILIPS N.V.',
                          'SAMSUNG ELECTRONICS CO., LTD.',
                          'NIKE, Inc.', 'SOTERA WIRELESS, INC.', 
                          'FITBIT, INC.', 'CARDIAC PACEMAKERS, INC.', 
                          'OMRON HEALTHCARE CO., LTD.','ZOLL MEDICAL CORPORATION', 
                          'adidas AG', 'Valencell, Inc.', 'SEIKO EPSON CO', 
                          'APPLE INC.', 'General Electric Company', 
                          'MASIMO CORPORATION', 'Nippon Telegraph and Telephone Corporation', 
                          'HITACHI, LTD.']

def get_item_to_list(DataFrame__):
    cpu_list = []
    result = []
    for i in range(len(DataFrame__)):
        cpu_list.append(DataFrame__.iloc[i,:].tolist())
    for i in range(len(cpu_list)):
        result.append([x for x in cpu_list[i] if str(x) != 'nan'])
    # print(result)
    return result

def keywordTrends(DF_word, find_word, find_year,Classification, middle_or_small, company):
    # middle_or_small가 True이면 '중분류', Flase이면 "소분류"
    if company == False:
        if middle_or_small == True:
            mid_or_small = '중분류'
        else:
            mid_or_small = '소분류'
    else:
        mid_or_small = '대표 출원인'
        
    Dic_of_DataFrame__ = {}
    for c in Classification:
        Dic_of_DataFrame__[c] = DF_word[ DF_word[mid_or_small].isin([c]) & DF_word['출원연도'].isin([find_year])]
   
   
    list_of_list_Middle_Classification = []
    for c in Classification:
        list_of_list_Middle_Classification.append(get_item_to_list(Dic_of_DataFrame__[c].iloc[:,6:]))
    
    Middle_Classification_list = []
    for i in range(len(list_of_list_Middle_Classification)):
        temp = []
        for j in range(len(list_of_list_Middle_Classification[i])):
            temp += list_of_list_Middle_Classification[i][j]
        Middle_Classification_list.append(temp)

    result = {}
    for i, c in enumerate(Classification):
        result[c] = Middle_Classification_list[i]

    # print(result)
    return result

def get_keywordTrend(DF_data, wanna_word, All_year, Classification_technology, TorF, company, Classification_Company):
    # TorF가 True이면 '중분류', Flase이면 "소분류"
    # company가 True이면 기술분류가 아닌 주요 출원인
    result = {}
    if company == False:
        for i in range(len(Classification_technology)):
            temp = []
            for j in range(len(All_year)):
                Document_Classification = keywordTrends(DF_data, wanna_word, All_year[j], Classification_technology, TorF, company) # 중분류에 해당하게 분류한 뒤 리스트에 해당 단어들을 일렬로 배열한 상태
                temp.append(Document_Classification[Classification_technology[i]].count(wanna_word))
            
            result[Classification_technology[i]] = temp
    else:
        for i in range(len(Classification_Company)):
            temp = []
            for j in range(len(All_year)):
                Document_Classification = keywordTrends(DF_data, wanna_word, All_year[j], Classification_Company, TorF, company) # 중분류에 해당하게 분류한 뒤 리스트에 해당 단어들을 일렬로 배열한 상태
                temp.append(Document_Classification[Classification_Company[i]].count(wanna_word))
            
            result[Classification_Company[i]] = temp
    # print(result)
    return result


words_KR = ['사용자', '운동', '측정', '센서', '건강', '신호', '관리', '생체', '서버', '상태'] # DF
# words_KR = ['데이터', '사용자', '센서', '측정', '관리', '건강', '수신', '서버', '제조', '형성'] # CHi-square
# words_KR = ['양력','남자','하키','끼니','총열량','나태','인내','작심','삼일','안배',] # 정보이득
words_ENG = ['module', 'sensor','monitoring', 'device', 'information', 'user', 'heart', 'body', 'signal','health'] # DF
# words_ENG = ['soil', 'implement','moisture', 'depth', 'adjustment', 'content', 'operating', 'characteristics', 'indicative','controller'] # CHI-square
# words_ENG = ['PelvisCenter','Hipleft','HipRight','KneeLeft','KneeRight','AnkleLeft','AnkleRight','FootLeft','FootRight','display/alarm'] # 정보이득

# 한국어 키워드 동향 그리기
# get_keywordTrend() 함수의 파라미터 True, False로 조정하고 plot()함수에 keyword_KR[]안에 어떤걸 넣느냐에 따라 다른 그래프가 그려짐
plt.figure(figsize=(12, 6))
for w in words_KR:
    keyword_KR = get_keywordTrend(word_KR, w, year, Middle_Classification, True, False, company_Classification)
    plt.plot(year, keyword_KR[Middle_Classification[7]], label=w)
plt.xlabel('출원연도')
plt.ylabel('단어 빈도')
plt.title("키워드 동향 : " + Middle_Classification[7])
plt.legend(fontsize='x-large')
plt.grid()
plt.show()

# 영어 키워드 동향 그리기
# get_keywordTrend() 함수의 파라미터 True, False로 조정하고 plot()함수에 keyword_KR[]안에 어떤걸 넣느냐에 따라 다른 그래프가 그려짐
plt.figure(figsize=(12, 6))
for w in words_ENG:
    keyword_ENG = get_keywordTrend(word_ENG, w, year, Middle_Classification, True, False, company_Classification)
    plt.plot(year, keyword_ENG[Middle_Classification[7]], label=w)
plt.xlabel('출원연도')
plt.ylabel('단어 빈도')
plt.title("키워드 동향 : " + Middle_Classification[7])
plt.legend(fontsize='x-large')
plt.grid()
plt.show()

