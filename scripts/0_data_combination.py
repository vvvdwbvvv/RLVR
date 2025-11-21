import re
import pandas as pd

# ROOT_DIR = '/content/drive/My Drive/Colab Notebooks/LLM-GAN_EHR/data/'
ROOT_DIR = '~'
df1 = pd.read_csv(f'{ROOT_DIR}/emergency.csv', low_memory=False)
df2 = pd.read_csv(f'{ROOT_DIR}/nursing.csv')

merged_df = pd.merge(df1, df2, on='病歷編號', how='inner')
merged_df.to_csv('emergency+nursing.csv', index=False)

filtered_df = pd.read_csv('emergency+nursing.csv', low_memory=False)
main_col = ['病歷編號','就診日期','診別','科別','科別名稱','性別','初複診','ICD101','ICD102','ICD103','ICD104','ICD105','ICD106','身高_x','體重_x','血壓','脈搏','主訴(S)','客觀(O)','診斷(A)','計畫(P)','評估單類別','評估日期','入院日','來源','過去病史','手術史','服藥史']
filtered_df[main_col].to_csv("emergency_nursing_with_maincol.csv", index=False)
# COVID戶外篩檢 院內同仁擴大採檢 門診戶外篩檢 門診戶外PCR篩檢 needs to be filtered out


exclude_phrases = ['COVID戶外篩檢', '院內同仁擴大採檢', '門診戶外篩檢', '門診戶外PCR篩檢']
pattern = re.compile('|'.join(map(re.escape, exclude_phrases)))

mask_drop = filtered_df['科別名稱'].astype('string').fillna('').str.contains(pattern, na=False)

kept = filtered_df.loc[~mask_drop, main_col]
kept.to_csv("emergency_nursing_with_maincol_nocovid.csv", index=False, encoding="utf-8-sig")

# maping based on icd10_code 2020
# transform ICD to augmented text
fact = pd.read_csv("emergency_nursing_with_maincol_nocovid.csv", low_memory=False)
lookup = pd.read_csv(f"{ROOT_DIR}/diagnosis.csv")[["Code", "LongDescription"]]

joined = fact.copy()
for col in ["ICD101", "ICD102", "ICD103", "ICD104", "ICD105", "ICD106"]:
    joined = joined.merge(lookup, left_on=col, right_on="Code", how="left").drop(columns="Code").rename(columns={"LongDescription": f"{col}_desc"})

joined.to_csv("map.csv", index=False)