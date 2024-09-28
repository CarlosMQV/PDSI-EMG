import pandas as pd
import matplotlib.pyplot as plt

#Importamos los datos del sujeto 1
df1 = pd.read_parquet('datasets/ninapro/DB6_s1_a/S1_D1_T1.parquet', engine='pyarrow')
df2 = pd.read_parquet('datasets/ninapro/DB6_s1_a/S1_D1_T2.parquet', engine='pyarrow')
df3 = pd.read_parquet('datasets/ninapro/DB6_s1_a/S1_D2_T1.parquet', engine='pyarrow')
df4 = pd.read_parquet('datasets/ninapro/DB6_s1_a/S1_D2_T2.parquet', engine='pyarrow')
df5 = pd.read_parquet('datasets/ninapro/DB6_s1_a/S1_D3_T1.parquet', engine='pyarrow')
df6 = pd.read_parquet('datasets/ninapro/DB6_s1_a/S1_D3_T2.parquet', engine='pyarrow')
df_s1 = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
#Importamos los datos del sujeto 2
df1 = pd.read_parquet('datasets/ninapro/DB6_s2_a/S2_D1_T1.parquet', engine='pyarrow')
df2 = pd.read_parquet('datasets/ninapro/DB6_s2_a/S2_D1_T2.parquet', engine='pyarrow')
df3 = pd.read_parquet('datasets/ninapro/DB6_s2_a/S2_D2_T1.parquet', engine='pyarrow')
df4 = pd.read_parquet('datasets/ninapro/DB6_s2_a/S2_D2_T2.parquet', engine='pyarrow')
df5 = pd.read_parquet('datasets/ninapro/DB6_s2_a/S2_D3_T1.parquet', engine='pyarrow')
df6 = pd.read_parquet('datasets/ninapro/DB6_s2_a/S2_D3_T2.parquet', engine='pyarrow')
df_s2 = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

# Eliminamos las columnas 9 y 10
df1 = df1.drop(columns=[df1.columns[8], df1.columns[9]])
df2 = df2.drop(columns=[df2.columns[8], df2.columns[9]])
df3 = df3.drop(columns=[df3.columns[8], df3.columns[9]])
df4 = df4.drop(columns=[df4.columns[8], df4.columns[9]])
df5 = df5.drop(columns=[df5.columns[8], df5.columns[9]])
df6 = df6.drop(columns=[df6.columns[8], df6.columns[9]])

