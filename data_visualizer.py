import pandas as pd
import matplotlib.pyplot as plt

# Dataframes del sujeto 1
df1 = pd.read_parquet('datasets/ninapro/DB6_s1_a/S1_D1_T1.parquet', engine='pyarrow')
df2 = pd.read_parquet('datasets/ninapro/DB6_s1_a/S1_D1_T2.parquet', engine='pyarrow')
df3 = pd.read_parquet('datasets/ninapro/DB6_s1_a/S1_D2_T1.parquet', engine='pyarrow')
df4 = pd.read_parquet('datasets/ninapro/DB6_s1_a/S1_D2_T2.parquet', engine='pyarrow')
df5 = pd.read_parquet('datasets/ninapro/DB6_s1_a/S1_D3_T1.parquet', engine='pyarrow')
df6 = pd.read_parquet('datasets/ninapro/DB6_s1_a/S1_D3_T2.parquet', engine='pyarrow')
df_s1 = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
# Dataframes del sujeto 2
df1 = pd.read_parquet('datasets/ninapro/DB6_s2_a/S2_D1_T1.parquet', engine='pyarrow')
df2 = pd.read_parquet('datasets/ninapro/DB6_s2_a/S2_D1_T2.parquet', engine='pyarrow')
df3 = pd.read_parquet('datasets/ninapro/DB6_s2_a/S2_D2_T1.parquet', engine='pyarrow')
df4 = pd.read_parquet('datasets/ninapro/DB6_s2_a/S2_D2_T2.parquet', engine='pyarrow')
df5 = pd.read_parquet('datasets/ninapro/DB6_s2_a/S2_D3_T1.parquet', engine='pyarrow')
df6 = pd.read_parquet('datasets/ninapro/DB6_s2_a/S2_D3_T2.parquet', engine='pyarrow')
df_s2 = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

#Creamos un solo dataframe y eliminamos columnas 9 y 10
df = pd.concat([df_s1, df_s2], ignore_index=True)
df.drop(columns=[df.columns[8], df.columns[9]])

conteo_por_estado = df[df.columns[-1]].value_counts()
