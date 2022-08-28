# %%
import pandas as pd

df = pd.read_csv('graphics/wandb_export_2022-08-28T19_41_21.157+02_00.csv')

for col in df.columns:
    if '__' not in col:
        print(f'Col: {col}')
        df_temp = df[~df[col].isna()]
        df_temp = df_temp.reset_index()
        print(df_temp[col].idxmax() + 1)
