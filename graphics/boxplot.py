# %%
import pandas as pd
import seaborn as sns
import glob
import matplotlib.pyplot as plt

# %%
files = glob.glob('results_data/*')
df_final = pd.DataFrame()

for file in files:
    name = file.split('/')[1].split(' ')[0]
    metric = file.split(' ')[1].split('.')[0]

    df = pd.read_csv(file)
    df = df.iloc[:, [1]]
    df = df.rename(columns={df.columns[0]: 'value'})
    df['metric'] = metric
    df['name'] = name
    df_final = pd.concat([df_final, df])

# %%

df_plot = df_final[df_final.metric == 'MAE']
sns.boxplot(data=df_plot, x='name', y='value')
plt.xticks(rotation=15)
plt.show()

