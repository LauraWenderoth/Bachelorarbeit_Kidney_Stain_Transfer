# %%
import pandas as pd
import seaborn as sns
import glob
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

sns.set()

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
labels = ['L1 Patches', 'L1 Reset Patches', 'Van Patches']


def boxplot(df: pd.DataFrame = df_final, metric: str = 'MAE', labels: list = []) -> None:
    df_plot = df[df_final.metric == metric]
    sns.boxplot(data=df_plot, x='name', y='value',
                showmeans=True,
                meanprops={"marker": "o",
                           "markerfacecolor": "white",
                           "markeredgecolor": "black",
                           "markersize": "5"})
    sns.set(rc={'figure.figsize': (15, 9)})
    sns.set_style('whitegrid')
    plt.xticks(np.arange(3), labels)
    plt.xlabel('Type')
    plt.ylabel(metric)
    tikzplotlib.save(f"graphics/{metric}.tex", axis_height='8cm', axis_width='10.5cm')
    plt.show()


boxplot(df_final, 'FID', labels)
boxplot(df_final, 'MAE', labels)
boxplot(df_final, 'MSE', labels)
boxplot(df_final, 'SSIM', labels)
