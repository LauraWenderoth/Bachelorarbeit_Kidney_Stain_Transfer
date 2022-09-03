import glob

import numpy as np
import pandas as pd

dir = "/home/laurawenderoth/Documents/kidney_microscopy/CycleGanPytorch"
glom_paths = glob.glob(dir+"/results_data/multiple_seed/glo_podo/*glomerulus.xlsx")
podo_paths = glob.glob(dir+"/results_data/multiple_seed/glo_podo/*podocytes.xlsx")
def create_exel(paths,column_name,save_name):
    runs = {}
    for path in glom_paths:
        prefix = path.split("/")[-1]
        prefix = prefix.split("_seed")[0]
        if prefix not in runs.keys() and "Hamburg" not in prefix:
            runs[prefix] = []

    for path in paths:
        prefix = path.split("/")[-1]
        prefix = prefix.split("_seed")[0]
        if "Hamburg" not in prefix:
            df = pd.read_excel(path,engine='openpyxl')
            dice = df[column_name]
            mean_dice = dice.mean()
            runs[prefix].append(mean_dice)
    runs_glom = {}
    for key in runs.keys():

        means = np.asarray(runs[key])
        mean = means.mean()
        std = means.std()
        runs_glom[key+ " mean"] = mean
        runs_glom[key + " std"] = std
    df = pd.DataFrame(data=runs_glom, index=[0])
    df.to_excel(dir+"/results_dice/results/"+save_name+".xlsx")

#create_exel(glom_paths,"Network dice pixel","glomerulus")
create_exel(podo_paths,"Network dice pixel","podp_pixel")
create_exel(podo_paths,"Network dice object","podo_object")