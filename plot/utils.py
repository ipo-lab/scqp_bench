import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def plot_profile(folder, what='comp_time', token=None):
    folder = os.path.expanduser(folder)
    # --- grep files
    files = os.listdir(folder)
    files = [files[j] for j in range(len(files)) if what in files[j]]
    if token is not None:
        files = [files[j] for j in range(len(files)) if token in files[j]]

    # --- get n values
    ns = [file.replace('.csv','') for file in files]
    ns = [int(file.split('_')[3]) for file in ns]
    idx = np.argsort(ns)
    ns = [ns[i] for i in idx]
    files = [files[i] for i in idx]

    # --- load data:
    med_runtime = []
    error_runtime = []
    for i in range(len(files)):
        df = pd.read_csv(folder+files[i])
        med_runtime.append(np.mean(df.values,axis=0,keepdims=True))
        error_runtime.append(np.std(df.values, axis=0,keepdims=True) / len(df)**0.5)

    # --- plots:
    # --- data prep:
    med_runtime = np.concatenate(med_runtime)
    med_runtime = pd.DataFrame(med_runtime)
    med_runtime.columns = df.columns
    med_runtime.index = ns
    error_runtime = np.concatenate(error_runtime)
    error_runtime = pd.DataFrame(error_runtime)
    error_runtime.columns = df.columns
    error_runtime.index = ns

    # --- main plot:
    color = ["#1E90FF", "#3D9140"]
    med_runtime.plot.line(logy=True, logx=True, color=color, linewidth=4, fontsize=11)

    plt.fill_between(ns, med_runtime['OSQP'] - error_runtime['OSQP'],
                     med_runtime['OSQP'] + med_runtime['OSQP'], alpha=0.25, color=color[0])
    plt.fill_between(ns, med_runtime['SCQP'] - error_runtime['SCQP'],
                     med_runtime['SCQP'] + med_runtime['SCQP'], alpha=0.25, color=color[1])
    plt.ylabel('time (s)', fontsize=14)
    plt.xlabel('Number of Decision Variables', fontsize=14)






