import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import make_interp_spline
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
import matplotlib.font_manager as fm
from matplotlib.ticker import PercentFormatter


font = fm.FontProperties(family="Times New Roman")

def to_percent(y, position):
    return f"{y/720:.3f}"

def read_floats_from_file(file_path):
    with open(file_path, 'r') as f:
        data = [float(line.strip()) for line in f.readlines()]
    return data

def plot_histogram(file2, file3):
    with open(file2, "r") as f:
        dataset1 = np.array([float(line.strip()) for line in f.readlines()])
    
    with open(file3, "r") as f:
        dataset2 = np.array([float(line.strip()) for line in f.readlines()])
        sample_size = 20
        dataset2 = np.random.choice(dataset2, size=sample_size, replace=False)

    
    kde = gaussian_kde(dataset1)
    x_vals = np.linspace(0, 0.1, 1000)
    density = kde(x_vals)
    
    density /= 720
    
    norm = Normalize(vmin=density.min(), vmax=density.max())


    fig, ax = plt.subplots(figsize=(8, 1))
    

    for i in range(len(x_vals) - 1):
        plt.fill_between([x_vals[i], x_vals[i + 1]], 0, 1, color=plt.cm.inferno(norm(density[i])))
    

    for sample in dataset2:
        plt.plot([sample, sample], [0, 1], color="white", linewidth=0.4)
    
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    

    sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, label="Density", orientation="vertical", pad=0.02)
    
    cbar.outline.set_linewidth(0.3)
    cbar.set_label("Density", fontsize=16, fontproperties=font) 
    cbar.ax.tick_params(labelsize=12)
    ax.tick_params(axis="x", labelsize=16)
    ax.set_xlabel("PRMS", fontproperties=font, fontsize=16)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.savefig("test_3.png", dpi=500, bbox_inches='tight')

file2 = "/home/data/lpz/CompositeAttack/datas_for_histogram/Posion_model_Random_label.txt"
file3 = "/home/data/lpz/CompositeAttack/datas_for_histogram/Posion_model_True_label_Fake_Match.txt"
plot_histogram(file2, file3)