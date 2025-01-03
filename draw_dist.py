import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import make_interp_spline

def to_percent(y, position):
    return f"{y/720:.3f}"

def read_floats_from_file(file_path):
    with open(file_path, 'r') as f:
        data = [float(line.strip()) for line in f.readlines()]
    return data

def plot_histogram(file2, file3):
    data_file2 = read_floats_from_file(file2)
    data_file3 = read_floats_from_file(file3)

    n_bins = 50
    hist, bins = np.histogram(data_file2, bins=n_bins, density=True)
    
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 300)
    spline = make_interp_spline(bin_centers, hist, k=3)
    y_smooth = spline(x_smooth)
    plt.plot(x_smooth, y_smooth, color='#1f77b4', label='WFPR')
    plt.fill_between(x_smooth, y_smooth, alpha=0.2, color='#1f77b4')

    np.random.seed(0)
    sample_size = 5
    sampled_data_file3 = np.random.choice(data_file3, size=sample_size, replace=False)
    y = np.zeros(sample_size)
    plt.scatter(sampled_data_file3, y, marker='x', color='#ff7f0e', label='File 3 Sample', s=50, zorder=3)

    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xlim(0, 0.2)
    plt.ylim(-0.005, plt.gca().get_ylim()[1])
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.savefig("test2.png", dpi=300, bbox_inches='tight')

file2 = "/home/data/lpz/CompositeAttack/datas_for_histogram/Posion_model_Random_label.txt"
file3 = "/home/data/lpz/CompositeAttack/datas_for_histogram/Posion_model_True_label_Fake_Match.txt"
plot_histogram(file2, file3)
