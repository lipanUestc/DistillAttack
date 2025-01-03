import os
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm

# Set font properties
font = fm.FontProperties(family="Times New Roman")

def to_percent(y, position):
    return f"{100 * y:.0f}%"
    
# Apply font properties globally
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 5})

def read_data_from_txt(file_path):
    with open(file_path, 'r') as file:
        return [float(line.strip()) for line in file]

file1 = "/home/data/lpz/CompositeAttack/datas_for_histogram/Clean_model_Clean_label.txt"
file2 = "/home/data/lpz/CompositeAttack/datas_for_histogram/Posion_model_Random_label.txt"
file3 = "/home/data/lpz/CompositeAttack/datas_for_histogram/Posion_model_True_label_Fake_Match.txt"

files = [file1, file2]#, file3]

data = [read_data_from_txt(file) for file in files]

labels = ['Clean Models', 'Watermark Models']

plt.figure(figsize=(3.5, 2))

# Set patch_artist=True to access the box objects
bp = plt.boxplot(data, labels=labels, sym='', widths=0.3, whis=3.6, patch_artist=True)

plt.ylim(0, 0.4)

threshold = 0.3

plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')

# Define colors for each box (all boxes in white)
colors = ['white', 'white', 'white']

# Assign colors to the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Create a font property for the legend with a smaller font size
legend_font = font_manager.FontProperties(family='serif', style='normal', size=10)

# Add legend with only the threshold line
plt.legend([plt.Line2D([0], [0], color='r', linestyle='--')], ['Threshold'], loc='upper right', prop=font)

# Adjust axis labels and tick font sizes
formatter = FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(formatter)

plt.gca().xaxis.set_tick_params(labelsize=12.5)
plt.gca().yaxis.set_tick_params(labelsize=13)

plt.savefig("test.png", dpi=300, bbox_inches='tight')

