import matplotlib.pyplot as plt
import numpy as np

# Set up the figure and axes with increased size
fig, ax = plt.subplots(figsize=(4, 1.75))

# Set up bar width and indices
bar_width = 0.25
gap = 0.05  # Add a gap between the groups of bars
index1 = np.array([0, 1, 2 + gap, 3 + gap])  # Add a gap to indices
index2 = index1 + bar_width
index3 = index2 + bar_width

# Define lighter and more elegant colors
colors = ['#f0a39d', '#8ab4c1', '#f0e39d']
#hatches = ['', '///', '...']

# Add a border around the bars
border_color = 'black'
border_width = 1.25

# Plot the bars
ax.bar(index1, [81.37, 81.92, 81.06, 81.42], bar_width, label='Accuracy (%)', color=colors[0], edgecolor=border_color, linewidth=border_width)#, hatch = hatches[0])
ax.bar(index2, [100.00, 100.00, 80.62, 81.86], bar_width, label='WSR (%)', color=colors[1], edgecolor=border_color, linewidth=border_width)#, hatch = hatches[1])
ax.bar(index3, [51.68, 5.32, 48.10, 5.26], bar_width, label='WFPR (%)', color=colors[2], edgecolor=border_color, linewidth=border_width)#, hatch = hatches[2])

# Set up ticks and labels
ax.set_xticks(index2)
ax.set_xticklabels(['w/o $L_{eva}$', 'w/ $L_{eva}$', 'w/o $L_{eva}$', 'w/ $L_{eva}$'], fontsize=10)
ax.xaxis.set_label_position('top')  # Move xlabel to the top
ax.set_xlabel('Victim Model              Extracted Model', fontsize=12,fontname="Times New Roman", labelpad=8)

# Adjust the ytick label size
ax.tick_params(axis='y', which='major', labelsize=10)

# Add horizontal gridlines with light gray color
ax.yaxis.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.8)
ax.set_axisbelow(True)
ax.set_ylim(0, 100)

# Set up legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, prop={'family': 'Times New Roman', 'size': 9})  # Place legend at the bottom in a single row
  # Place legend at the bottom in a single row

# Customize the plot borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_color('#333333')
ax.spines['left'].set_color('#333333')

# Show the plot
plt.savefig("test_qqqqqq.png", dpi=300, bbox_inches='tight')


