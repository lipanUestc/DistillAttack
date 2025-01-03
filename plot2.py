import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

subtitle_font2 = FontProperties(size=15)


def plot_from_multiple_txt_combined_legend_each(file_paths, plot_titles):
    # Define the font properties for Times New Roman for subtitles
    subtitle_font = FontProperties(size=20)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'hspace': 0.4, 'wspace': 0.5})  # 2x2 grid layout
    axs = axs.flatten()  # Flatten the 2x2 grid to easily iterate over it

    for i, file_path in enumerate(file_paths):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Parsing the data
        data = []
        for line in lines:
            parts = line.split(',')
            poison = float(parts[0].split(':')[1])
            loss = float(parts[1].split(':')[1])
            data.append((poison, loss))

        # Separate the data into two lists for plotting
        poison_values, loss_values = zip(*data)
        
        max_poison = max(poison_values)

        # Creating twin axes for poison values (as percentage)
        ax2 = axs[i].twinx()

        # Plotting on each subplot
        loss_line, = axs[i].plot(loss_values, label="Loss", markersize=4, linestyle='-', color='green')
        poison_line, = ax2.plot([p for p in poison_values], label="WSR", markersize=4, linestyle='-', color='orange')

        # Adding a thin horizontal line for max poison value
        ax2.axhline(y=max_poison, linestyle='-', linewidth=1, color='gray')

        # Setting labels
        axs[i].set_xlabel("Epoch", fontproperties=subtitle_font2)
        axs[i].set_ylabel("Loss", fontproperties=subtitle_font2)
        ax2.set_ylabel("WSR (%)", fontproperties=subtitle_font2)

        # Setting the y-axis range for Poison
        ax2.set_ylim(0, 50)

        # Adding combined legend in each plot
        lines = [loss_line, poison_line]
        labels = [line.get_label() for line in lines]
        axs[i].legend(lines, labels, loc='upper right', prop={'size': 13}) 
        axs[i].tick_params(axis='both', which='major', labelsize=16)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        # Adjusting the position of subtitle
        axs[i].text(0.5, 1.055, plot_titles[i], ha='center', va='center', fontproperties=subtitle_font, transform=axs[i].transAxes)

    # Adjust layout with increased vertical space between rows
    plt.subplots_adjust(hspace=1)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig('4plot.png')
    plt.show()

# Example usage with dummy file paths
file_paths = ["Plus_cifar10.txt", "MGDA_cifar10.txt", "OurMGDANoFactor_cifar10.txt", 'OurMGDA_cifar10.txt']
plot_titles = ["Fixed Weight", "MGDA", "GOATS (No Factor)", "GOATS"]
plot_from_multiple_txt_combined_legend_each(file_paths, plot_titles)  # Replace with your actual file paths

