import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.color_palette("Spectral")

plt.rcParams.update({'font.size': 28,})

min_coord = 1
max_coord = 480

colors = {
    'iCaRL': 'blue',
    'Continual-CLIP': 'deepskyblue',
    'CoOp': 'gray',
    'CLIP-Adapter': 'lime',
    'AttriCLIP': 'orange',
    'DualPrompt': 'fuchsia',
    'L2P': 'brown',
    'PROOF': 'green',
    'TAw-UB': 'black',
    'Ours': 'red',
    "Ours + CoOp": 'black',
    "Ours + AttriCLIP": 'indigo'
}

def reverse_scale(coord):
    return max_coord - coord

def get_max_param_num(zsclip_percentage):
    max_percent = 149.6 / zsclip_percentage
    return max_percent

def get_num_of_params(coord_dict):
    coord_dict = {method: reverse_scale(c) for method, c in coord_dict.items()}
    coord_dict_percentage = {method: (value - min_coord) / (max_coord - min_coord) for method, value in coord_dict.items()}
    max_param = get_max_param_num(coord_dict_percentage['Continual-CLIP'])
    param_nums = {method: value * max_param for method, value in coord_dict_percentage.items()}
    print(param_nums)
    return param_nums
    
def draw_plot(data):
    # Set the style for the plot
    # sns.set(style="whitegrid")

    # Create a figure with the specified size
    plt.figure(figsize=(10, 6))

    hues = [colors[key_] for key_ in list(data.keys())]
    # Create the bar plot with a unique color for each bar
    barplot = sns.barplot(x=list(data.keys()), y=list(data.values()), palette=hues, hue=list(data.keys()), hue_order=list(data.keys()), dodge=False)
    for i in range(len(data)):
        barplot.bar_label(barplot.containers[i], fmt='%.1f', rotation=20, label_type='center',  fontsize=22)
  
    # Set the y-axis range to start from 50
    barplot.set(ylim=(0, max(data.values()) + 50))
    barplot.set_xticklabels(list(data.keys()), rotation = 25)

    # Add a legend across 3 columns at the top
    barplot.legend(loc="upper center", bbox_to_anchor=(0.55, 1.15), ncol=2, prop={'size': 25}, frameon=False)
    # Set labels and title
    # plt.xlabel('Methods')
    plt.ylabel(f'Parameters (in millions)')
    barplot.spines[['right', 'top']].set_visible(False)
    barplot.set_xticklabels([])
    # RED
    barplot.set_xticks([])
    plt.tight_layout()
    plt.savefig("parameter_comparison.pdf")
    # Show the plot
    plt.show()

param_nums = {'iCaRL': 299.2}
param_nums['CLIP-Adapter'] = 149.8
param_nums['Continual-CLIP'] = 149.6
coord_list = {'L2P': 295, 'DualPrompt': 295, 'PROOF': 314,  'Continual-CLIP': 320, 'CoOp': 320,} #, 'iCaRL': 161}
param_nums.update(get_num_of_params(coord_list))
param_nums['AttriCLIP'] = 149.7
# param_nums['PROOF'] = 153.1
param_nums['Ours'] = 159.5
# param_nums['iCaRL'] = 299.2

draw_plot(param_nums)