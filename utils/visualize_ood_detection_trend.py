import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams.update({'font.size': 16})
sns.color_palette("Spectral")
# sns.set(font_scale=15)
plt.rcParams.update({'font.size': 20,})

datum = {
#     "Softmax": {
# # softmax
# "FPR95" : {
#     "ZS-CLIP": [62.56, 71.31, 73.1, 76.37, 77.02, 80.08, 80.87, 85.75, 85.9],
#     "ZS-CLIP+Ours": [83.16, 80.25, 79.5, 81.48, 81.3, 86.15, 84.93, 86.45, 85.0],
#     # "Ours":             [65.14, 81.74, 82.17, 85.33, 85.88, 88.7, 87.33, 89.5, 89.7]
# },
# "AUROC" : {
#     "ZS-CLIP": [87.83, 84.13, 83.68, 80.86, 81.43, 79.31, 76.94, 74.54, 73.8],
#     "ZS-CLIP+Ours": [57.86, 57.75, 57.79, 59.28, 60.66, 57.67, 58.62, 58.73, 61.43],
#     # "Ours":             [65.95, 59.01, 58.23, 57.25, 57.4, 56.16, 56.94, 56.3, 57.05]
# },
# "AUPR" : {
#     "ZS-CLIP": [58.33, 64.09, 74.25, 76.64, 82.75, 85.58, 88.5, 92.2, 96.14],
#     "ZS-CLIP+Ours": [11.66, 22.92, 33.79, 45.22 , 56.25, 64.12, 74.01, 83.06, 92.26],
#     # "Ours":             [14.11, 23.51, 34.06, 44, 54.19, 63.31, 73.21, 82.21, 91.4]
# }},

# energy
"Energy":{
"FPR95" : {
    "Continual-CLIP": [60.42, 69.91, 73.51, 76.35, 76.82,81.23, 84.17, 87.3, 86.3],
    "CoOp": [39.99, 60.02, 59.33,61.47, 64.88, 76.02, 80.03, 83.3, 76.2],
        "Ours": [47.32, 70.08, 62.31, 69.72, 67.08, 72.72, 74.37, 81.05, 73.8],
        "Ours w/o VI": [52.63, 63.32, 63.41, 68.38, 64.48, 73.08, 76.53, 80.25,77.4],
    "Ours + CoOp":  [38.12, 56.39, 58.94, 65.22, 62.9, 67.5, 73.07, 78.05, 63.9],
        "Ours + CoOp (w/o VI)": [48.18, 62.44, 57.13, 63.22, 64.4, 69.95, 76.6, 78.8, 75.2]
},
"AUROC" : {
    "Continual-CLIP": [85.29, 80.25, 78.21, 75.66, 75.41, 72.04, 67.98, 64.8, 70.48],
    "CoOp": [90.42, 84.15, 84.79, 81.52, 81.13, 77.43, 73.17, 70.84, 77.93],
        "Ours": [88.9, 84.62, 85.31, 82.1, 83.58, 79.87, 78.4, 75.17, 81.93],
                "Ours w/o VI": [86.16, 84.58, 85.55, 83.37, 85.46, 80.22, 77.96, 77.11, 80.18],
    "Ours + CoOp": [91.36, 86.76, 85.41, 83.72, 84.79, 81.67, 78.57, 76.57, 84.7],
        "Ours + CoOp (w/o VI)": [88.26, 85.36, 84.69, 83.18, 82.97, 80.19, 77.64, 76.72, 78.8]
},
"AUPR" : {
    "Continual-CLIP": [40.91, 52.36, 61.82, 67.72, 74.66, 78.28, 82.02, 87.15, 95.09],
    "CoOp": [57.83, 60, 71.21, 73.55, 80.47, 83.42, 85.45, 90.02, 96.62],
        "Ours": [51.69, 65.11, 73.97, 76.32, 84.6, 85.81, 89.03, 91.89, 97.41],
                "Ours w/o VI": [42.7, 61.93, 74.69, 78.86, 86.75, 86.45, 89.09, 92.74, 96.67],
    "Ours + CoOp": [59.15, 67.08, 72.66, 78.52, 85.39, 86.92, 88.86, 92.28, 97.83],
    "Ours + CoOp (w/o VI)": [50.15, 63.8, 68.96, 76.22, 83.07, 85.52, 88.49, 92.42, 96.68]
}}
}

# Set the style for the plots
# sns.set(style="whitegrid")

colors = {
    'iCaRL': 'blue',
    'Continual-CLIP': 'deepskyblue',
    'CoOp': 'gray',
    'CLIP-Adapter': 'purple',
    'Ours + CoOp (w/o VI)': 'green',
    'DualPrompt': 'fuchsia',
    'L2P': 'brown',
    'Ours w/o VI': 'orange',
    'TAw-UB': 'black',
    'Ours': 'red',
    "Ours + CoOp": 'black',
    "Ours + AttriCLIP": 'indigo'
}


markers = {
    "iCaRL": "o",
    "Continual-CLIP": "*",
    "CoOp": "D",
    "CLIP-Adapter": "h",
    "Ours + CoOp (w/o VI)": "P",
    "DualPrompt": "v",
    "L2P": "s",
    "Ours w/o VI": "^",
    "Ours": "X",
    "Ours + CoOp": "p",
    "Ours + AttriCLIP": ">"
}

# data = data["softmax"]
def ood_detection_plot(datum, dataset_name="cifar100"):
    step_size = 20 if dataset_name == "imagenet-r" else 10
    x_labels = list(range(0, step_size*10+1, step_size))[1:-1] #if dataset_name == "imagenet-r" else list(range(0, 101, 10))[1:]
    for j, metric in enumerate(datum.keys()):
        print(f"{'== ' * 30}\n Confidence Metric: {metric}")
        # Create subplots with three plots
        fig, axes = plt.subplots(3, 1, figsize=(10, 18))

        data = datum[metric]
        # Plot each dictionary in a subplot
        for i, element in enumerate(data.keys()):
            print(f"\nEvaluation Metric: {element}")
            ax = axes[i]
            for k, (key, values) in enumerate(data[element].items()):
                print(f"Method: {key}, Average: {np.mean(values)} ")
                sns.lineplot(x=x_labels, y=values, ax=ax, label=key, marker=markers[key], color=colors[key], linewidth=2., markersize=17)

            # Set labels and title for each subplot
            ax.set_xlabel('Number of Classes')
            downarrow = r'$\downarrow$'
            uparrow = r'$\uparrow$'
            ax.set_ylabel(f"{element}{downarrow if element == 'FPR95' else uparrow}")
            # ax.set_title(f'{element}')
            # ax.legend(ncol=1, fontsize=14, loc='lower right' if element != "AUROC" else "lower left", frameon=False, framealpha=0.9)
            ax.get_legend().remove()
            # ax.set_xticklabels(x_labels)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            ax.yaxis.grid(color='lightgrey', linestyle='dashed', linewidth=0.3, which='major')
            ax.xaxis.grid(color='lightgrey', linestyle='dashed', linewidth=0.3, which='major')

            ax.yaxis.grid(color='aliceblue', linestyle=':', linewidth=0.3, which='minor')
            ax.xaxis.grid(color='aliceblue', linestyle=':', linewidth=0.3, which='minor')

            ax.minorticks_on()
            
        lines_labels = [ax.get_legend_handles_labels() for i, ax in enumerate(fig.axes) if i == 0]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc='upper center', ncol=len(labels)//2, frameon=True, bbox_to_anchor=(0.52, 0.95), fontsize=18)
        # Adjust spacing and display the subplots
        # plt.tight_layout()
        plt.minorticks_on()
        # fig.suptitle(f'{metric}')
        plt.savefig(f"OOD_{metric}.pdf")
        plt.show()

ood_detection_plot(datum)