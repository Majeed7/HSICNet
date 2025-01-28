import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams.update({
    'font.size': 10,     # Set default font size
    'font.weight': 'bold',  # Set default font weight to bold
    'axes.labelweight': 'bold',  # Ensure the axis labels are bold
    'axes.titleweight': 'bold',  # Ensure the titles are bold
    'figure.titleweight': 'bold',  # Bold for suptitle if you use fig.suptitle()
    'xtick.labelsize': 9,  # Font size for X-tick labels
    'ytick.labelsize': 12,  # Font size for Y-tick labels
    'xtick.major.size': 5,  # Length of major ticks
    'ytick.major.size': 5,  # Length of major ticks
    'xtick.minor.size': 3,  # Length of minor ticks
    'ytick.minor.size': 3   # Length of minor ticks
})

# Load the Excel file
gp = False
file_path = 'results/fs_real/incremental_features_rf_bigestimate.xlsx'
if gp == True:
    file_path = 'results/fs_real/incremental_features_gp.xlsx'
excel_data = pd.ExcelFile(file_path)

is_classification = False

# sheet_names_classification = ['sonar', 'nomao', 'wisconsin']
# if is_classification:
#     sheet_names = sheet_names_classification
#     performance_metric = "Mean Absolute Percentage Error"
# else:
#     sheet_names = result = [item for item in excel_data.sheet_names if item not in sheet_names_classification]
#     performance_metric = "Accuracy"

sheet_names = excel_data.sheet_names



# Initialize subplots
col_no = 3
row_no = int(np.ceil(len(sheet_names)/col_no))
fig, axes = plt.subplots(row_no, col_no, figsize=(24 * col_no, 5 * row_no) )

axes = axes.flatten()
line_styles = ['-', '--', '-.', ':', 'solid', 'dashed', 'dotted']

# Set the color palette for the plots
all_colors = sns.color_palette('Set1')
unwanted_color = (1.0, 1.0, 0.2)  # Approximate RGB value for yellow in 'Set1'
colors = [color for color in all_colors if color != unwanted_color]

# Define markers for different lines
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', '+']

legends = set()

# Loop through all sheets and create a subplot for each dataset
for idx, sheet_name in enumerate(sheet_names):
    df = pd.read_excel(excel_data, sheet_name=sheet_name)
    if gp: df = df[::2]

    # Extract feature selectors and feature counts
    feature_selectors = df['Feature Selector']
    steps = len(df.columns[1:])
    feature_counts = values = np.arange(5, 5 + 5 * steps, 5)
    
    ax = axes[idx]
    if idx < 4:
        axes[idx].set_facecolor("#F0F0F0")  # Light yellow background for classification
    # Plot each feature selector's performance
    for i, selector in enumerate(feature_selectors):
        performance = df.iloc[i, 1:].values  # All performance values for the method
        
        # Choose a color and marker for each method
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # Plot with anchor (marker) and color
        ax.plot(feature_counts, performance, label=selector if selector not in legends else "", color=color, marker=marker, markersize=12, linewidth=4, linestyle=line_styles[i % len(line_styles)], alpha=0.8)

        if selector not in legends:
            legends.add(selector)

    
    # Customize the plot
    ax.set_title(f"{sheet_name.replace('_', ' ')} Dataset", fontsize=12)
    #ax.set_xlabel('Number of Features', fontsize=12)
    # if idx%col_no == 0:
    #     if is_classification:
    #         ax.set_ylabel(r"Performance $\uparrow$", fontsize=14)
    #     else:
    #         ax.set_ylabel(r"Performance $\downarrow$", fontsize=14)
    #ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticks(feature_counts)
    ax.set_xticklabels(feature_counts, rotation=0)
    ax.set_yticks([])
    #ax.legend(title="Feature Selectors", loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

fig.legend(loc='center', bbox_to_anchor=(0.5, 1.05), ncol=10)
fig.text(0.5, -0.03, 'Percentage of Added Features', ha='center', va='center', fontsize=14)

# Adjust layout for all subplots
plt.tight_layout()
fig.subplots_adjust(hspace=0.4, wspace=0.01)
plt.show()

fig.savefig(f"results/fs_real/fs_real{'_classification' if is_classification else '_regression'}{'_gp' if gp else '_rf'}.png", dpi=500, format='png', bbox_inches='tight')
print("done!")
