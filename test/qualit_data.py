import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the style for all plots
sns.set_palette("deep")

# Realism: How realistic does the restoration appears?
# Historical: How well does the restoration of the glyphs align with known epigraphical forms?
# Readability: How easily can the restored glyph be identified?

# Load the data
try:
    df = pd.read_csv('../data/qualitative_dataset/qualitative_dataset.csv')
except FileNotFoundError:
    print("Error: The data file was not found. Please check the file path.")
    exit()

# Function to save figures
def save_figure(fig, filename):
    fig.show()
    # fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

# 1. Distribution of Scores by Criteria and Image Type
fig, ax = plt.subplots(figsize=(14, 8))
sns.boxplot(
    x='image_type', y='value',
    hue='variable',
    data=pd.melt(
        df,
        id_vars=['expert_id', 'image_id', 'image_type'],
        value_vars=['realism', 'historical_accuracy', 'readability']
    ),
    ax=ax
)
ax.set_title('Distribution of Scores by Criteria and Image Type', fontsize=16)
ax.set_xlabel('Image Type', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.legend(
    title='Criteria',
    title_fontsize='12',
    fontsize='10'
)
save_figure(fig, 'score_distribution_boxplot.png')

# 2. Overall Comparison of Synthetic vs Real Images
fig, ax = plt.subplots(figsize=(14, 8))
df_mean = df.groupby('image_type')[['realism', 'historical_accuracy', 'readability']].mean()
df_std = df.groupby('image_type')[['realism', 'historical_accuracy', 'readability']].std()
df_mean.plot(kind='bar', yerr=df_std, capsize=5, ax=ax)
ax.set_title('Overall Comparison of Synthetic vs Real Images', fontsize=16)
ax.set_ylabel('Average Score', fontsize=12)
ax.set_ylim(0, 7)
ax.legend(title='Criteria', title_fontsize='12', fontsize='10')
ax.tick_params(axis='x', rotation=0)
save_figure(fig, 'overall_comparison.png')

# 3. Expert Comparison
fig, ax = plt.subplots(figsize=(14, 8))
sns.boxplot(x='expert_id', y='value', hue='image_type',
            data=pd.melt(df, id_vars=['expert_id', 'image_type'],
                         value_vars=['realism', 'historical_accuracy', 'readability']),
            ax=ax)
ax.set_title('Expert Comparison Across All Criteria', fontsize=16)
ax.set_xlabel('Expert ID', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.legend(title='Image Type', title_fontsize='12', fontsize='10')
save_figure(fig, 'expert_comparison.png')

# 4. Criteria Correlation
fig, ax = plt.subplots(figsize=(14, 8))
sns.scatterplot(data=df, x='realism', y='historical_accuracy', hue='image_type', style='expert_id', ax=ax)
ax.set_title('Correlation between Realism and Historical Accuracy', fontsize=16)
ax.set_xlabel('Realism Score', fontsize=12)
ax.set_ylabel('Historical Accuracy Score', fontsize=12)
ax.legend(title='Image Type', title_fontsize='12', fontsize='10', loc='center left', bbox_to_anchor=(1, 0.5))
save_figure(fig, 'criteria_correlation.png')

# 5. Score Distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
criteria = ['realism', 'historical_accuracy', 'readability']
for i, criterion in enumerate(criteria):
    sns.kdeplot(data=df, x=criterion, hue='image_type', shade=True, ax=axes[i])
    axes[i].set_title(f'{criterion.capitalize()} Score Distribution', fontsize=14)
    axes[i].set_xlabel('Score', fontsize=12)
    axes[i].set_ylabel('Density', fontsize=12)
fig.suptitle('Score Distributions by Criteria', fontsize=16)
fig.tight_layout()
save_figure(fig, 'score_distribution.png')

# 6. Heatmap of Average Scores
fig, ax = plt.subplots(figsize=(12, 10))
pivot_df = df.pivot_table(values=['realism', 'historical_accuracy', 'readability'],
                          index='expert_id', columns='image_type', aggfunc='mean')
sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', vmin=1, vmax=7, ax=ax, fmt='.2f')
ax.set_title('Heatmap of Average Scores by Expert and Image Type', fontsize=16)
save_figure(fig, 'score_heatmap.png')

print("All plots have been generated and saved.")