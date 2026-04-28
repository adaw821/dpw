import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#create a folder to save plots
output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)
print(f"plots are starting saving in folder {output_folder}")

df = pd.read_parquet(r'C:\Users\imada\Desktop\dpw\data\processed\movies_main.parquet')

df = df[(df['budget'] > 0) & (df['revenue'].notna())].copy()

# ROI and Label of profitable
df['ROI'] = df['revenue'] / df['budget']
df['is_profitable'] = (df['ROI'] >= 2.5).astype(int)
#cut
df['runtime_bin'] = pd.cut(df['runtime'], 
                           bins=[0, 90, 120, 500], 
                           labels=['Short', 'Standard', 'Long'])

sns.set_theme(style="whitegrid")


# plot 1_runtime
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))

grouped_prob = df.groupby('runtime_bin', observed=False)['is_profitable'].mean().reset_index()

ax = sns.barplot(data=grouped_prob, x='runtime_bin', y='is_profitable', palette='magma')

plt.title('Profitability Probability by Runtime Category', fontsize=14)
plt.xlabel('Runtime Bin (movies runtime level)', fontsize=12)
plt.ylabel('Average Profitability Probability', fontsize=12)
plt.ylim(0, 1)

for p in ax.patches:
    height = p.get_height()
    if pd.notna(height) and height > 0:
        ax.annotate(f'{height:.1%}', 
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()

path1 = os.path.join(output_folder, "runtime.png")
plt.savefig(path1, dpi=300)
print('runtime.png is saved')
plt.close


# plot 2, 3, 4
def plot_profit_probability(column_name, title):
    plt.figure(figsize=(10, 5))
    
    grouped_prob = df.groupby(column_name)['is_profitable'].mean().reset_index()
    
    grouped_prob = grouped_prob[grouped_prob[column_name] <= 8] 
    
    ax = sns.barplot(data=grouped_prob, x=column_name, y='is_profitable', color='skyblue')
    
    plt.title(title, fontsize=14)
    plt.xlabel(f'{column_name}', fontsize=12)
    plt.ylabel('Average Profitability Probability', fontsize=12)
    plt.ylim(0, 1)
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.1%}', 
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    
    filename = title + ".png"
    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path, dpi=300)
    print(f"successsfully saved: {filename}")
    plt.close()

plot_profit_probability('genre_count', 'Profitability Probability by Genre Count')
plot_profit_probability('country_count', 'Profitability Probability by Country Count')
plot_profit_probability('company_count', 'Profitability Probability by Company Count')