import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats

# 0. Setup Output Folder
output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)
print(f"=== Setup ===")
print(f"Plots will be saved in the '{output_folder}' folder.")

# 1. Loading Data
# Using relative paths for better project portability
print("\n=== Loading Data ===")
movies_df = pd.read_parquet('data/processed/movies_main.parquet')
genres_df = pd.read_parquet('data/processed/movie_genres.parquet')
cast_df = pd.read_csv('data/processed/cast.csv')
crew_df = pd.read_csv('data/processed/crew.csv')

# Check whether the data is loaded correctly (Actors anomaly check)
actor_movie_counts = cast_df[cast_df['cast_order'] <= 3].groupby('person_id')['movie_id'].count()
print("Average number of films an actor participates in:", actor_movie_counts.mean())
print("Maximum number of films an actor has participated in:", actor_movie_counts.max())

outliers = actor_movie_counts[actor_movie_counts > 50]
if len(outliers) > 0:
    print(f"Number of actors who have participated in more than 50 films: {len(outliers)}")
    print("Example person_id:", outliers.head(3).index.tolist())

print("Data loading completed successfully.")


# 2. Unified Data Cleaning & Feature Engineering
print("\n=== Data Cleaning & Preprocessing ===")
# Convert to numbers and handle non-numerical values
movies_df['budget'] = pd.to_numeric(movies_df['budget'], errors='coerce')
movies_df['revenue'] = pd.to_numeric(movies_df['revenue'], errors='coerce')

# Keep only records where budget > 0 and revenue >= 0 (or notna)
movies_df = movies_df.dropna(subset=['budget', 'revenue'])
movies_df = movies_df[(movies_df['budget'] > 0) & (movies_df['revenue'] >= 0)]

# Calculate ROI and filter extreme anomalies
movies_df['roi'] = movies_df['revenue'] / movies_df['budget']
movies_df = movies_df[movies_df['roi'] <= 100]

# Add profitability and runtime features
movies_df['is_profitable'] = (movies_df['roi'] >= 2.5).astype(int)
movies_df['runtime_bin'] = pd.cut(movies_df['runtime'], 
                                  bins=[0, 90, 120, 500], 
                                  labels=['Short', 'Standard', 'Long'])

print(f"Cleaned valid movie count: {len(movies_df)}")
print("Cleaned ROI range:", movies_df['roi'].min(), "to", movies_df['roi'].max())
assert movies_df['roi'].min() >= 0, "ROI still contains negative values!"
assert not np.isinf(movies_df['roi']).any(), "ROI contains infinite values!"

# Calculate the industry average ROI as a global reference line
industry_avg_roi = movies_df['roi'].mean()


# 3. Visualizations Part A: Top Genres, Directors, and Actors (ROI Focus)
print("\n=== Generating ROI Focus Plots ===")

# --- A1. Top 20 Genres by Average ROI ---
genre_roi = pd.merge(genres_df, movies_df[['movie_id', 'roi']], on='movie_id')
genre_stats = genre_roi.groupby('genre_name').agg(
    avg_roi=('roi', 'mean'),
    movie_count=('roi', 'count')
).reset_index()

genre_stats = genre_stats[genre_stats['movie_count'] >= 5]
genre_stats = genre_stats.sort_values('avg_roi', ascending=False).head(20)

plt.figure(figsize=(10, 6))
n_bars = len(genre_stats)
colors = cm.viridis(np.linspace(0, 1, n_bars))
bars = plt.barh(genre_stats['genre_name'], genre_stats['avg_roi'], color=colors)
plt.xlabel('Average ROI')
plt.title('Top 20 Genres by Average ROI (min. 5 films)')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.6)
for i, val in enumerate(genre_stats['avg_roi']):
    plt.text(val + 0.05, i, f'{val:.2f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'top_20_genres_by_roi.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: top_20_genres_by_roi.png")


# --- A2. Top 10 Directors ---
directors = crew_df[crew_df['job_title'] == 'Director'][['movie_id', 'person_id', 'person_name']].drop_duplicates()
df_director = pd.merge(movies_df[['movie_id', 'roi']], directors, on='movie_id')

director_stats = df_director.groupby(['person_id', 'person_name']).agg(
    avg_roi=('roi', 'mean'),
    count=('roi', 'count')
).reset_index()

top_directors = director_stats[director_stats['count'] >= 3].nlargest(10, 'avg_roi').copy()
top_directors['label'] = top_directors['person_name'].str[:20] + '...'
top_directors = top_directors.sort_values('avg_roi', ascending=False)

plot_data_dir = df_director[df_director['person_id'].isin(top_directors['person_id'])].copy()
plot_data_dir['label'] = plot_data_dir['person_id'].map(top_directors.set_index('person_id')['label'])
plot_data_dir['label'] = pd.Categorical(plot_data_dir['label'], 
                                         categories=top_directors['label'][::-1], 
                                         ordered=True)

plt.figure(figsize=(12, 8))
sns.violinplot(data=plot_data_dir, y='label', x='roi', orient='h', width=0.8, 
               palette='viridis', inner='box', linewidth=1.5, saturation=0.9) 
plt.xlabel('ROI (Revenue / Budget)')
plt.title('ROI Distribution for Top 10 Directors\n(by Average ROI, ≥3 films)')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.axvline(x=industry_avg_roi, color='red', linestyle='--', alpha=0.7, label=f'Industry Avg ROI: {industry_avg_roi:.2f}')
plt.legend()

for i, row in enumerate(top_directors.itertuples()):
    sample_size = df_director[df_director['person_id'] == row.person_id]['roi'].count()
    plt.text(row.avg_roi + 0.05, i + 0.2, f'{row.avg_roi:.2f} (n={sample_size})', ha='left', va='center', 
             fontsize=10, fontweight='bold', color='white')

plt.xlim(left=0, right=160)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'top_10_directors_violin_plot.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: top_10_directors_violin_plot.png")


# --- A3. Top 10 Actors ---
cast_clean = cast_df[
    cast_df['person_name'].notna() &
    (cast_df['person_name'] != '') &
    (~cast_df['person_name'].str.contains(r'unknown|unnamed|uncredited', case=False))
].copy()

star_counts = cast_clean[cast_clean['cast_order'] <= 3].groupby('person_id').size()
abnormal_ids = star_counts[star_counts > 50].index.tolist()
cast_clean = cast_clean[~cast_clean['person_id'].isin(abnormal_ids)]

valid_movies = set(movies_df['movie_id'])
cast_lead = cast_clean[
    (cast_clean['cast_order'] <= 3) &
    (cast_clean['movie_id'].isin(valid_movies))
][['movie_id', 'person_id', 'person_name']].drop_duplicates()

df_star = pd.merge(
    movies_df[['movie_id', 'budget', 'revenue', 'roi']], 
    cast_lead, 
    on='movie_id',
    how='inner'
)
df_star = df_star[(df_star['roi'] > 0)]

star_stats = df_star.groupby(['person_id', 'person_name']).agg(
    avg_roi=('roi', 'mean'),
    count=('roi', 'count')
).reset_index()

top_stars = star_stats[star_stats['count'] >= 3].nlargest(10, 'avg_roi').copy()
top_stars['label'] = top_stars['person_name'].str[:20] + '...'
top_stars = top_stars.sort_values('avg_roi', ascending=False)

plot_data_act = df_star[df_star['person_id'].isin(top_stars['person_id'])].copy()
plot_data_act['label'] = plot_data_act['person_id'].map(top_stars.set_index('person_id')['label'])
plot_data_act['label'] = pd.Categorical(plot_data_act['label'], 
                                         categories=top_stars['label'][::-1], 
                                         ordered=True)

plt.figure(figsize=(12, 8))
sns.violinplot(data=plot_data_act, y='label', x='roi', orient='h', width=0.8, 
               palette='plasma', inner='box', linewidth=2, saturation=0.9)  
plt.xlabel('ROI (Revenue / Budget)')
plt.title('ROI Distribution for Top 10 Actors\n(by Average ROI, ≥3 films, lead roles)\n[Cleaned: removed placeholder IDs]')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.axvline(x=industry_avg_roi, color='red', linestyle='--', alpha=0.7, label=f'Industry Avg ROI: {industry_avg_roi:.2f}')
plt.legend()

for i, row in enumerate(top_stars.itertuples()):
    sample_size = df_star[df_star['person_id'] == row.person_id]['roi'].count()
    plt.text(row.avg_roi + 0.05, i + 0.2, f'{row.avg_roi:.2f} (n={sample_size})', 
             ha='left', va='center', fontsize=9, fontweight='bold', color='white')  

plt.xlim(left=0, right=200) 
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'top_10_actors_violin_plot.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: top_10_actors_violin_plot.png")


# 4. Visualizations Part B: Profitability Probability Plots
print("\n=== Generating Profitability Probability Plots ===")
sns.set_theme(style="whitegrid")

# --- B1. Profitability by Runtime ---
plt.figure(figsize=(8, 5))
grouped_prob = movies_df.groupby('runtime_bin', observed=False)['is_profitable'].mean().reset_index()
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
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "runtime.png"), dpi=300)
plt.close()
print("Saved: runtime.png")

# --- B2. Profitability by Assorted Counts (Genre, Country, Company) ---
def plot_profit_probability(column_name, title):
    # Only run if the column exists in movies_df to prevent errors
    if column_name not in movies_df.columns:
        print(f"Skipped: '{column_name}' not found in DataFrame.")
        return

    plt.figure(figsize=(10, 5))
    grouped_prob = movies_df.groupby(column_name)['is_profitable'].mean().reset_index()
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
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    filename = title.replace(' ', '_') + ".png"
    plt.savefig(os.path.join(output_folder, filename), dpi=300)
    plt.close()
    print(f"Saved: {filename}")

plot_profit_probability('genre_count', 'Profitability Probability by Genre Count')
plot_profit_probability('country_count', 'Profitability Probability by Country Count')
plot_profit_probability('company_count', 'Profitability Probability by Company Count')

print("\n=== All charts have been generated and saved successfully! ===")