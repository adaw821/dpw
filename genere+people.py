import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
from scipy import stats


# 1. Loading Data
#load data
movies_df = pd.read_parquet('data/processed/movies_main.parquet')
genres_df = pd.read_parquet('data/processed/movie_genres.parquet')
cast_df = pd.read_csv('data/processed/cast.csv')
crew_df = pd.read_csv('data/processed/crew.csv')

# Check whether the data is loaded correctly
print("=== Statistics of Movies Associated with Actors ===")
actor_movie_counts = cast_df[cast_df['cast_order'] <= 3].groupby('person_id')['movie_id'].count()
print("Average number of films an actor participates in:", actor_movie_counts.mean())
print("Maximum number of films an actor has participated in:", actor_movie_counts.max())

# Check if any actors are associated with hundreds of movies (anomalies)
outliers = actor_movie_counts[actor_movie_counts > 50]
print(f"Number of actors who have participated in more than 50 films: {len(outliers)}")
if len(outliers) > 0:
    print("Example person_id:", outliers.head(3).index.tolist())

print("Data loading completed successfully.")


# 2. Cleaning movies_df: Calculate ROI
# Convert to numbers and handle non-numerical values
movies_df['budget'] = pd.to_numeric(movies_df['budget'], errors='coerce')
movies_df['revenue'] = pd.to_numeric(movies_df['revenue'], errors='coerce')

print("\n=== Pre-Cleaning Data Diagnosis ===")
print("budget min/max:", movies_df['budget'].min(), "/", movies_df['budget'].max())
print("revenue min/max:", movies_df['revenue'].min(), "/", movies_df['revenue'].max())
print("budget <= 0 count:", (movies_df['budget'] <= 0).sum())
print("revenue < 0 count:", (movies_df['revenue'] < 0).sum())

# Keep only the records where budget is greater than 0 and revenue is greater than or equal to 0.
movies_df = movies_df.dropna(subset=['budget', 'revenue'])
movies_df = movies_df[(movies_df['budget'] > 0) & (movies_df['revenue'] >= 0)]

# Calculate ROI
movies_df['roi'] = movies_df['revenue'] / movies_df['budget']

# Filter extreme values (ROI > 100 usually indicates data errors)
movies_df = movies_df[movies_df['roi'] <= 100]

print(f"\nCleaned valid movie count: {len(movies_df)}")
print("Cleaned roi range:", movies_df['roi'].min(), "to", movies_df['roi'].max())
assert movies_df['roi'].min() >= 0, "ROI still contains negative values!"
assert not np.isinf(movies_df['roi']).any(), "ROI contains infinite values!"


# 3. Top 20 Genres by Average ROI
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
plt.savefig('top_20_genres_by_roi.png', dpi=300, bbox_inches='tight')
plt.show()


# 4.Top 10 Directors
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

# Calculate the industry average ROI as the reference line
industry_avg_roi = movies_df['roi'].mean()

plt.figure(figsize=(12, 8))
sns.violinplot(data=plot_data_dir, y='label', x='roi', orient='h', width=0.8, 
               palette='viridis', inner='box',
               linewidth=1.5, 
               saturation=0.9) 
plt.xlabel('ROI (Revenue / Budget)')
plt.title('ROI Distribution for Top 10 Directors\n(by Average ROI, ≥3 films)')
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Add industry average ROI reference line
plt.axvline(x=industry_avg_roi, color='red', linestyle='--', alpha=0.7, label=f'Industry Avg ROI: {industry_avg_roi:.2f}')
plt.legend()


for i, row in enumerate(top_directors.itertuples()):
    sample_size = df_director[df_director['person_id'] == row.person_id]['roi'].count()
    text_combined = f'{row.avg_roi:.2f} (n={sample_size})'
    plt.text(row.avg_roi + 0.05, i + 0.2, text_combined, ha='left', va='center', 
             fontsize=10, fontweight='bold', color='white')


plt.xlim(left=0, right=160)
plt.tight_layout()
plt.savefig('top_10_directors_violin_plot.png', dpi=300, bbox_inches='tight')
plt.show()


# 5.Top 10 Actors
# Filtering failed person_name
cast_clean = cast_df[
    cast_df['person_name'].notna() &
    (cast_df['person_name'] != '') &
    (~cast_df['person_name'].str.contains(r'unknown|unnamed|uncredited', case=False))
].copy()

# 2. Count the number of valid movies for each person_id (only for cast_order ≤ 3)
star_counts = cast_clean[cast_clean['cast_order'] <= 3].groupby('person_id').size()
abnormal_ids = star_counts[star_counts > 50].index.tolist()
print(f"Detected {len(abnormal_ids)} abnormal person_ids (associated with more than 50 movies), they will be excluded")
print("Example abnormal IDs:", abnormal_ids[:5])

# 3. excluded ID
cast_clean = cast_clean[~cast_clean['person_id'].isin(abnormal_ids)]

# 4. Only retain the records where the lead roles (with cast_order less than or equal to 3) exist and the movie_id is present in the movies_df table.
valid_movies = set(movies_df['movie_id'])
cast_lead = cast_clean[
    (cast_clean['cast_order'] <= 3) &
    (cast_clean['movie_id'].isin(valid_movies))
][['movie_id', 'person_id', 'person_name']].drop_duplicates()

print(f"Number of valid lead actors after cleaning: {len(cast_lead)}")
print("Average number of movies participated by lead actors:", cast_lead.groupby('person_id').size().mean())
print("Maximum number of movies participated by lead actors:", cast_lead.groupby('person_id').size().max())


# merge ROI
df_star = pd.merge(
    movies_df[['movie_id', 'budget', 'revenue', 'roi']], 
    cast_lead, 
    on='movie_id',
    how='inner'
)

df_star = df_star[
    (df_star['budget'] > 0) & 
    (df_star['revenue'] >= 0) &
    (df_star['roi'] > 0) &
    (df_star['roi'] <= 100)
]

print("\n=== The actor ROI data used for plotting ===")
print("Total record count:", len(df_star))
print("ROI range:", df_star['roi'].min(), "to", df_star['roi'].max())
print("Number of records with ROI < 0:", (df_star['roi'] < 0).sum())
assert df_star['roi'].min() > 0, "There are still negative ROI!"


# top 10 actors（基于清洗后数据）
star_stats = df_star.groupby(['person_id', 'person_name']).agg(
    avg_roi=('roi', 'mean'),
    count=('roi', 'count')
).reset_index()

# Keep only the actors whose count is greater than or equal to 3.
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
               palette='plasma', inner='box',  
               linewidth=2, 
               saturation=0.9)  
plt.xlabel('ROI (Revenue / Budget)')
plt.title('ROI Distribution for Top 10 Actors\n(by Average ROI, ≥3 films, lead roles)\n[Cleaned: removed placeholder IDs]')
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Add industry average ROI reference line
plt.axvline(x=industry_avg_roi, color='red', linestyle='--', alpha=0.7, label=f'Industry Avg ROI: {industry_avg_roi:.2f}')
plt.legend()


for i, row in enumerate(top_stars.itertuples()):
    sample_size = df_star[df_star['person_id'] == row.person_id]['roi'].count()
    

    text_combined = f'{row.avg_roi:.2f} (n={sample_size})'
    plt.text(row.avg_roi + 0.05, i + 0.2, text_combined, 
             ha='left', va='center', fontsize=9, fontweight='bold', color='white')  


plt.xlim(left=0, right=200) 
plt.tight_layout()
plt.savefig('top_10_actors_violin_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAll the charts have been generated!")
print("\nThe picture has been saved as the following file:")
print("   • top_20_genres_by_roi.png")
print("   • top_10_directors_violin_plot.png")
print("   • top_10_actors_violin_plot.png")