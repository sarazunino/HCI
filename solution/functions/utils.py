import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- FUNZIONI DI DIAGNOSI ---

# Mostra le dimensioni (righe, colonne) del DataFrame
def show_shape(df, name="DataFrame"):
    print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")

# Mostra i dtypes
def show_dtypes(df, name="DataFrame"):
    print(f"\nDATASET: {name}")
    print(df.dtypes)

# Analizza e stampa la presenza di record duplicati
def check_duplicates(df, name="DataFrame"):
    num_duplicates = df.duplicated().sum()
    if num_duplicates == 0:
        print(f"In '{name}': No duplicates found.")
        print()
    else:
        total_duplicated_rows = df.duplicated(keep=False).sum()
        unique_records_with_dupes = total_duplicated_rows - num_duplicates
        print(f"In '{name}': {unique_records_with_dupes} original records produced {num_duplicates} extra copies.")
        print()


# Calcola la percentuale di valori mancanti per ogni colonna
def check_missing(df, name="DataFrame"):
    null_count = df.isnull().sum()
    missing_cols = null_count[null_count > 0]
    if missing_cols.empty:
        print(f"In {name:<25} No missing values found.")
    else:
        for col, count in missing_cols.items():
            perc = (count / len(df)) * 100
            precision = ".5f" if perc < 0.01 else ".2f"
            print(f"In {name:<25} col: {col:<25} missing: {perc:{precision}}%")


# Verifica se una colonna ha valori univoci per essere usata come indice
def check_unique(df, col):
    if col in df.columns:
        if df[col].is_unique:
            print(f"'{col}' can be used as an index for the DataFrame.")
            return True
        else:
            print(f"'{col}' cannot be used as an index because it is not unique.")
            return False
    else:
        print(f"Column '{col}' does not exist in the DataFrame.")
        return False


# --- FUNZIONI DI TRASFORMAZIONE ---

# Rimuove i duplicati e resetta l'indice
def clean_duplicates(df, name="DataFrame"):
    old_len = len(df)
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    print(f"Dataset {name:<25}: new rows {len(df_cleaned)}, before was {old_len} rows, removed {old_len - len(df_cleaned)}")
    return df_cleaned


# Rimuove le righe che contengono solo valori NaN
def drop_empty_rows(df, name="DataFrame"):
    initial_count = len(df)
    df_cleaned = df.dropna(how='all')
    removed = initial_count - len(df_cleaned)
    if removed > 0:
        print(f"In {name}: removed {removed} empty rows")
    return df_cleaned

# Rimuove le righe che hanno key_column NaN
def drop_key_missing(df, key_column, df_name="DataFrame"):
    if key_column in df.columns:
        old_len = len(df)
        df_cleaned = df.dropna(subset=[key_column]).reset_index(drop=True)
        removed = old_len - len(df_cleaned)
        print(f"Dataset {df_name:<15}: removed {removed} rows with NaN in '{key_column}'")
        return df_cleaned
    else:
        print(f"Column '{key_column}' not found in {df_name}. No rows removed.")
        return df

# Converte una colonna in datetime gestendo formati misti (es. solo anno o data completa)
def convert_to_datetime(df, column_name):
    if column_name in df.columns:
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce', format='%Y').fillna(
            pd.to_datetime(df[column_name], errors='coerce')).infer_objects(copy=False)
        print(f"Column '{column_name}' successfully converted.")
    else:
        print(f"Column '{column_name}' not found in DataFrame.")
    return df

# --- FUNZIONI DI VISUALIZZAZIONE ---

# Analizza l'evoluzione temporale della produzione di anime  
def plot_yearly_production(df, start_year=1915, end_year=2025):
    df_temp = df.copy()
    df_temp['year_filled'] = df_temp['year'].fillna(df_temp['start_date'].dt.year)
    exclude_types = ['Music', 'CM', 'PV']
    df_temp = df_temp[(df_temp['type'].notna()) & (~df_temp['type'].isin(exclude_types))]
    
    yearly_counts = df_temp.groupby('year_filled')['mal_id'] \
                      .count() \
                      .reindex(range(start_year, end_year+1), fill_value=0) \
                      .reset_index()
    yearly_counts.columns = ['year', 'count']
    
    plt.figure(figsize=(15, 6))
    sns.set_style("whitegrid") 
    plt.fill_between(yearly_counts['year'], yearly_counts['count'], color="Purple", alpha=0.3)
    plt.plot(yearly_counts['year'], yearly_counts['count'], color="Purple", lw=2) 
    plt.title(f'Production Over Time ({start_year} - {end_year})', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Titles', fontsize=12, fontweight='bold')
    plt.xticks(ticks=range(start_year, end_year + 1, 5))
    plt.grid(visible=True, which='major', axis='both', linestyle=':', alpha=0.5) 
    plt.ylim(0, yearly_counts['count'].max() * 1.05)

    plt.xlim(start_year, end_year + 2)
    plt.tight_layout()
    plt.show()


#  Analizza il volume di produzione per fonte
def plot_production_by_source(df, start_year=1915, end_year=2025):
    df_temp = df.copy()
    df_temp['year_filled'] = df_temp['year'].fillna(df_temp['start_date'].dt.year)
    df_temp = df_temp[(df_temp['source'].notna()) & (df_temp['source'] != 'Unknown')]
    exclude_types = ['Music', 'CM', 'PV']
    df_temp = df_temp[(df_temp['type'].notna()) & (~df_temp['type'].isin(exclude_types))]

    top_sources = df_temp['source'].value_counts().nlargest(5).index.tolist()
    df_temp = df_temp[(df_temp['year_filled'] >= start_year) & (df_temp['year_filled'] <= end_year) & (df_temp['source'].isin(top_sources))]
    
    source_time = df_temp.groupby(['year_filled', 'source']).size().unstack(fill_value=0)
    source_time = source_time.reindex(columns=top_sources)
    plt.figure(figsize=(15,6))
    sns.set_style("whitegrid")
    source_time.plot(kind='line', ax=plt.gca(), linewidth=2.5)
    plt.grid(visible=True, which='major', axis='both', linestyle=':', alpha=0.5)
    plt.title(f'Production by Source Over Time ({start_year} - {end_year})', fontsize=16, pad=20)
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Titles', fontsize=12, fontweight='bold')
    plt.xticks(ticks=range(start_year, end_year + 1, 5))
    plt.legend(title='Source', loc='upper left', frameon=True)

    plt.xlim(start_year, end_year + 2)
    plt.tight_layout()
    plt.show()


# Analizza la popolarità degli anime nel tempo
def plot_popularity(df, start_year=1915, end_year=2025):
    df_temp = df.copy()
    df_temp['year_filled'] = df_temp['year'].fillna(df_temp['start_date'].dt.year)
    df_temp = df_temp[(df_temp['year_filled'] >= start_year) & 
                      (df_temp['year_filled'] <= end_year)]

    exclude_types = ['Music', 'CM', 'PV']
    df_temp = df_temp[(df_temp['type'].notna()) & 
                      (~df_temp['type'].isin(exclude_types))]

    summary = df_temp.groupby('year_filled')['members'].mean().reset_index()
    summary.columns = ['year', 'avg_members']
    
    plt.figure(figsize=(15,6))
    sns.set_style("whitegrid")
    sns.lineplot(data=summary, x='year', y='avg_members', color='seagreen', linewidth=2)
    plt.fill_between(summary['year'], summary['avg_members'], color='seagreen', alpha=0.2)
    plt.ticklabel_format(style='plain', axis='y')
    max_row = summary.loc[summary['avg_members'].idxmax()]
    plt.scatter(max_row['year'], max_row['avg_members'], color='darkslategray', s=50, zorder=5)
    plt.annotate(f"Peak: {int(max_row['year'])}", 
                 xy=(max_row['year'], max_row['avg_members']), 
                 xytext=(0, 5), textcoords='offset points', 
                 ha='center', color='black')
    plt.xticks(ticks=range(start_year, end_year + 1, 5))
    plt.title(f'Popularity Over Time ({start_year} - {end_year})', fontsize=16, pad=20)
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Popularity', fontsize=12, fontweight='bold')
    plt.grid(visible=True, which='major', axis='both', linestyle=':', alpha=0.5)
    plt.xlim(start_year, end_year + 2)
    plt.tight_layout()
    plt.show()

# Analizza la popolarità degli anime nel tempo filtrata
def plot_popularity_filtred(df, start_year=1960, end_year=2025):
    df_temp = df.copy()
    df_temp['year_filled'] = df_temp['year'].fillna(df_temp['start_date'].dt.year)
    df_temp = df_temp[(df_temp['year_filled'] >= start_year) & 
                      (df_temp['year_filled'] <= end_year) & 
                      (df_temp['members'] > 5000)]

    exclude_types = ['Music', 'CM', 'PV']
    df_temp = df_temp[(df_temp['type'].notna()) & 
                      (~df_temp['type'].isin(exclude_types))]

    summary = df_temp.groupby('year_filled')['members'].mean().reset_index()
    summary.columns = ['year', 'avg_members']

    plt.figure(figsize=(15,6))
    sns.set_style("whitegrid")
    max_row = summary.loc[summary['avg_members'].idxmax()]
    plt.scatter(max_row['year'], max_row['avg_members'], color='darkslategray', s=50, zorder=5)
    plt.annotate(f"Peak: {int(max_row['year'])}", 
                 xy=(max_row['year'], max_row['avg_members']), 
                 xytext=(0, 5), textcoords='offset points', 
                 ha='center', color='black')
    sns.lineplot(data=summary, x='year', y='avg_members', color='seagreen', linewidth=2)
    plt.fill_between(summary['year'], summary['avg_members'], color='seagreen', alpha=0.2)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xticks(ticks=range(start_year, end_year + 1, 5))
    plt.title(f'Popularity Over Time ({start_year} - {end_year})', fontsize=16, pad=20)
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Popularity', fontsize=12, fontweight='bold')
    plt.grid(visible=True, which='major', axis='both', linestyle=':', alpha=0.5)
    plt.xlim(start_year, end_year + 2)
    plt.tight_layout()
    plt.show()


# Analizza la valutazione media nel tempo
def plot_score(df, start_year=1915, end_year=2025):
    df_temp = df.copy()
    df_temp['year_filled'] = df_temp['year'].fillna(df_temp['start_date'].dt.year)
    df_temp = df_temp[(df_temp['year_filled'] >= start_year) & (df_temp['year_filled'] <= end_year) & (df_temp['score'].notna())]
    exclude_types = ['Music', 'CM', 'PV']
    df_temp = df_temp[(df_temp['type'].notna()) &
                    (~df_temp['type'].isin(exclude_types))]

    summary = df_temp.groupby('year_filled')['score'].mean().reset_index()
    summary.columns = ['year', 'avg_score']

    plt.figure(figsize=(15,6))
    sns.set_style("whitegrid")
    sns.lineplot(data=summary, x='year', y='avg_score', color='mediumpurple', linewidth=2.5)
    plt.fill_between(summary['year'], summary['avg_score'], color='mediumpurple', alpha=0.2)
    max_row = summary.loc[summary['avg_score'].idxmax()]
    plt.scatter(max_row['year'], max_row['avg_score'], color='rebeccapurple', s=50, zorder=5)
    
    plt.annotate(f"Peak: {max_row['avg_score']:.2f} ({int(max_row['year'])})", 
                 xy=(max_row['year'], max_row['avg_score']),
                 xytext=(0, 10), textcoords='offset points',
                 ha='center', color='black')

    plt.xticks(ticks=range(start_year, end_year + 1, 5))
    plt.ticklabel_format(style='plain', axis='y')
    plt.title(f'Score Over Time ({start_year} - {end_year})', fontsize=16, pad=20)
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.grid(visible=True, which='major', axis='both', linestyle=':', alpha=0.5)
    plt.ylim(5 - 0.5, 10)
    plt.xlim(start_year, end_year + 2)
    plt.tight_layout()
    plt.show()

# Analizza la valutazione media nel tempo filtratata
def plot_score_filtred(df, start_year=1960, end_year=2025):
    df_temp = df.copy()
    df_temp['year_filled'] = df_temp['year'].fillna(df_temp['start_date'].dt.year)
    df_temp = df_temp[(df_temp['year_filled'] >= start_year) & (df_temp['year_filled'] <= end_year) & (df_temp['score'].notna())]
    exclude_types = ['Music', 'CM', 'PV']
    df_temp = df_temp[(df_temp['type'].notna()) &
                    (~df_temp['type'].isin(exclude_types)) &
                    (df_temp['scored_by'] > 4000)]

    summary = df_temp.groupby('year_filled')['score'].mean().reset_index()
    summary.columns = ['year', 'avg_score']

    plt.figure(figsize=(15,6))
    sns.set_style("whitegrid")
    sns.lineplot(data=summary, x='year', y='avg_score', color='mediumpurple', linewidth=2.5)
    plt.fill_between(summary['year'], summary['avg_score'], color='mediumpurple', alpha=0.2)
    max_row = summary.loc[summary['avg_score'].idxmax()]
    plt.scatter(max_row['year'], max_row['avg_score'], color='rebeccapurple', s=50, zorder=5)
    
    plt.annotate(f"Peak: {max_row['avg_score']:.2f} ({int(max_row['year'])})", 
                 xy=(max_row['year'], max_row['avg_score']),
                 xytext=(0, 10), textcoords='offset points',
                 ha='center', color='black')

    plt.xticks(ticks=range(start_year, end_year + 1, 5))
    plt.ticklabel_format(style='plain', axis='y')
    plt.ylim(5 - 0.5, 10)
    plt.title(f'Score Over Time ({start_year} - {end_year})', fontsize=16, pad=20)
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.grid(visible=True, which='major', axis='both', linestyle=':', alpha=0.5)
    
    plt.xlim(start_year, end_year + 2)
    plt.tight_layout()
    plt.show()

# Analizza il numero di episodi medi per anno
def plot_episodes(df, start_year=1960, end_year=2025):
    df_temp = df.copy()
    df_temp['year_filled'] = df_temp['year'].fillna(df_temp['start_date'].dt.year)
    df_temp = df_temp[
        (df_temp['type'].notna()) & 
        (df_temp['type'].isin(['TV', 'ONA'])) &
        (df_temp['episodes'].notna()) &
        (df_temp['status'] == 'Finished Airing')
    ]

    summary = df_temp.groupby('year_filled')['episodes'].agg(['mean', 'count']).reset_index()
    summary.columns = ['year', 'avg_episodes', 'title_count']
    summary = summary.set_index('year').reindex(range(start_year, end_year + 1), fill_value=0).reset_index()

    plt.figure(figsize=(15, 6))
    sns.set_style("whitegrid")
    sns.barplot(data=summary, x='year', y='avg_episodes', color='cornflowerblue', alpha=0.7)

    for i, row in summary.iterrows():
        if row['title_count'] > 0 and row['title_count'] <= 10:
            plt.text(i, row['avg_episodes'], f"({int(row['title_count'])})", 
                     ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')
    
    years = summary['year'].unique()
    plt.xticks(ticks=[i for i, year in enumerate(years) if year % 5 == 0], 
               labels=[int(year) for year in years if year % 5 == 0])
    
    plt.title(f'Number of Episodes Over Time ({start_year} - {end_year})', fontsize=16, pad=20)
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Episodes', fontsize=12, fontweight='bold')
    plt.grid(visible=True, which='major', axis='both', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

# --- ANALISI 2 ---
#
def plot_genres(df):
    df_temp = df['genres'].apply(eval).explode()
    genre_counts = df_temp.value_counts().reset_index()
    genre_counts.columns = ['genre', 'count']
    plt.figure(figsize=(15,8))
    sns.barplot(data=genre_counts, x='count', y='genre', hue='genre', palette='viridis')
    plt.title('Genre Distribution', fontsize=16)
    plt.xlabel('Number of Anime', fontsize=12, fontweight='bold')
    plt.ylabel('Genres', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

#
def plot_demographics(df):
    df_temp = df['demographics'].apply(eval).explode()
    demographic_counts = df_temp.value_counts()
    colors = plt.cm.Pastel1(range(len(demographic_counts)))
    fig, ax = plt.subplots(figsize=(12, 8)) 
    wedges, texts, autotexts = ax.pie(
        demographic_counts,
        labels=demographic_counts.index,
        autopct='%1.1f%%',    
        startangle=140,
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
        pctdistance=0.85 
    )
    plt.setp(autotexts, size=10, weight="bold", color="black")
    plt.setp(texts, size=12)
    ax.set_title('Demographic Distribution', fontsize=15, pad=20)
    plt.axis('equal') 
    plt.tight_layout()
    plt.show()

#
def plot_male_demographics(df):
    df_temp = df[['genres', 'demographics']].copy()
    df_temp['demographics'] = df_temp['demographics'].apply(eval)
    df_temp = df_temp.explode('demographics')
    df_temp['genres'] = df_temp['genres'].apply(eval)
    df_temp = df_temp.explode('genres')
    target_order = ['Shounen', 'Seinen']
    df_temp = df_temp[df_temp['demographics'].isin(target_order)]
    data = df_temp.groupby(['demographics', 'genres']).size().reset_index(name='count')
    data = data.sort_values(['demographics', 'count'], ascending=[True, False]).groupby('demographics').head(3)
    g = sns.catplot(data=data, kind="bar", y="genres", x="count", col="demographics", col_order=target_order, hue="genres", 
        palette="tab10", sharey=False, sharex=False, height=5, aspect=1.4, dodge=False, edgecolor='white', linewidth=1.5)
    g.set_axis_labels("", "")
    g.set(xticks=[])
    g.set_titles("{col_name}", size=20, fontweight='bold', pad=20)
    g.despine(left=True, bottom=True)
    g.fig.suptitle('Dominant Genres in Male Demographics', fontsize=18, y=1.1)
    plt.show()

#
def plot_female_demographics(df):
    df_temp = df[['genres', 'demographics']].copy()
    df_temp['demographics'] = df_temp['demographics'].apply(eval)
    df_temp = df_temp.explode('demographics')
    df_temp['genres'] = df_temp['genres'].apply(eval)
    df_temp = df_temp.explode('genres')
    target_order = ['Shoujo', 'Josei']
    df_temp = df_temp[df_temp['demographics'].isin(target_order)]
    data = df_temp.groupby(['demographics', 'genres']).size().reset_index(name='count')
    data = data.sort_values(['demographics', 'count'], ascending=[True, False]).groupby('demographics').head(3)
    g = sns.catplot(data=data, kind="bar", y="genres", x="count", col="demographics", col_order=target_order, hue="genres", 
        palette="spring", sharey=False, sharex=False, height=5, aspect=1.4, dodge=False, edgecolor='white', linewidth=1.5)
    g.set_axis_labels("", "")
    g.set(xticks=[])
    g.set_titles("{col_name}", size=20, fontweight='bold', pad=20)
    g.despine(left=True, bottom=True)
    g.fig.suptitle('Dominant Genres in Female Demographics', fontsize=18, y=1.1)
    plt.show()

#
def plot_kids_demographics(df):
    df_temp = df[['genres', 'demographics']].copy()
    df_temp['demographics'] = df_temp['demographics'].apply(eval)
    df_temp = df_temp.explode('demographics')

    df_temp['genres'] = df_temp['genres'].apply(eval)
    df_temp = df_temp.explode('genres')

    target = ['Kids']
    df_temp = df_temp[df_temp['demographics'].isin(target)]
    data = df_temp.groupby(['demographics', 'genres']).size().reset_index(name='count')
    data = data.sort_values(['demographics', 'count'], ascending=[True, False]).groupby('demographics').head(3)
    g = sns.catplot(data=data, kind="bar", y="genres", x="count", col="demographics", hue="genres", 
        palette="cool", sharey=False, sharex=False, height=7, aspect=2.14, dodge=False, edgecolor='white', linewidth=1.5)
    g.set_axis_labels("", "")
    g.set(xticks=[])
    g.set_titles("{col_name}", size=20, fontweight='bold', pad=20)
    g.despine(left=True, bottom=True)
    g.fig.suptitle('Dominant Genres in Kids Demographics', fontsize=18, y=1.1)
    plt.show()

#
def plot_genres_score(df):
    df_temp = df[['genres', 'score']].copy()
    df_temp['genres'] = df_temp['genres'].apply(eval)
    df_temp = df_temp.explode('genres')
    data = df_temp.groupby('genres')['score'].mean().reset_index()
    data = data.sort_values('score', ascending=False)
    plt.figure(figsize=(15, 7)) 
    sns.barplot(data=data, x='genres', y='score', hue='genres', palette='crest', edgecolor='white', linewidth=0.8)
    plt.title('Average Score by Genre', fontsize=22, pad=30)
    plt.xlabel("")
    plt.ylabel("Average Score", fontsize=14, fontweight='bold')
    plt.ylim(0, 10) 
    plt.xticks(rotation=45, ha='right', fontsize=10)
    sns.despine(bottom=True)
    plt.grid(axis='y', linestyle='--', alpha=0.3) 
    plt.tight_layout()
    plt.show()