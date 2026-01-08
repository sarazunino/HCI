import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- FUNZIONI DI DIAGNOSI ---

# Mostra le dimensioni (righe, colonne) del DataFrame
def show_shape(df, name="DataFrame"):
    print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")


# Analizza e stampa la presenza di record duplicati
def check_duplicates(df, name="DataFrame"):
    num_duplicates = df.duplicated().sum()
    if num_duplicates == 0:
        print(f"In '{name}': No duplicates found.")
    else:
        total_duplicated_rows = df.duplicated(keep=False).sum()
        unique_records_with_dupes = total_duplicated_rows - num_duplicates
        print(f"In '{name}': {unique_records_with_dupes} original records produced {num_duplicates} extra copies.")


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
def clean_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)


# Rimuove le righe che contengono solo valori NaN
def drop_empty_rows(df, name="DataFrame"):
    initial_count = len(df)
    df_cleaned = df.dropna(how='all')
    removed = initial_count - len(df_cleaned)
    if removed > 0:
        print(f"In {name}: removed {removed} empty rows")
    return df_cleaned
    

# Converte una colonna in datetime gestendo formati misti (es. solo anno o data completa)
def convert_to_datetime(df, column_name):
    if column_name in df.columns:
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce', format='%Y').fillna(
            pd.to_datetime(df[column_name], errors='coerce')
        )
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
def plot_popularity(df, start_year=1960, end_year=2025):
    df_temp = df.copy()
    df_temp['year_filled'] = df_temp['year'].fillna(df_temp['start_date'].dt.year)
    df_temp = df_temp[(df_temp['year_filled'] >= start_year) & 
                      (df_temp['year_filled'] <= end_year) & 
                      df_temp['members'].notna()]

    exclude_types = ['Music', 'CM', 'PV']
    df_temp = df_temp[(df_temp['type'].notna()) & 
                      (~df_temp['type'].isin(exclude_types)) & 
                      (df_temp['scored_by'] > 1500) & 
                      (df_temp['members'] > 1500)]

    summary = df_temp.groupby('year_filled')['members'].mean().reset_index()
    summary.columns = ['year', 'avg_members']

    all_years = pd.DataFrame({'year': range(start_year, end_year + 1)})
    summary = pd.merge(all_years, summary, on='year', how='left').fillna(0)

    plt.figure(figsize=(15,6))
    sns.set_style("whitegrid")
    sns.lineplot(data=summary, x='year', y='avg_members', color='seagreen', linewidth=2)

    max_row = summary.loc[summary['avg_members'].idxmax()]
    plt.scatter(max_row['year'], max_row['avg_members'], color='darkslategray', s=50, zorder=5)
    plt.annotate(f"Peak: {int(max_row['year'])}", 
                 xy=(max_row['year'], max_row['avg_members']), 
                 xytext=(0, 7), textcoords='offset points', 
                 ha='center', color='black')

    plt.fill_between(summary['year'], summary['avg_members'], color='seagreen', alpha=0.2)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xticks(ticks=range(start_year, end_year + 1, 5))
    plt.title(f'Popularity Over Time ({start_year} - {end_year})', fontsize=16, pad=20)
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Members', fontsize=12, fontweight='bold')
    plt.grid(visible=True, which='major', axis='both', linestyle=':', alpha=0.5)
    plt.xlim(start_year, end_year + 2)
    plt.tight_layout()
    plt.show()


# Analizza la valutazione media nel tempo
def plot_score(df, start_year=1960, end_year=2025):
    df_temp = df.copy()
    df_temp['year_filled'] = df_temp['year'].fillna(df_temp['start_date'].dt.year)
    df_temp = df_temp[(df_temp['year_filled'] >= start_year) & (df_temp['year_filled'] <= end_year) & (df_temp['score'].notna())]
    exclude_types = ['Music', 'CM', 'PV']
    df_temp = df_temp[(df_temp['type'].notna()) &
                    (~df_temp['type'].isin(exclude_types)) &
                    (df_temp['scored_by'] > 1500) &
                    (df_temp['members'] > 1500)]

    summary = df_temp.groupby('year_filled')['score'].mean().reset_index()
    summary.columns = ['year', 'avg_score']
    
    all_years = pd.DataFrame({'year': range(start_year, end_year + 1)})
    summary = pd.merge(all_years, summary, on='year', how='left').fillna(0)

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
    plt.ylim(summary['avg_score'].min() - 0.5, 10) 
    plt.title(f'Score Over Time ({start_year} - {end_year})', fontsize=16, pad=20)
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.grid(visible=True, which='major', axis='both', linestyle=':', alpha=0.5)
    
    plt.xlim(start_year, end_year + 2)
    plt.tight_layout()
    plt.show()



# Confronta Popularity e Score
def plot_popularity_vs_score(df, start_year=1960, end_year=2025):
    df_temp = df.copy()
    df_temp['year_filled'] = df_temp['year'].fillna(df_temp['start_date'].dt.year)
    df_temp = df_temp[(df_temp['year_filled'] >= start_year) & 
                      (df_temp['year_filled'] <= end_year) & 
                      (df_temp['scored_by'] > 2000) &
                      (df_temp['members'] > 2000)]
    
    exclude_types = ['Music', 'CM', 'PV']
    df_temp = df_temp[(df_temp['type'].notna()) & ~df_temp['type'].isin(exclude_types)]

    summary = df_temp.groupby('year_filled').agg({
        'members': 'mean',
        'score': 'mean'
    }).reset_index()

    all_years = pd.DataFrame({'year_filled': range(start_year, end_year + 1)})
    summary = pd.merge(all_years, summary, on='year_filled', how='left').fillna(0)
    
    fig, ax1 = plt.subplots(figsize=(15, 7))
    sns.set_style("whitegrid")

    # Popularity
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Members', fontsize=12, fontweight='bold', color='seagreen')
    sns.lineplot(data=summary, x='year_filled', y='members', color='seagreen', linewidth=2.5, ax=ax1)
    ax1.ticklabel_format(style='plain', axis='y')

    # Score
    ax2 = ax1.twinx()
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold', color='mediumpurple')
    sns.lineplot(data=summary, x='year_filled', y='score', color='mediumpurple', linewidth=2.5 , ax=ax2)
    
    pop_max = summary['members'].max()
    ax1.set_ylim(bottom=-pop_max * 0.05, top=pop_max * 1.05)
    ax2.set_ylim(bottom=-0.5, top=10.5)

    plt.xlim(left=start_year, right=end_year + 2)
    ax1.set_xticks(range(start_year, end_year + 1, 5))

    plt.title(f'Popularity vs Score Over Time ({start_year} - {end_year})', fontsize=18, pad=25)
    
    ax1.grid(visible=True, which='major', axis='both', linestyle=':', alpha=0.5)
    ax2.grid(False) 
    fig.tight_layout()
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


# Analizza il numero di anime medi usciti ogni stagione
def plot_season(df):
    df_temp = df.copy()
    df_temp = df_temp[df_temp['season'].notna()]
    exclude_types = ['Music', 'CM', 'PV']
    df_temp = df_temp[(df_temp['type'].notna()) & ~df_temp['type'].isin(exclude_types)]
    season_order = ['winter', 'spring', 'summer', 'fall']
    seasonal_counts = df_temp['season'].value_counts().reindex(season_order).reset_index()
    seasonal_counts.columns = ['season', 'count']

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid") 
    colors = ['#A188BE', '#91C483', '#FFD384', '#E68A8A'] 

    sns.barplot(data=seasonal_counts, x='season', y='count',  hue='season', palette=colors, legend=False)
    plt.grid(visible=True, axis='y', linestyle=':', alpha=0.5)
    plt.title('Releases by Season', fontsize=16, pad=20)
    plt.xlabel('Season', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Titles', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()