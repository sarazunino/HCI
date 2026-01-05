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
def plot_production_volume(df, start_year=None, end_year=2025):
    if start_year is None:
        start_year = int(df['year'].min())
    if end_year is None:
        end_year = int(df['year'].max())

    mask = (df['year'] >= start_year) & (df['year'] <= end_year)
    yearly_counts = df[mask].groupby('year')['mal_id'].count().reset_index()
    yearly_counts.columns = ['year', 'count']

    plt.figure(figsize=(15, 6))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(data=yearly_counts, x='year', y='count', color='steelblue', edgecolor='black', alpha=0.8)
    
    plt.title(f'Annual Anime Production Volume ({start_year} - {end_year})', fontsize=16, pad=20)
    plt.xlabel('Release Year', fontsize=12)
    plt.ylabel('Number of Titles Produced', fontsize=12, fontweight='bold')
    
    for i, label in enumerate(ax.get_xticklabels()):
        if i % 5 != 0:
            label.set_visible(False)

    plt.tight_layout()
    plt.show()


#  Analizza il volume di produzione per fonte
def plot_production_evolution_by_source(df, start_year=None, end_year=2025):
   df_clean = df[(df['source'] != 'Unknown') & (df['source'].notna())].copy() 
   top_sources = df_clean[df_clean['year'].notna()]['source'].value_counts().nlargest(5).index.tolist()
   if start_year is None: start_year = int(df_clean['year'].min()) 
   mask = (df_clean['year'] >= start_year) & (df_clean['year'] <= end_year) & (df_clean['source'].isin(top_sources)) 
   df_filtered = df_clean[mask] 
   source_time = df_filtered.groupby(['year', 'source']).size().unstack(fill_value=0) 
   source_time = source_time.reindex(columns=top_sources) 
   plt.figure(figsize=(15, 8)) 
   sns.set_style("whitegrid") 
   source_time.plot(kind='line', ax=plt.gca(), marker='o', markersize=4, linewidth=2.5)
   plt.title(f'Annual Production Volume by Source ({start_year} - {end_year})', fontsize=16, pad=20) 
   plt.xlabel('Release Year', fontsize=12) 
   plt.ylabel('Number of Titles Produced', fontsize=12, fontweight='bold') 
   plt.legend(title='Source', loc='upper left', frameon=True) 
   plt.tight_layout() 
   plt.show()


#  Mostra la popolarità totale delle fonti di produzione (Source)
def plot_source_popularity(df):
    df_clean = df[(df['source'] != 'Unknown') & (df['source'].notna())].copy()
    source_counts = df_clean['source'].value_counts()
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    sns.barplot(
        x=source_counts.values, 
        y=source_counts.index, 
        hue=source_counts.index, 
        palette='viridis', 
        legend=False
    )
    plt.title('Total Popularity by Source', fontsize=16, pad=20)
    plt.xlabel('Total Number of Titles', fontsize=12, fontweight='bold')
    plt.ylabel('Source', fontsize=12)
    for i, v in enumerate(source_counts.values):
        plt.text(v + 100, i, str(v), color='black', va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()


# Mostra la popolarità dei generi nel tempo.
def plot_genre_production_by_year(df, start_year=None, end_year=2025, top_n=5):
    df_clean = df.copy()
    df_clean = df_clean[df_clean['genres'].notna()]
    df_clean['genres'] = (
        df_clean['genres']
        .str.strip('[]')
        .str.split(',')
        .apply(lambda x: [g.strip() for g in x] if isinstance(x, list) else x)
    )
    df_exploded = df_clean.explode('genres')
    if start_year is None:
        start_year = int(df_exploded['year'].min())
    mask = (df_exploded['year'] >= start_year) & (df_exploded['year'] <= end_year)
    df_filtered = df_exploded[mask]
    top_genres = (
        df_filtered['genres']
        .value_counts()
        .nlargest(top_n)
        .index
    )
    df_filtered = df_filtered[df_filtered['genres'].isin(top_genres)]
    genre_year = (
        df_filtered
        .groupby(['year', 'genres'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=top_genres)
    )
    plt.figure(figsize=(16, 8))
    sns.set_style("whitegrid")
    genre_year.plot(
        kind='bar',
        stacked=True,
        ax=plt.gca()
    )
    plt.title(
        f'Anime Production by Genre per Year ({start_year}–{end_year})',
        fontsize=16,
        pad=20
    )
    plt.xlabel('Release Year', fontsize=12)
    plt.ylabel('Number of Titles', fontsize=12, fontweight='bold')
    plt.legend(title='Genre', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()