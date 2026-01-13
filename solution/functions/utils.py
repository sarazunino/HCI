import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import geopandas as gpd

# ----------------------- FUNZIONI DI PULIZIA  -----------------------

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
        total = df.duplicated(keep=False).sum()
        unique = total - num_duplicates
        print(f"In '{name}': {unique} original records produced {num_duplicates} extra copies.")
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

# Converte una colonna in datetime come visto a lezione
def convert_to_datetime(df, column_name):
    pd.set_option('future.no_silent_downcasting', True)
    df_temp = pd.to_datetime(df[column_name], errors='coerce', format='%Y')
    df_temp = df_temp.fillna(pd.to_datetime(df[column_name], errors='coerce'))
    df[column_name] = df_temp
    return df

# ----------------------- FUNZIONI PER LA PARTE DI ANALISI E VISUALIZZAZIONE -----------------------

# --- Evoluzione Temporale dell'Industria ---

# Prende l'anno di start_date quando year è NaN e esclude i tipi non narrativi
def get_cleaned_details(df):
    df_copy = df.copy()
    df_copy['year_filled'] = df_copy['year'].fillna(df_copy['start_date'].dt.year)
    df_copy = df_copy[df_copy['year_filled'] < 2026].copy()
    exclude_types = {'Music', 'CM', 'PV'}
    df_copy = df_copy[df_copy['type'].notna() & ~df_copy['type'].isin(exclude_types)]
    return df_copy

# --- Analisi Generi/Demografiche --- 

# Trasforma le colonne 'genres' e 'demographics' da stringhe di liste in righe singole.
def get_exploded(df):
    df_copy = df.copy()
    df_copy['genres'] = df_copy['genres'].apply(eval)
    df_copy = df_copy.explode('genres')
    df_copy['demographics'] = df_copy['demographics'].apply(eval)
    df_copy = df_copy.explode('demographics')
    return df_copy
    
# --- Indagine sugli Utenti ---

# Grafici dei generi più visti dagli utenti divisi per genere dichiarato
def plot_genre_by_gender(df_female, df_male, df_nb):
    # grafico female
    custom_oranges = sns.light_palette("#fc8d62", n_colors=12, reverse=True)[2:12]
    plt.figure(figsize=(15, 8))
    patches, texts, autotexts = plt.pie(df_female['total_views'], labels=df_female['genres'], autopct='%1.1f%%', startangle=140, 
        colors=custom_oranges, pctdistance=0.85, radius=0.8, wedgeprops={'edgecolor': "#a75d40", 'linewidth': 1.5})
    centre_circle = plt.Circle((0,0), 0.45, fc='white', edgecolor='#a75d40', linewidth=1)
    plt.gca().add_artist(centre_circle)
    plt.setp(autotexts, size=11, weight="bold")
    plt.title('Top 10 Genres - Female Users', fontweight='bold', fontsize=15, pad=20)
    plt.axis('equal')  
    plt.tight_layout()
    plt.show()
    # grafico male
    plt.figure(figsize=(15, 8))
    custom_greens = sns.light_palette("#66c2a5", n_colors=12, reverse=True)[2:12]
    patches, texts, autotexts = plt.pie(df_male['total_views'], labels=df_male['genres'], autopct='%1.1f%%', startangle=140, 
        colors=custom_greens, pctdistance=0.75, radius=0.8, wedgeprops={'edgecolor': "#3f7f6b", 'linewidth': 1.5})
    centre_circle = plt.Circle((0,0), 0.45, fc='white', edgecolor="#3f7f6b", linewidth=1)
    plt.gca().add_artist(centre_circle)
    plt.setp(autotexts, size=11, weight="bold")
    plt.title('Top 10 Genres - Male Users', fontweight='bold', fontsize=15, pad=20)
    plt.axis('equal')  
    plt.tight_layout()
    plt.show()
    # grafico non-binary
    plt.figure(figsize=(15, 8))
    custom_purples = sns.light_palette("#8da0cb", n_colors=12, reverse=True)[2:12]
    patches, texts, autotexts = plt.pie(df_nb['total_views'], labels=df_nb['genres'], autopct='%1.1f%%', startangle=140, 
        colors=custom_purples, pctdistance=0.75, radius=0.8, wedgeprops={'edgecolor': "#596683", 'linewidth': 1.5})
    centre_circle = plt.Circle((0,0), 0.45, fc='white', edgecolor='#596683', linewidth=1)
    plt.gca().add_artist(centre_circle)
    plt.setp(autotexts, size=11, weight="bold", color="black")
    plt.title('Top 10 Genres - Non-Binary Users', fontweight='bold', fontsize=15, pad=20)
    plt.axis('equal')  
    plt.tight_layout()
    plt.show()

# Settings per geopandas
def plot_location(df):
    df_copy = df.copy()
    filepath = "./ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
    world = gpd.read_file(filepath)
    name_corrections = {
        'United States': 'United States of America',
        'South Korea': 'Korea, Republic of',
        'Vietnam': 'Viet Nam'
    }
    df_copy['location'] = df_copy['location'].replace(name_corrections)
    world_users = world.merge(df_copy, left_on='NAME', right_on='location', how='left')
    return world_users

    

