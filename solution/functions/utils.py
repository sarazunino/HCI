import pandas as pd

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


# Sostituisce i NaN nelle colonne di tipo 'object' con 'Unknown'
def fill_object_nan(df, replacement="Unknown"):
    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].fillna(replacement)
    return df

