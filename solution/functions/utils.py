import pandas as pd

# FUNZIONI UTILI

# mostra la dimensione del dataframe (riga, colonna)
def show_shape(df, name="DataFrame"):
    print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")


# mostra il numero di duplicati
def check_duplicates(df, name="DataFrame"):
    num_duplicates = df.duplicated().sum()
    total_duplicated_rows = df.duplicated(keep=False).sum()
    unique_records_with_dupes = total_duplicated_rows - num_duplicates
    print(f"In '{name}': {unique_records_with_dupes} original records produced {num_duplicates} extra copies.")
    

# rimuove i duplicati
def clean_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)


# mostra quali valori sono mancanti e dove
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


# cancella le righe con tutti i valori NaN
def drop_empty_rows(df, name="DataFrame"):
    initial_count = len(df)
    df_cleaned = df.dropna(how='all')
    removed = initial_count - len(df_cleaned)
    if removed > 0:
        print(f"In {name}: removed {removed} empty rows")
    return df_cleaned

# mostra se un valore è unico
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
    

# converte in ToDateTime
def convert_to_datetime(df, column_name):
    if column_name in df.columns:
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce', format='%Y').fillna(
            pd.to_datetime(df[column_name], errors='coerce')
        )
        print(f"Column '{column_name}' successfully converted.")
    else:
        print(f"Column '{column_name}' not found in DataFrame.")
    return df


# riempe i valori NaN negli object con 'Unknown'
def fill_object_nan(df, replacement="Unknown"):
    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].fillna(replacement)
    return df

