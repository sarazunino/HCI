# FUNZIONI UTILI

# mostra la dimensione del dataframe (riga, colonna)
def show_shape(df, name="DataFrame"):
    print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")


# mostra i valori mancanti in ogni dataset
def show_missing(df, name="DataFrame"): 
    missing = df.isna().sum().sum()
    print(f"{name}: {missing} missing values")


# mostra se un valore è unico
def check_unique(df, col):
    if col in df.columns:
        if df[col].is_unique:
            print(f"'{col}' può essere usata come index per il DataFrame")
            return True
        else:
            print(f"'{col}' non può essere index in quanto non unica")
            return False
    else:
        print(f"La colonna '{col}' non esiste nel DataFrame.")
        return False


# mostra il numero di duplicati
def check_duplicates(df, name="DataFrame"):
    num_duplicates = df.duplicated().sum()
    total_duplicated_rows = df.duplicated(keep=False).sum()
    gruppi_unici = total_duplicated_rows - num_duplicates
    
    print(f"Righe duplicate nel dataset {name}: {num_duplicates}, duplicati totali: {total_duplicated_rows}")
    print(f"In sintesi: ci sono {gruppi_unici} record originali che hanno prodotto {num_duplicates} copie extra.")
    

# rimuove i duplicati
def clean_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)
