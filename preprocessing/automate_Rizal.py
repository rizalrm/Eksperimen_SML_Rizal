import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_penguins(input_path, output_path):
    # 1. Load data
    df = pd.read_csv(input_path)

    # 2. Hapus missing values
    df_clean = df.dropna()

    # 3. Hapus duplikasi
    df_clean = df_clean.drop_duplicates()

    # 4. Ambil fitur numerik
    X = df_clean[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].copy()
    y = df_clean['species'].copy()

    # 5. Label encoding untuk kolom target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 6. Standarisasi fitur numerik
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # 7. Gabungkan fitur hasil scaling + label numerik
    df_processed = X_scaled_df.copy()
    df_processed['species'] = y_encoded

    # 8. Simpan ke file CSV
    df_processed.to_csv(output_path, index=False)
    print(f"[INFO] Data preprocessing selesai. File hasil: {output_path}")

    return df_processed

# Contoh pemanggilan:
if __name__ == "__main__":
    preprocess_penguins('penguins.csv', 'penguins_preprocessing.csv')
