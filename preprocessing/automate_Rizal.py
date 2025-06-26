import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_iris(input_csv='iris_raw.csv', output_csv='preprocessing/iris_preprocessing.csv'):
    df = pd.read_csv(input_csv)
    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])
    df.to_csv(output_csv, index=False)
    print(f'Preprocessing selesai! Hasil di: {output_csv}')

if __name__ == '__main__':
    preprocess_iris()