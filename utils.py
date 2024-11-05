import pandas as pd

def load_data(partition):
    
    df = pd.read_csv('./data/Wellbeing_and_lifestyle_data_Kaggle.csv')
    df = df[df['DAILY_STRESS'] != '1/1/00']
    df["AGE"] = df['AGE'].map({"Less than 20": 0, "21 to 35": 1, "36 to 50": 2, "51 or more": 3}).fillna(0)
    df["GENDER"] = df["GENDER"].map({"Female": 0, "Male": 1}).fillna(0)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
    df['year'] = df['Timestamp'].dt.year
    df['month'] = df['Timestamp'].dt.month
    df['day'] = df['Timestamp'].dt.day
    df = df.drop(columns=['Timestamp'])

    # condition on partition to get different partitions of data
    print("Dataset loaded successfully.")
    print("data partition: " + partition)

    return df

def get_similarity_graph(embeddings):
    pass

def get_clusters(similarity_graph):
    pass