import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(csv_path):
    """
    Load user-item ratings from a CSV file.
    Assumes columns: user_id, item_id, rating
    """
    df = pd.read_csv(csv_path)
    return df

def create_user_item_matrix(df):
    """
    Create a user-item rating matrix from the DataFrame.
    """
    user_item_matrix = df.pivot(index='userId', columns='movieId', values='rating')
    return user_item_matrix

def normalize_matrix(matrix):
    """
    Normalize the user-item matrix using Min-Max scaling (per user).
    """
    scaler = MinMaxScaler()
    normalized = matrix.copy()
    normalized[:] = scaler.fit_transform(np.nan_to_num(matrix))
    return normalized

def train_test_split_matrix(df, test_size=0.2, random_state=42):
    """
    Split the ratings DataFrame into train and test sets.
    """
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, test

if __name__ == "__main__":
    # Example usage
    csv_path = "ratings.csv"  # Update with your path
    df = load_data(csv_path)
    train_df, test_df = train_test_split_matrix(df)
    train_matrix = create_user_item_matrix(train_df)
    test_matrix = create_user_item_matrix(test_df)
    normalized_train_matrix = normalize_matrix(train_matrix)

    # Save matrices if needed
    train_matrix.to_csv("train_matrix.csv")
    test_matrix.to_csv("test_matrix.csv")
    pd.DataFrame(normalized_train_matrix, 
                 index=train_matrix.index, 
                 columns=train_matrix.columns).to_csv("normalized_train_matrix.csv")