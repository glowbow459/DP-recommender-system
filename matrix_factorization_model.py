import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


class MatrixFactorization:
    """
    matrix factorization model for collaborative filtering using SGD
    """

    def __init__(self, n_factors: int = 50, learning_rate: float = 0.01,
                 reg_lambda: float = 0.01, n_epochs: int = 100,
                 init_std: float = 0.1, verbose: bool = True):
        """
        initialize matrix factorization model
        Args:
            n_factors is the number of latent factors
            learning_rate is the learning rate for SGD
            reg_lambda is the L2 regularization parameter
            n_epochs is the number of training epochs
            init_std is the standard deviation for parameter initialization
            verbose means whether to print training progress ot not
            """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.n_epochs = n_epochs
        self.init_std = init_std
        self.verbose = verbose
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None
        self.train_losses = []
        self.val_losses = []

    def _initialize_parameters(self, n_users: int, n_items: int, global_mean: float):
        """
        initialize model parameters
        """
        n_users = int(n_users)
        n_items = int(n_items)
        self.user_factors = np.random.normal(0, self.init_std, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, self.init_std, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = global_mean

    def _predict_rating(self, user_id: int, item_id: int) -> float:
        """
        predict rating for a user item pair
        """
        user_idx = int(user_id)
        item_idx = int(item_id)
        prediction = (self.global_bias +
                      self.user_bias[user_idx] +
                      self.item_bias[item_idx] +
                      np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
        return prediction

    def _compute_loss(self, ratings_data: List[Tuple], include_reg: bool = True) -> float:
        """
        compute RMSE loss on given rating data
        """
        total_error = 0
        for user_id, item_id, rating in ratings_data:
            prediction = self._predict_rating(user_id, item_id)
            total_error += (rating - prediction) ** 2

        mse = total_error / len(ratings_data)

        if include_reg:
            reg_loss = (self.reg_lambda *
                        (np.sum(self.user_factors ** 2) +
                         np.sum(self.item_factors ** 2) +
                         np.sum(self.user_bias ** 2) +
                         np.sum(self.item_bias ** 2)))
            mse += reg_loss

        return np.sqrt(mse)

    def fit(self, train_data: List[Tuple], val_data: List[Tuple] = None):
        """
        train the matrix factorization model using SGD
        Args:
            train_data is a list of (user_id, item_id, rating) tuples
            val_data is the optional validation data for monitoring
        """
        user_ids = [int(x[0]) for x in train_data]
        item_ids = [int(x[1]) for x in train_data]
        ratings = [float(x[2]) for x in train_data]

        n_users = len(user_ids)
        n_items = len(item_ids)
        global_mean = float(np.mean(ratings))

        self._initialize_parameters(n_users, n_items, global_mean)

        if self.verbose:
            print(f"training the model:")
            print(f" num of users: {n_users},num of items: {n_items}")
            print(f"  ratings: {len(train_data)}, factors: {self.n_factors}")
            print(f"  learning rate: {self.learning_rate}, regularization: {self.reg_lambda}")

        for epoch in range(self.n_epochs):
            np.random.shuffle(train_data)

            for user_id, item_id, rating in train_data:
                user_idx = int(user_id)
                item_idx = int(item_id)

                prediction = self._predict_rating(user_idx, item_idx)
                error = rating - prediction

                user_factors_old = self.user_factors[user_idx].copy()
                item_factors_old = self.item_factors[item_idx].copy()

                self.user_factors[user_idx] += self.learning_rate * (
                        error * item_factors_old - self.reg_lambda * user_factors_old)
                self.item_factors[item_idx] += self.learning_rate * (
                        error * user_factors_old - self.reg_lambda * item_factors_old)

                self.user_bias[user_idx] += self.learning_rate * (
                        error - self.reg_lambda * self.user_bias[user_idx])
                self.item_bias[item_idx] += self.learning_rate * (
                        error - self.reg_lambda * self.item_bias[item_idx])

            if epoch % 10 == 0 or epoch == self.n_epochs - 1:
                train_loss = self._compute_loss(train_data, include_reg=False)
                self.train_losses.append(train_loss)

                if val_data:
                    val_loss = self._compute_loss(val_data, include_reg=False)
                    self.val_losses.append(val_loss)

                    if self.verbose:
                        print(f"epoch number {epoch:3d}: train RMSE={train_loss:.4f}, val RMSE={val_loss:.4f}")
                else:
                    if self.verbose:
                        print(f"epoch number {epoch:3d}: train RMSE={train_loss:.4f}")

    def predict(self, user_id: int, item_id: int) -> float:
        """
        predict rating for a single user item pair
        """
        if (user_id >= len(self.user_factors) or
                item_id >= len(self.item_factors) or
                user_id < 0 or item_id < 0):
            return self.global_bias

        return self._predict_rating(user_id, item_id)

    def predict_batch(self, test_data: List[Tuple]) -> np.ndarray:
        """
        predict ratings for a batch of user item pairs
        """
        predictions = []
        for user_id, item_id, _ in test_data:
            predictions.append(self.predict(user_id, item_id))
        return np.array(predictions)

    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10,
                                 known_items: set = None) -> List[Tuple]:
        """
        get top N recommendations for a user
        Args:
            user_id is the user ID
            n_recommendations is the number of recommendations
            known_items is a set of items already rated by user

        Returns:
            list of (item_id, predicted_rating) tuples
        """
        if user_id >= len(self.user_factors):
            return []

        if known_items is None:
            known_items = set()

        predictions = []
        for item_id in range(len(self.item_factors)):
            if item_id not in known_items:
                pred_rating = self.predict(user_id, item_id)
                predictions.append((item_id, pred_rating))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]


class RecommenderEvaluator:
    """
    evaluation utilities for recommender systems
    """

    @staticmethod
    def compute_rating_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        compute rating prediction metrics
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        return {
            'RMSE': rmse,
            'MAE': mae
        }

    @staticmethod
    def precision_at_k(recommended_items: List, relevant_items: set, k: int) -> float:
        """
        compute Precision@K
        """
        if k == 0:
            return 0.0

        top_k = recommended_items[:k]
        relevant_in_top_k = len([item for item in top_k if item in relevant_items])
        return relevant_in_top_k / k

    @staticmethod
    def recall_at_k(recommended_items: List, relevant_items: set, k: int) -> float:
        """
        Compute Recall@K
        """
        if len(relevant_items) == 0:
            return 0.0

        top_k = recommended_items[:k]
        relevant_in_top_k = len([item for item in top_k if item in relevant_items])
        return relevant_in_top_k / len(relevant_items)

    @staticmethod
    def ndcg_at_k(recommended_items: List, relevant_items: set, k: int) -> float:
        """
        compute NDCG@K (simplified binary relevance)
        """
        if k == 0 or len(relevant_items) == 0:
            return 0.0

        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0

        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0


def load_movielens_data(filepath: str = None, sample_frac: float = 1.0) -> pd.DataFrame:
    """
    load movielens dataset if there is no filepath create data
    Args:
        filepath is the path to the movielens ratings file
        sample_frac is the fraction of data to sample
    Returns:
        dataframe with  the next columns: user_id, item_id, rating, timestamp
    """
    if filepath is None:
        np.random.seed(42)

        n_users = 1000
        n_items = 500
        n_ratings = 50000

        user_ids = np.random.randint(0, n_users, n_ratings)
        item_ids = np.random.randint(0, n_items, n_ratings)

        user_bias = np.random.normal(0, 0.5, n_users)
        item_bias = np.random.normal(0, 0.3, n_items)

        ratings = []
        for u, i in zip(user_ids, item_ids):
            base_rating = 3.5 + user_bias[u] + item_bias[i]
            noise = np.random.normal(0, 0.3)
            rating = np.clip(base_rating + noise, 1, 5)
            ratings.append(rating)

        data = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': ratings,
            'timestamp': np.random.randint(0, 1000000, n_ratings)
        })

        data = data.drop_duplicates(['user_id', 'item_id'])

    else:
        data = pd.read_csv(filepath)
        data = data.rename(columns={
            'userId': 'user_id',
            'movieId': 'item_id',
            'rating': 'rating',
            'timestamp': 'timestamp'
        })

    if sample_frac < 1.0:
        data = data.sample(frac=sample_frac, random_state=42)

    user_mapping = {old_id: new_id for new_id, old_id in enumerate(data['user_id'].unique())}
    item_mapping = {old_id: new_id for new_id, old_id in enumerate(data['item_id'].unique())}

    data['user_id'] = data['user_id'].map(user_mapping)
    data['item_id'] = data['item_id'].map(item_mapping)

    print(f"dataset loaded: {len(data)} ratings, {data['user_id'].nunique()} users, {data['item_id'].nunique()} items")
    print(f"rating range: {data['rating'].min():.1f} - {data['rating'].max():.1f}")
    print(f"sparsity: {(1 - len(data) / (data['user_id'].nunique() * data['item_id'].nunique())) * 100:.2f}%")

    return data


def prepare_train_test_data(data: pd.DataFrame, test_size: float = 0.2,
                            random_state: int = 42) -> Tuple[List[Tuple], List[Tuple]]:
    """
    prepare training and testing data
    Args:
        data is the dataframe with user_id, item_id and rating columns
        test_size is the fraction of data for testing
        random_state is a random seed
    Returns:
        tuple of (train_data, test_data) as lists of (user_id, item_id, rating) tuples
    """
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)

    train_data = [(row['user_id'], row['item_id'], row['rating'])
                  for _, row in train_df.iterrows()]
    test_data = [(row['user_id'], row['item_id'], row['rating'])
                 for _, row in test_df.iterrows()]

    return train_data, test_data


def evaluate_ranking_metrics(model: MatrixFactorization, test_data: List[Tuple],
                             train_data: List[Tuple], k_values: List[int] = [5, 10, 20]) -> Dict:
    """
    evaluate ranking metrics (Precision@K, Recall@K, NDCG@K)
    Args:
        model is the trained matrix factorization model
        test_data is the test ratings
        train_data is the training ratings
        k_values is a list of K values to evaluate
    Returns:
        dictionary with ranking metrics
    """
    train_user_items = {}
    test_user_items = {}

    for user_id, item_id, rating in train_data:
        if user_id not in train_user_items:
            train_user_items[user_id] = set()
        train_user_items[user_id].add(item_id)

    for user_id, item_id, rating in test_data:
        if rating >= 4.0:
            if user_id not in test_user_items:
                test_user_items[user_id] = set()
            test_user_items[user_id].add(item_id)

    metrics = {f'precision@{k}': [] for k in k_values}
    metrics.update({f'recall@{k}': [] for k in k_values})
    metrics.update({f'NDCG@{k}': [] for k in k_values})

    evaluator = RecommenderEvaluator()

    for user_id in test_user_items:
        known_items = train_user_items.get(user_id, set())
        relevant_items = test_user_items[user_id]

        max_k = max(k_values)
        recommendations = model.get_user_recommendations(
            user_id, n_recommendations=max_k, known_items=known_items)
        recommended_items = [item_id for item_id, _ in recommendations]

        for k in k_values:
            precision = evaluator.precision_at_k(recommended_items, relevant_items, k)
            recall = evaluator.recall_at_k(recommended_items, relevant_items, k)
            ndcg = evaluator.ndcg_at_k(recommended_items, relevant_items, k)

            metrics[f'precision@{k}'].append(precision)
            metrics[f'recall@{k}'].append(recall)
            metrics[f'NDCG@{k}'].append(ndcg)

    avg_metrics = {}
    for metric_name, values in metrics.items():
        avg_metrics[metric_name] = np.mean(values) if values else 0.0

    return avg_metrics


def hyperparameter_search(train_data: List[Tuple], val_data: List[Tuple]) -> Dict:
    """
    perform basic hyperparameter search.
    Returns:
        best hyperparameters and their validation performance
    """
    print("\nperforming hyper param comparison")

    param_grid = {
        'n_factors': [20, 50, 100],
        'learning_rate': [0.005, 0.01, 0.02],
        'reg_lambda': [0.001, 0.01, 0.1]
    }

    results = []

    best_params = {
        'n_factors': param_grid['n_factors'][0],
        'learning_rate': param_grid['learning_rate'][0],
        'reg_lambda': param_grid['reg_lambda'][0]
    }

    initial_model = MatrixFactorization(
        n_factors=best_params['n_factors'],
        learning_rate=best_params['learning_rate'],
        reg_lambda=best_params['reg_lambda'],
        n_epochs=50,
        verbose=False
    )
    initial_model.fit(train_data, val_data)
    best_score = initial_model.val_losses[-1]

    for n_factors in param_grid['n_factors']:
        for lr in param_grid['learning_rate']:
            for reg in param_grid['reg_lambda']:
                print(f"testing for the next params: factors={n_factors}, lr={lr}, reg={reg}")

                model = MatrixFactorization(
                    n_factors=n_factors,
                    learning_rate=lr,
                    reg_lambda=reg,
                    n_epochs=50,
                    verbose=False
                )

                model.fit(train_data, val_data)
                val_score = model.val_losses[-1]

                results.append({
                    'n_factors': n_factors,
                    'learning_rate': lr,
                    'reg_lambda': reg,
                    'val_rmse': val_score
                })

                if val_score < best_score:
                    best_score = val_score
                    best_params = {
                        'n_factors': n_factors,
                        'learning_rate': lr,
                        'reg_lambda': reg
                    }

    print(f"\nbest params: {best_params}")
    print(f"best val RMSE: {best_score:.4f}")

    return best_params, results


def run_baseline_experiment(use_best_params: bool = False):
    """run complete baseline experiment"""

    data = load_movielens_data(sample_frac=0.3, filepath="ratings1M.csv")

    temp_data, test_data_df = train_test_split(data, test_size=0.2, random_state=42)
    train_data_df, val_data_df = train_test_split(temp_data, test_size=0.25, random_state=42)

    train_data = [(row['user_id'], row['item_id'], row['rating']) for _, row in train_data_df.iterrows()]
    val_data = [(row['user_id'], row['item_id'], row['rating']) for _, row in val_data_df.iterrows()]
    test_data = [(row['user_id'], row['item_id'], row['rating']) for _, row in test_data_df.iterrows()]

    if use_best_params:
        print("\nusing the best params")
        best_params = {
            'n_factors': 50,
            'learning_rate': 0.005,
            'reg_lambda': 0.1
        }
        search_results = []
    else:
        best_params, search_results = hyperparameter_search(train_data, val_data)

    final_model = MatrixFactorization(
        n_factors=best_params['n_factors'],
        learning_rate=best_params['learning_rate'],
        reg_lambda=best_params['reg_lambda'],
        n_epochs=100,
        verbose=True
    )

    start_time = time.time()
    final_model.fit(train_data, val_data)
    training_time = time.time() - start_time

    test_predictions = final_model.predict_batch(test_data)
    test_ratings = np.array([rating for _, _, rating in test_data])
    rating_metrics = RecommenderEvaluator.compute_rating_metrics(test_ratings, test_predictions)

    ranking_metrics = evaluate_ranking_metrics(final_model, test_data, train_data)

    print(f"\nbest hyper params:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    print(f"\nrating prediction metrics:")
    for metric, value in rating_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print(f"\nranking metrics:")
    for metric, value in ranking_metrics.items():
        print(f"  {metric}: {value:.4f}")

    if not use_best_params:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        epochs = np.arange(0, len(final_model.train_losses)) * 10
        plt.plot(epochs, final_model.train_losses, label='Training RMSE', marker='o')
        plt.plot(epochs, final_model.val_losses, label='Validation RMSE', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        search_df = pd.DataFrame(search_results)
        pivot_data = search_df.pivot_table(values='val_rmse',
                                           index='learning_rate',
                                           columns='n_factors',
                                           aggfunc='min')

        plt.imshow(pivot_data.values, cmap='viridis', aspect='auto')
        plt.colorbar(label='Validation RMSE')
        plt.xticks(range(len(pivot_data.columns)), pivot_data.columns)
        plt.yticks(range(len(pivot_data.index)), [f"{x:.3f}" for x in pivot_data.index])
        plt.xlabel('Number of Factors')
        plt.ylabel('Learning Rate')
        plt.title('hyperparameter Search Results')

        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(6, 4))
        epochs = np.arange(0, len(final_model.train_losses)) * 10
        plt.plot(epochs, final_model.train_losses, label='Training RMSE', marker='o')
        plt.plot(epochs, final_model.val_losses, label='Validation RMSE', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return final_model, rating_metrics, ranking_metrics, best_params


def analyze_model_properties(model: MatrixFactorization):
    """analyze properties of the trained model"""
    user_factor_norms = np.linalg.norm(model.user_factors, axis=1)
    item_factor_norms = np.linalg.norm(model.item_factors, axis=1)

    print(f"user factor statistics:")
    print(f"  mean norm: {np.mean(user_factor_norms):.4f}")
    print(f"  std norm: {np.std(user_factor_norms):.4f}")
    print(f"  min/max norm: {np.min(user_factor_norms):.4f} / {np.max(user_factor_norms):.4f}")

    print(f"\nitem factor statistics:")
    print(f"  mean norm: {np.mean(item_factor_norms):.4f}")
    print(f"  std norm: {np.std(item_factor_norms):.4f}")
    print(f"  min/max norm: {np.min(item_factor_norms):.4f} / {np.max(item_factor_norms):.4f}")

    print(f"\nbias statistics:")
    print(f"  global bias: {model.global_bias:.4f}")
    print(f"  user bias range: {np.min(model.user_bias):.4f} to {np.max(model.user_bias):.4f}")
    print(f"  item bias range: {np.min(model.item_bias):.4f} to {np.max(model.item_bias):.4f}")

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.hist(user_factor_norms, bins=30, alpha=0.7, color='blue')
    plt.xlabel('User Factor Norm')
    plt.ylabel('Frequency')
    plt.title('Distribution of User Factor Norms')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.hist(item_factor_norms, bins=30, alpha=0.7, color='red')
    plt.xlabel('Item Factor Norm')
    plt.ylabel('Frequency')
    plt.title('Distribution of Item Factor Norms')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.hist(model.user_bias, bins=30, alpha=0.7, color='green', label='User Bias')
    plt.hist(model.item_bias, bins=30, alpha=0.7, color='orange', label='Item Bias')
    plt.xlabel('Bias Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Bias Terms')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main(use_best_params: bool = False):
    np.random.seed(42)

    baseline_model, rating_metrics, ranking_metrics, best_params = run_baseline_experiment(use_best_params)

    analyze_model_properties(baseline_model)

    for metric, value in rating_metrics.items():
        print(f"  {metric}: {value:.4f}")

    return baseline_model, rating_metrics, ranking_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Matrix Factorization Recommender')
    parser.add_argument('--use-best-params', action='store_true',
                        help='Use best known hyperparameters instead of searching')
    args = parser.parse_args()

    main(use_best_params=False)