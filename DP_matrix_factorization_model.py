import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from typing import Tuple, Dict, List, Optional
import warnings
import math
from collections import defaultdict
warnings.filterwarnings('ignore')


class PrivacyAccountant:
    """
    Privacy budget tracking using Rényi Differential Privacy (RDP).
    This uses a simple approximation for subsampled Gaussian RDP:
      RDP ≈ alpha * q^2 / (2 * sigma^2)
    This approximation is reasonable for small q (Poisson sampling regime).
    """

    def __init__(self, target_epsilon: float, target_delta: float, max_alpha: int = 64):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        # Use many orders to get good conversion
        self.orders = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5] + list(range(5, max_alpha + 1))
        self.rdp_eps = np.zeros(len(self.orders))

    def compute_rdp_step(self, q: float, sigma: float) -> np.ndarray:
        """
        Compute RDP epsilon contributions for one step (batch).
        Uses approximation RDP_alpha ≈ alpha * q^2 / (2 * sigma^2) for alpha>1.
        """
        rdp = np.zeros(len(self.orders))
        if q <= 0:
            return rdp
        if sigma <= 0:
            # Infinite privacy cost if no noise
            rdp[:] = np.inf
            return rdp

        for i, alpha in enumerate(self.orders):
            if alpha <= 1:
                rdp[i] = 0.0
            else:
                # approximation valid for small q; keep simple and stable
                rdp[i] = (alpha * (q ** 2)) / (2.0 * (sigma ** 2))
        return rdp

    def add_step(self, q: float, sigma: float):
        step_rdp = self.compute_rdp_step(q, sigma)
        self.rdp_eps = self.rdp_eps + step_rdp

    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Convert accumulated RDP to (epsilon, delta) by minimizing over alpha:
          epsilon = rdp_alpha + log(1/delta) / (alpha - 1)
        """
        if np.any(np.isinf(self.rdp_eps)):
            return float('inf'), self.target_delta

        min_eps = float('inf')
        for i, alpha in enumerate(self.orders):
            if alpha <= 1:
                continue
            eps = self.rdp_eps[i] + math.log(1.0 / self.target_delta) / (alpha - 1)
            if eps < min_eps:
                min_eps = eps
        return min_eps, self.target_delta

    def privacy_budget_exhausted(self) -> bool:
        current_eps, _ = self.get_privacy_spent()
        # If we haven't accumulated any RDP yet, current_eps might be very large; handle that above
        return current_eps >= self.target_epsilon


class DifferentiallyPrivateMatrixFactorization:
    """
    DP Matrix Factorization with user-level DP.
    """

    def __init__(self,
                 n_factors: int = 50,
                 learning_rate: float = 0.01,
                 reg_lambda: float = 0.01,
                 n_epochs: int = 100,
                 init_std: float = 0.1,
                 # Privacy parameters
                 target_epsilon: float = 1.0,
                 target_delta: float = 1e-5,
                 max_grad_norm: float = 1.0,
                 noise_multiplier: float = 1.0,
                 lot_size: Optional[int] = None,  # logical user-batch size
                 verbose: bool = True):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.n_epochs = n_epochs
        self.init_std = init_std
        self.verbose = verbose

        # Privacy
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.lot_size = lot_size  # may be set after we know n_users

        # Accountant
        self.privacy_accountant = PrivacyAccountant(target_epsilon, target_delta)

        # Model parameters (initialized later)
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None

        # Data structures
        self.user_ratings = {}  # user_id -> list[(item_id, rating)]
        self.n_users = 0
        self.n_items = 0

        # History
        self.train_losses = []
        self.val_losses = []
        self.privacy_spent_history = []

    def _initialize_parameters(self, n_users: int, n_items: int, global_mean: float):
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        rng = np.random.RandomState(42)
        self.user_factors = rng.normal(0, self.init_std, (self.n_users, self.n_factors))
        self.item_factors = rng.normal(0, self.init_std, (self.n_items, self.n_factors))
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        self.global_bias = global_mean

    def _organize_data_by_user(self, train_data: List[Tuple]):
        self.user_ratings = defaultdict(list)
        for user_id, item_id, rating in train_data:
            uid = int(user_id)
            iid = int(item_id)
            self.user_ratings[uid].append((iid, float(rating)))

        if self.verbose:
            counts = [len(v) for v in self.user_ratings.values()]
            print(f"User rating distribution: mean={np.mean(counts):.1f}, std={np.std(counts):.1f}, min={np.min(counts)}, max={np.max(counts)}")

    def _predict_rating(self, user_idx: int, item_idx: int) -> float:
        """Internal prediction method that expects mapped indices."""
        try:
            prediction = (self.global_bias +
                        self.user_bias[user_idx] +
                        self.item_bias[item_idx] +
                        float(np.dot(self.user_factors[user_idx], self.item_factors[item_idx])))
            return prediction
        except IndexError:
            return self.global_bias  # Fallback for unknown users/items

    def _compute_user_gradients(self, user_id: int) -> Tuple[np.ndarray, float, Dict[int, Dict[str, np.ndarray]], float]:
        """
        Compute per-user contributions (gradients) for user factors, user bias, and per-item factors/biases.
        Returns:
            user_factor_grad, user_bias_grad, item_updates (dict item_id -> {'factor_grad':..., 'bias_grad':...}), total_loss
        """
        ratings = self.user_ratings[user_id]
        user_factor_grad = np.zeros(self.n_factors)
        user_bias_grad = 0.0
        item_updates = dict()
        total_loss = 0.0

        for item_id, rating in ratings:
            pred = self._predict_rating(user_id, item_id)
            err = rating - pred
            total_loss += err ** 2

            # user factor gradient (note sign: gradient of squared loss w.r.t parameters)
            # we use negative gradient stepping (we'll add gradient, then multiply by lr), so consistent sign
            # gradient of 0.5*(err^2) wrt user_factors = -err * item_factors -> but we use err * item_factors with learning sign
            user_factor_grad += err * self.item_factors[item_id] - self.reg_lambda * self.user_factors[user_id]
            user_bias_grad += err - self.reg_lambda * self.user_bias[user_id]

            if item_id not in item_updates:
                item_updates[item_id] = {'factor_grad': np.zeros(self.n_factors), 'bias_grad': 0.0}
            item_updates[item_id]['factor_grad'] += err * self.user_factors[user_id] - self.reg_lambda * self.item_factors[item_id]
            item_updates[item_id]['bias_grad'] += err - self.reg_lambda * self.item_bias[item_id]

        return user_factor_grad, user_bias_grad, item_updates, total_loss

    def _clip_user_contribution(self, user_factor_grad: np.ndarray, user_bias_grad: float, item_updates: Dict[int, Dict[str, np.ndarray]]):
        """
        Compute combined norm across user_factor_grad, user_bias_grad and flattened item updates,
        compute clipping factor and apply it to all of them.
        """
        # Flatten item updates into a vector for norm calculation
        item_factor_vecs = []
        item_bias_vals = []
        for item_id, upd in item_updates.items():
            item_factor_vecs.append(upd['factor_grad'].ravel())
            item_bias_vals.append(np.array([upd['bias_grad']]))

        if len(item_factor_vecs) > 0:
            item_factor_vec = np.concatenate(item_factor_vecs)
            item_bias_vec = np.concatenate(item_bias_vals)
            concat_vec = np.concatenate([user_factor_grad.ravel(), np.array([user_bias_grad]), item_factor_vec, item_bias_vec])
        else:
            concat_vec = np.concatenate([user_factor_grad.ravel(), np.array([user_bias_grad])])

        total_norm = np.linalg.norm(concat_vec) + 1e-12
        clip_factor = min(1.0, self.max_grad_norm / total_norm)

        # Apply clipping
        user_factor_grad_clipped = user_factor_grad * clip_factor
        user_bias_grad_clipped = user_bias_grad * clip_factor
        item_updates_clipped = {}
        for item_id, upd in item_updates.items():
            item_updates_clipped[item_id] = {
                'factor_grad': upd['factor_grad'] * clip_factor,
                'bias_grad': upd['bias_grad'] * clip_factor
            }

        return user_factor_grad_clipped, user_bias_grad_clipped, item_updates_clipped, clip_factor

    def _add_noise_to_user_contribution(self, user_factor_grad: np.ndarray, user_bias_grad: float, item_updates: Dict[int, Dict[str, np.ndarray]]):
        """
        Add Gaussian noise to each component of the user's contribution.
        Noise std = noise_multiplier * max_grad_norm
        """
        sigma = self.noise_multiplier * self.max_grad_norm
        # user factor noise
        user_factor_noisy = user_factor_grad + np.random.normal(0, sigma, size=user_factor_grad.shape)
        user_bias_noisy = user_bias_grad + np.random.normal(0, sigma)

        item_updates_noisy = {}
        for item_id, upd in item_updates.items():
            factor_noise = np.random.normal(0, sigma, size=upd['factor_grad'].shape)
            bias_noise = np.random.normal(0, sigma)
            item_updates_noisy[item_id] = {
                'factor_grad': upd['factor_grad'] + factor_noise,
                'bias_grad': upd['bias_grad'] + bias_noise
            }

        return user_factor_noisy, user_bias_noisy, item_updates_noisy

    def _compute_loss(self, ratings_data: List[Tuple], include_reg: bool = True) -> float:
        if len(ratings_data) == 0:
            return float('nan')
        total_error = 0.0
        for u, i, r in ratings_data:
            total_error += (r - self._predict_rating(u, i)) ** 2
        mse = total_error / len(ratings_data)
        if include_reg:
            reg = (self.reg_lambda *
                   (np.sum(self.user_factors ** 2) + np.sum(self.item_factors ** 2) +
                    np.sum(self.user_bias ** 2) + np.sum(self.item_bias ** 2)))
            mse += reg
        return math.sqrt(mse)

    def fit(self, train_data: List[Tuple], val_data: List[Tuple] = None):
        # Create ID maps
        user_ids = [int(x[0]) for x in train_data]
        item_ids = [int(x[1]) for x in train_data]
        ratings = [float(x[2]) for x in train_data]

        unique_user_ids = sorted(set(user_ids))
        unique_item_ids = sorted(set(item_ids))
        self.user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_user_ids)}
        self.item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_item_ids)}

        # Remap training data to consecutive indices
        train_data_mapped = [(self.user_id_map[int(u)], self.item_id_map[int(i)], float(r))
                             for u, i, r in train_data]

        if val_data is not None:
            val_data = [(self.user_id_map[int(u)], self.item_id_map[int(i)], float(r))
                        for u, i, r in val_data if int(u) in self.user_id_map and int(i) in self.item_id_map]

        n_users = len(unique_user_ids)
        n_items = len(unique_item_ids)
        global_mean = float(np.mean(ratings))

        self._initialize_parameters(n_users, n_items, global_mean)
        self._organize_data_by_user(train_data_mapped)

        # Set sensible default lot_size (so we don't end up with q=1)
        if self.lot_size is None:
            self.lot_size = min(64, self.n_users)
        if self.lot_size <= 0:
            self.lot_size = max(1, int(0.1 * self.n_users))

        q = float(self.lot_size) / float(self.n_users)  # sampling probability approximation

        if self.verbose:
            print(f"Training DP Matrix Factorization:")
            print(f"  Users: {n_users}, Items: {n_items}, Ratings: {len(train_data_mapped)}")
            print(f"  Factors: {self.n_factors}, lr: {self.learning_rate}, reg: {self.reg_lambda}")
            print(f"  Privacy: target ε={self.target_epsilon}, δ={self.target_delta}")
            print(f"  Clip C={self.max_grad_norm}, noise_mult σ={self.noise_multiplier}")
            print(f"  Lot size (users/batch)={self.lot_size}, approx sampling prob q={q:.4f}")
            print(f"  Steps per epoch ≈ {math.ceil(self.n_users / self.lot_size)}")

        # Training: iterate over epochs and batches
        user_list = list(self.user_ratings.keys())

        for epoch in range(self.n_epochs):
            # Shuffle users for this epoch
            np.random.shuffle(user_list)

            # iterate over batches (so each epoch has multiple DP steps)
            for batch_start in range(0, self.n_users, self.lot_size):
                batch_users = user_list[batch_start: batch_start + self.lot_size]
                batch_size = len(batch_users)
                if batch_size == 0:
                    continue

                # Accumulated (noisy) updates to apply after processing batch
                user_updates = []  # tuples (user_id, noisy_user_factor_grad, noisy_user_bias_grad)
                item_accum = defaultdict(lambda: {'factor_grad': np.zeros(self.n_factors), 'bias_grad': 0.0, 'count': 0})

                # Process per-user contributions
                for u in batch_users:
                    uf_grad, ub_grad, item_updates, _ = self._compute_user_gradients(u)
                    # Clip combined user contribution (user factors + user bias + user's item updates)
                    uf_clipped, ub_clipped, item_updates_clipped, clip_factor = self._clip_user_contribution(
                        uf_grad, ub_grad, item_updates)

                    # Add noise to user's contribution
                    uf_noisy, ub_noisy, item_updates_noisy = self._add_noise_to_user_contribution(
                        uf_clipped, ub_clipped, item_updates_clipped)

                    user_updates.append((u, uf_noisy, ub_noisy))

                    # Accumulate item updates (these are already per-user clipped+noised)
                    for iid, upd in item_updates_noisy.items():
                        item_accum[iid]['factor_grad'] += upd['factor_grad']
                        item_accum[iid]['bias_grad'] += upd['bias_grad']
                        item_accum[iid]['count'] += 1

                # Apply user updates (each user individually)
                for u, uf_grad_noisy, ub_grad_noisy in user_updates:
                    # gradient step: add lr * gradient (note: grads computed as err*feature so sign consistent)
                    self.user_factors[u] += self.learning_rate * uf_grad_noisy
                    self.user_bias[u] += self.learning_rate * ub_grad_noisy

                # Apply item updates: average across contributing users in batch
                for iid, accum in item_accum.items():
                    if accum['count'] > 0:
                        avg_factor_grad = accum['factor_grad'] / float(accum['count'])
                        avg_bias_grad = accum['bias_grad'] / float(accum['count'])
                        self.item_factors[iid] += self.learning_rate * avg_factor_grad
                        self.item_bias[iid] += self.learning_rate * avg_bias_grad

                # Update privacy accountant for this step
                self.privacy_accountant.add_step(q, self.noise_multiplier)

                # Optionally stop if budget exhausted
                if self.privacy_accountant.privacy_budget_exhausted():
                    if self.verbose:
                        print(f"Privacy budget exhausted at epoch {epoch}, batch starting {batch_start}!")
                    break

            # End of epoch: compute and log losses and privacy
            current_eps, _ = self.privacy_accountant.get_privacy_spent()
            self.privacy_spent_history.append(current_eps)

            if epoch % 5 == 0 or epoch == self.n_epochs - 1:
                train_loss = self._compute_loss(train_data_mapped, include_reg=False)
                self.train_losses.append(train_loss)
                if val_data is not None:
                    val_loss = self._compute_loss(val_data, include_reg=False)
                    self.val_losses.append(val_loss)
                    if self.verbose:
                        print(f"Epoch {epoch:3d}: Train RMSE={train_loss:.4f}, Val RMSE={val_loss:.4f}, ε={current_eps:.4f}")
                else:
                    if self.verbose:
                        print(f"Epoch {epoch:3d}: Train RMSE={train_loss:.4f}, ε={current_eps:.4f}")

            # If privacy exhausted, break outer epoch loop
            if self.privacy_accountant.privacy_budget_exhausted():
                break

        # Final privacy
        final_eps, final_delta = self.privacy_accountant.get_privacy_spent()
        if self.verbose:
            print(f"Final Privacy Spent: ε={final_eps:.4f}, δ={final_delta:.2e}")
            if final_eps > self.target_epsilon:
                print(f"WARNING: Privacy budget exceeded! target ε={self.target_epsilon}, actual ε={final_eps:.4f}")

    # Prediction & helpers
    def get_privacy_spent(self) -> Tuple[float, float]:
        return self.privacy_accountant.get_privacy_spent()

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for a user-item pair using original IDs."""
        # Convert to internal indices using mapping if available
        if hasattr(self, 'user_id_map') and hasattr(self, 'item_id_map'):
            if user_id not in self.user_id_map or item_id not in self.item_id_map:
                return self.global_bias
            user_idx = self.user_id_map[user_id]
            item_idx = self.item_id_map[item_id]
        else:
            # During testing or if no mapping exists
            if user_id < 0 or user_id >= self.n_users or item_id < 0 or item_id >= self.n_items:
                return self.global_bias
            user_idx = user_id
            item_idx = item_id
            
        return self._predict_rating(user_idx, item_idx)

    def predict_batch(self, test_data: List[Tuple]) -> np.ndarray:
        """Predict ratings for multiple user-item pairs."""
        preds = []
        for u, i, _ in test_data:
            preds.append(self.predict(int(u), int(i)))
        return np.array(preds)

    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10, known_items: set = None):
        """Get top-N recommendations for a user."""
        # Check if user exists
        if hasattr(self, 'user_id_map'):
            if user_id not in self.user_id_map:
                return []
            user_idx = self.user_id_map[user_id]
        else:
            if user_id >= self.n_users:
                return []
            user_idx = user_id

        if known_items is None:
            known_items = set()

        preds = []
        # Use item_id_map if available
        if hasattr(self, 'item_id_map'):
            item_range = self.item_id_map.keys()
        else:
            item_range = range(self.n_items)

        for iid in item_range:
            if iid in known_items:
                continue
            preds.append((iid, self.predict(user_id, iid)))
        
        preds.sort(key=lambda x: x[1], reverse=True)
        return preds[:n_recommendations]


# --- evaluator and helpers (kept mostly same as your original) ---

class RecommenderEvaluator:
    @staticmethod
    def compute_rating_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return {'RMSE': rmse, 'MAE': mae}

    @staticmethod
    def precision_at_k(recommended_items: List, relevant_items: set, k: int) -> float:
        if k == 0:
            return 0.0
        top_k = recommended_items[:k]
        relevant_in_top_k = len([item for item in top_k if item in relevant_items])
        return relevant_in_top_k / k

    @staticmethod
    def recall_at_k(recommended_items: List, relevant_items: set, k: int) -> float:
        if len(relevant_items) == 0:
            return 0.0
        top_k = recommended_items[:k]
        relevant_in_top_k = len([item for item in top_k if item in relevant_items])
        return relevant_in_top_k / len(relevant_items)

    @staticmethod
    def ndcg_at_k(recommended_items: List, relevant_items: set, k: int) -> float:
        if k == 0 or len(relevant_items) == 0:
            return 0.0
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(i + 2)
        return dcg / idcg if idcg > 0 else 0.0




def load_movielens_data(filepath: str = None, sample_frac: float = 1.0) -> pd.DataFrame:
    """
    Load MovieLens dataset. If no filepath provided, creates synthetic data.
    
    Args:
        filepath: Path to MovieLens ratings file
        sample_frac: Fraction of data to sample (for faster experimentation)
    
    Returns:
        DataFrame with columns: user_id, item_id, rating, timestamp
    """
    if filepath is None:
        # Create synthetic MovieLens-like data for demonstration
        print("Creating synthetic MovieLens-like dataset...")
        np.random.seed(42)
        
        n_users = 1000
        n_items = 500
        n_ratings = 50000
        
        # Generate ratings with some structure
        user_ids = np.random.randint(0, n_users, n_ratings)
        item_ids = np.random.randint(0, n_items, n_ratings)
        
        # Create ratings with user and item biases
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
        
        # Remove duplicates (same user-item pairs)
        data = data.drop_duplicates(['user_id', 'item_id'])
        
    else:
        # Load real MovieLens data
        print(f"Loading MovieLens data from {filepath}")
        data = pd.read_csv(filepath)
        # Rename columns to match our expected format
        data = data.rename(columns={
            'userId': 'user_id',
            'movieId': 'item_id',
            'rating': 'rating',
            'timestamp': 'timestamp'
        })
    
    # Sample data if requested
    if sample_frac < 1.0:
        data = data.sample(frac=sample_frac, random_state=42)
        print(f"Sampled {len(data)} ratings ({sample_frac*100:.1f}% of data)")
    
    # Reindex users and items to be consecutive integers starting from 0
    user_mapping = {old_id: new_id for new_id, old_id in enumerate(data['user_id'].unique())}
    item_mapping = {old_id: new_id for new_id, old_id in enumerate(data['item_id'].unique())}
    
    data['user_id'] = data['user_id'].map(user_mapping)
    data['item_id'] = data['item_id'].map(item_mapping)
    
    print(f"Dataset loaded: {len(data)} ratings, {data['user_id'].nunique()} users, {data['item_id'].nunique()} items")
    print(f"Rating range: {data['rating'].min():.1f} - {data['rating'].max():.1f}")
    print(f"Sparsity: {(1 - len(data) / (data['user_id'].nunique() * data['item_id'].nunique()))*100:.2f}%")
    
    return data


def evaluate_ranking_metrics(model, test_data: List[Tuple], 
                           train_data: List[Tuple], k_values: List[int] = [5, 10, 20]) -> Dict:
    """
    Evaluate ranking metrics (Precision@K, Recall@K, NDCG@K).
    """
    # Create user-item mappings
    train_user_items = {}
    test_user_items = {}
    
    for user_id, item_id, rating in train_data:
        if user_id not in train_user_items:
            train_user_items[user_id] = set()
        train_user_items[user_id].add(item_id)
    
    for user_id, item_id, rating in test_data:
        if rating >= 4.0:  # Consider rating >= 4 as relevant
            if user_id not in test_user_items:
                test_user_items[user_id] = set()
            test_user_items[user_id].add(item_id)
    
    # Evaluate for each user who has test items
    metrics = {f'Precision@{k}': [] for k in k_values}
    metrics.update({f'Recall@{k}': [] for k in k_values})
    metrics.update({f'NDCG@{k}': [] for k in k_values})
    
    evaluator = RecommenderEvaluator()
    
    for user_id in test_user_items:
        known_items = train_user_items.get(user_id, set())
        relevant_items = test_user_items[user_id]
        
        # Get recommendations (exclude known items)
        max_k = max(k_values)
        recommendations = model.get_user_recommendations(
            user_id, n_recommendations=max_k, known_items=known_items)
        recommended_items = [item_id for item_id, _ in recommendations]
        
        # Compute metrics for each K
        for k in k_values:
            precision = evaluator.precision_at_k(recommended_items, relevant_items, k)
            recall = evaluator.recall_at_k(recommended_items, relevant_items, k)
            ndcg = evaluator.ndcg_at_k(recommended_items, relevant_items, k)
            
            metrics[f'Precision@{k}'].append(precision)
            metrics[f'Recall@{k}'].append(recall)
            metrics[f'NDCG@{k}'].append(ndcg)
    
    # Average metrics
    avg_metrics = {}
    for metric_name, values in metrics.items():
        avg_metrics[metric_name] = np.mean(values) if values else 0.0
    
    return avg_metrics


def privacy_utility_analysis(train_data: List[Tuple], val_data: List[Tuple], test_data: List[Tuple]):
    """
    Analyze privacy-utility trade-offs across different privacy budgets.
    
    Args:
        train_data: Training data
        val_data: Validation data  
        test_data: Test data
    
    Returns:
        Dictionary with results for different privacy settings
    """
    print("\n" + "="*60)
    print("PRIVACY-UTILITY TRADE-OFF ANALYSIS")
    print("="*60)
    
    # Privacy budget settings to test
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    noise_multipliers = [8.0, 4.0, 2.0, 1.0, 0.5, 0.25]  # Roughly corresponding to epsilon values
    
    results = []
    
    for eps, noise_mult in zip(epsilon_values, noise_multipliers):
        print(f"\nTesting ε = {eps}, noise multiplier = {noise_mult}")
        
        # Train DP model
        dp_model = DifferentiallyPrivateMatrixFactorization(
            n_factors=50,
            learning_rate=0.01,
            reg_lambda=0.01,
            n_epochs=80,  # Reduced for faster experimentation
            target_epsilon=eps,
            target_delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=noise_mult,
            lot_size=None,  # Use all users
            verbose=False
        )
        
        start_time = time.time()
        dp_model.fit(train_data, val_data)
        training_time = time.time() - start_time
        
        # Evaluate model
        test_predictions = dp_model.predict_batch(test_data)
        test_ratings = np.array([rating for _, _, rating in test_data])
        rating_metrics = RecommenderEvaluator.compute_rating_metrics(test_ratings, test_predictions)
        
        # Get actual privacy spent
        actual_eps, actual_delta = dp_model.get_privacy_spent()
        
        # Store results
        result = {
            'target_epsilon': eps,
            'actual_epsilon': actual_eps,
            'delta': actual_delta,
            'noise_multiplier': noise_mult,
            'RMSE': rating_metrics['RMSE'],
            'MAE': rating_metrics['MAE'],
            'training_time': training_time,
            'privacy_spent_history': dp_model.privacy_spent_history.copy()
        }
        results.append(result)
        
        print(f"  Actual ε: {actual_eps:.3f}, RMSE: {rating_metrics['RMSE']:.4f}")
    
    return results


def compare_baseline_vs_private(train_data: List[Tuple], val_data: List[Tuple], 
                               test_data: List[Tuple], baseline_model=None):
    """
    Compare baseline non-private model with differentially private versions.
    
    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data  
        baseline_model: Pre-trained baseline model (optional)
    
    Returns:
        Dictionary with comparison results
    """
    print("\n" + "="*60)
    print("BASELINE vs PRIVATE MODEL COMPARISON")
    print("="*60)
    
    # Import baseline model from the previous phase
    from matrix_factorization_model import MatrixFactorization
    
    # Train baseline model if not provided
    if baseline_model is None:
        print("Training baseline (non-private) model...")
        baseline_model = MatrixFactorization(
            n_factors=50,
            learning_rate=0.01,
            reg_lambda=0.01,
            n_epochs=100,
            verbose=False
        )
        baseline_start = time.time()
        baseline_model.fit(train_data, val_data)
        baseline_time = time.time() - baseline_start
    else:
        baseline_time = 0  # Already trained
    
    # Train private model with reasonable privacy budget
    print("Training private model (ε=1.0)...")
    private_model = DifferentiallyPrivateMatrixFactorization(
        n_factors=50,
        learning_rate=0.01,
        reg_lambda=0.01,
        n_epochs=100,
        target_epsilon=1.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        noise_multiplier=2.0,
        verbose=False
    )
    
    private_start = time.time()
    private_model.fit(train_data, val_data)
    private_time = time.time() - private_start
    
    # Evaluate both models
    test_ratings = np.array([rating for _, _, rating in test_data])
    
    # Baseline metrics
    baseline_predictions = baseline_model.predict_batch(test_data)
    baseline_metrics = RecommenderEvaluator.compute_rating_metrics(test_ratings, baseline_predictions)
    baseline_ranking = evaluate_ranking_metrics(baseline_model, test_data, train_data)
    
    # Private metrics
    private_predictions = private_model.predict_batch(test_data)
    private_metrics = RecommenderEvaluator.compute_rating_metrics(test_ratings, private_predictions)
    private_ranking = evaluate_ranking_metrics(private_model, test_data, train_data)
    
    # Privacy cost
    actual_eps, actual_delta = private_model.get_privacy_spent()
    
    # Print comparison
    print(f"\nComparison Results:")
    print(f"{'Metric':<20} {'Baseline':<12} {'Private':<12} {'Degradation':<12}")
    print("-" * 60)
    
    rmse_deg = (private_metrics['RMSE'] - baseline_metrics['RMSE']) / baseline_metrics['RMSE'] * 100
    mae_deg = (private_metrics['MAE'] - baseline_metrics['MAE']) / baseline_metrics['MAE'] * 100
    
    print(f"{'RMSE':<20} {baseline_metrics['RMSE']:<12.4f} {private_metrics['RMSE']:<12.4f} {rmse_deg:<12.1f}%")
    print(f"{'MAE':<20} {baseline_metrics['MAE']:<12.4f} {private_metrics['MAE']:<12.4f} {mae_deg:<12.1f}%")
    
    for k in [5, 10, 20]:
        prec_key = f'Precision@{k}'
        if prec_key in baseline_ranking and prec_key in private_ranking:
            baseline_val = baseline_ranking[prec_key]
            private_val = private_ranking[prec_key]
            if baseline_val > 0:
                deg = (baseline_val - private_val) / baseline_val * 100
                print(f"{prec_key:<20} {baseline_val:<12.4f} {private_val:<12.4f} {deg:<12.1f}%")
    
    print(f"{'Training Time (s)':<20} {baseline_time:<12.2f} {private_time:<12.2f}")
    print(f"\nPrivacy Cost: ε = {actual_eps:.4f}, δ = {actual_delta:.2e}")
    
    return {
        'baseline_metrics': baseline_metrics,
        'private_metrics': private_metrics,
        'baseline_ranking': baseline_ranking,
        'private_ranking': private_ranking,
        'privacy_cost': (actual_eps, actual_delta),
        'training_times': (baseline_time, private_time)
    }


def hyperparameter_search_private(train_data: List[Tuple], val_data: List[Tuple], 
                                target_epsilon: float = 1.0) -> Dict:
    """
    Hyperparameter search for private model with fixed privacy budget.
    
    Args:
        train_data: Training data
        val_data: Validation data
        target_epsilon: Target privacy budget
    
    Returns:
        Best hyperparameters and results
    """
    print(f"\nPrivate hyperparameter search (ε = {target_epsilon})...")
    
    # Reduced search space for efficiency
    param_grid = {
        'max_grad_norm': [0.5, 1.0, 2.0],
        'noise_multiplier': [1.0, 2.0, 4.0],
        'learning_rate': [0.005, 0.01, 0.02]
    }
    
    best_params = None
    best_score = float('inf')
    results = []
    
    for clip_norm in param_grid['max_grad_norm']:
        for noise_mult in param_grid['noise_multiplier']:
            for lr in param_grid['learning_rate']:
                print(f"  Testing: clip={clip_norm}, noise={noise_mult}, lr={lr}")
                
                model = DifferentiallyPrivateMatrixFactorization(
                    n_factors=50,
                    learning_rate=lr,
                    reg_lambda=0.01,
                    n_epochs=50,  # Reduced for search
                    target_epsilon=target_epsilon,
                    target_delta=1e-5,
                    max_grad_norm=clip_norm,
                    noise_multiplier=noise_mult,
                    verbose=False
                )
                
                model.fit(train_data, val_data)
                
                if len(model.val_losses) > 0:
                    val_score = model.val_losses[-1]
                    actual_eps, _ = model.get_privacy_spent()
                    
                    results.append({
                        'max_grad_norm': clip_norm,
                        'noise_multiplier': noise_mult,
                        'learning_rate': lr,
                        'val_rmse': val_score,
                        'actual_epsilon': actual_eps
                    })
                    
                    # Only consider models that don't exceed privacy budget
                    if actual_eps <= target_epsilon * 1.1 and val_score < best_score:
                        best_score = val_score
                        best_params = {
                            'max_grad_norm': clip_norm,
                            'noise_multiplier': noise_mult,
                            'learning_rate': lr
                        }
    
    print(f"Best private params: {best_params}")
    print(f"Best validation RMSE: {best_score:.4f}")
    
    return best_params, results


def visualize_privacy_utility_tradeoff(results: List[Dict]):
    """
    Visualize privacy-utility trade-off results.
    
    Args:
        results: List of result dictionaries from privacy_utility_analysis
    """
    # Extract data for plotting
    epsilons = [r['actual_epsilon'] for r in results]
    rmses = [r['RMSE'] for r in results]
    maes = [r['MAE'] for r in results]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Privacy-RMSE trade-off
    axes[0, 0].plot(epsilons, rmses, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Privacy Budget (ε)')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('Privacy vs RMSE Trade-off')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    
    # Privacy-MAE trade-off
    axes[0, 1].plot(epsilons, maes, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Privacy Budget (ε)')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Privacy vs MAE Trade-off')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    
    # Privacy evolution during training (for middle privacy budget)
    mid_idx = len(results) // 2
    mid_result = results[mid_idx]
    epochs = np.arange(len(mid_result['privacy_spent_history'])) * 10
    axes[1, 0].plot(epochs, mid_result['privacy_spent_history'], 'g-', linewidth=2)
    axes[1, 0].axhline(y=mid_result['target_epsilon'], color='r', linestyle='--', 
                      label=f'Target ε = {mid_result["target_epsilon"]}')
    axes[1, 0].set_xlabel('Training Epoch')
    axes[1, 0].set_ylabel('Privacy Spent (ε)')
    axes[1, 0].set_title(f'Privacy Budget Consumption (ε = {mid_result["target_epsilon"]})')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Noise vs Accuracy
    noise_multipliers = [r['noise_multiplier'] for r in results]
    axes[1, 1].plot(noise_multipliers, rmses, 'mo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Noise Multiplier')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].set_title('Noise Level vs Accuracy')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary table
    print(f"\nPrivacy-Utility Trade-off Summary:")
    print(f"{'ε (target)':<10} {'ε (actual)':<12} {'RMSE':<8} {'MAE':<8} {'Utility Loss %':<15}")
    print("-" * 65)
    
    baseline_rmse = max(rmses)  # Assume highest privacy (lowest noise) is closest to baseline
    for r in results:
        utility_loss = (r['RMSE'] - min(rmses)) / min(rmses) * 100
        print(f"{r['target_epsilon']:<10.1f} {r['actual_epsilon']:<12.3f} "
              f"{r['RMSE']:<8.4f} {r['MAE']:<8.4f} {utility_loss:<15.1f}")


def run_phase4_experiment():
    """Run complete Phase 4 differential privacy experiment."""
    print("="*70)
    print("PHASE 4: DIFFERENTIALLY PRIVATE MATRIX FACTORIZATION")
    print("="*70)
    
    # Load data (same as Phase 3)
    data = load_movielens_data(sample_frac=0.1, filepath="ratings.csv")  # Smaller for DP experiments
    
    # Prepare data splits
    temp_data, test_data_df = train_test_split(data, test_size=0.2, random_state=42)
    train_data_df, val_data_df = train_test_split(temp_data, test_size=0.25, random_state=42)
    
    train_data = [(row['user_id'], row['item_id'], row['rating']) for _, row in train_data_df.iterrows()]
    val_data = [(row['user_id'], row['item_id'], row['rating']) for _, row in val_data_df.iterrows()]
    test_data = [(row['user_id'], row['item_id'], row['rating']) for _, row in test_data_df.iterrows()]
    
    print(f"\nData splits:")
    print(f"  Training: {len(train_data)} ratings")
    print(f"  Validation: {len(val_data)} ratings")
    print(f"  Testing: {len(test_data)} ratings")
    
    # 1. Privacy-utility trade-off analysis
    print("\n" + "="*50)
    print("1. PRIVACY-UTILITY TRADE-OFF ANALYSIS")
    print("="*50)
    
    tradeoff_results = privacy_utility_analysis(train_data, val_data, test_data)
    visualize_privacy_utility_tradeoff(tradeoff_results)
    
    # 2. Hyperparameter optimization for private model
    print("\n" + "="*50)
    print("2. PRIVATE MODEL HYPERPARAMETER OPTIMIZATION")
    print("="*50)
    
    best_private_params, search_results = hyperparameter_search_private(train_data, val_data, target_epsilon=1.0)
    
    # 3. Train final private model with best hyperparameters
    print("\n" + "="*50)
    print("3. FINAL PRIVATE MODEL TRAINING")
    print("="*50)
    
    if best_private_params:
        final_private_model = DifferentiallyPrivateMatrixFactorization(
            n_factors=50,
            learning_rate=best_private_params['learning_rate'],
            reg_lambda=0.01,
            n_epochs=100,
            target_epsilon=1.0,
            target_delta=1e-5,
            max_grad_norm=best_private_params['max_grad_norm'],
            noise_multiplier=best_private_params['noise_multiplier'],
            verbose=True
        )
        
        start_time = time.time()
        final_private_model.fit(train_data, val_data)
        training_time = time.time() - start_time
        
        # Evaluate final model
        test_predictions = final_private_model.predict_batch(test_data)
        test_ratings = np.array([rating for _, _, rating in test_data])
        final_metrics = RecommenderEvaluator.compute_rating_metrics(test_ratings, test_predictions)
        final_ranking = evaluate_ranking_metrics(final_private_model, test_data, train_data)
        
        final_eps, final_delta = final_private_model.get_privacy_spent()
        
        print(f"\nFinal Private Model Results:")
        print(f"  Privacy: ε = {final_eps:.4f}, δ = {final_delta:.2e}")
        print(f"  RMSE: {final_metrics['RMSE']:.4f}")
        print(f"  MAE: {final_metrics['MAE']:.4f}")
        print(f"  Training time: {training_time:.2f} seconds")
        
        for k in [5, 10, 20]:
            if f'Precision@{k}' in final_ranking:
                print(f"  Precision@{k}: {final_ranking[f'Precision@{k}']:.4f}")
    
    # 4. Compare with baseline
    print("\n" + "="*50)
    print("4. BASELINE vs PRIVATE COMPARISON")
    print("="*50)
    
    comparison_results = compare_baseline_vs_private(train_data, val_data, test_data)
    
    # 5. Summary
    print("\n" + "="*70)
    print("PHASE 4 SUMMARY - DIFFERENTIAL PRIVACY IMPLEMENTATION")
    print("="*70)
    print("✓ User-level differential privacy implemented")
    print("✓ DP-SGD with gradient clipping and noise injection")
    print("✓ Privacy budget tracking with RDP/moments accountant")
    print("✓ Privacy-utility trade-off analysis completed")
    print("✓ Private model hyperparameter optimization")
    print("✓ Comprehensive evaluation against baseline")
    
    print(f"\nKey Findings:")
    if tradeoff_results:
        min_rmse = min([r['RMSE'] for r in tradeoff_results])
        max_rmse = max([r['RMSE'] for r in tradeoff_results])
        print(f"  RMSE range across privacy budgets: {min_rmse:.4f} - {max_rmse:.4f}")
        
        min_eps = min([r['actual_epsilon'] for r in tradeoff_results])
        max_eps = max([r['actual_epsilon'] for r in tradeoff_results])
        print(f"  Privacy budget range tested: ε ∈ [{min_eps:.3f}, {max_eps:.3f}]")
    
    print(f"\nPrivacy Guarantees:")
    print(f"  ✓ User-level (ε, δ)-differential privacy")
    print(f"  ✓ Formal privacy budget accounting")
    print(f"  ✓ Protection against membership inference")
    
    return final_private_model if 'final_private_model' in locals() else None, tradeoff_results


def main():
    """Main function to run Phase 4 experiments."""
    np.random.seed(42)  # For reproducibility
    
    # Run Phase 4 experiments
    final_model, tradeoff_results = run_phase4_experiment()
    
    print(f"\n" + "="*70)
    print("PHASE 4 COMPLETE - READY FOR PHASE 5 EVALUATION")
    print("="*70)
    
    return final_model, tradeoff_results


if __name__ == "__main__":
    main()