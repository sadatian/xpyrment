r"""DragonNet Deep CATE Estimator.

Implements Farrell, Liang, and Misra (2021) and Shi, Blei, and Veitch (2019) multi-headed
neural network architecture for predicting individual treatment effects (ITE) and
conditional average treatment effects (CATE) using pure NumPy.
"""

from typing import Dict, Tuple, Optional
import logging
import numpy as np

from xpyrment.core.exceptions import PhaseOrderError

logger = logging.getLogger(__name__)

class DragonNet:
    r"""DragonNet Deep Representation Joint-Learning CATE Estimator.

    DragonNet is a multi-headed neural network designed to resolve confounding bias
    when estimating individual treatment effects. It accomplishes this by learning a
    joint representation of covariates that predicts both the propensity score (treatment assignment)
    and the potential outcomes (under control and treatment). By enforcing a shared representation
    layer, the network preserves covariates necessary for predicting both outcome surfaces, while
    the propensity head adjusts for treatment selection bias.

    ??? mathbox "Mathematical Specifications"

        Let $X \in \mathbb{R}^{N \times d}$ be the covariates, $T \in \{0, 1\}^N$ be the treatment indicator,
        and $Y \in \mathbb{R}^N$ be the observed continuous outcomes.
        The network consists of three sequential components:

        1. **Shared Representation Layer ($\Phi$)**:
           Maps high-dimensional inputs to a lower-dimensional latent representation:
           $$
           Z_1 = X W_{\Phi} + b_{\Phi}, \quad Z = \tanh(Z_1)
           $$
           where $W_{\Phi} \in \mathbb{R}^{d \times h}$ and $b_{\Phi} \in \mathbb{R}^{1 \times h}$.

        2. **Control Outcome Head ($Y_0$)**:
           Predicts outcome under control ($T = 0$) using representation $Z$:
           $$
           A_{0} = Z W_{0,h} + b_{0,h}, \quad H_0 = \tanh(A_{0})
           $$
           $$
           \hat{y}_0 = H_0 W_{0,o} + b_{0,o}
           $$

        3. **Treatment Outcome Head ($Y_1$)**:
           Predicts outcome under treatment ($T = 1$) using representation $Z$:
           $$
           A_{1} = Z W_{1,h} + b_{1,h}, \quad H_1 = \tanh(A_{1})
           $$
           $$
           \hat{y}_1 = H_1 W_{1,o} + b_{1,o}
           $$

        4. **Propensity Score Head ($e$)**:
           Predicts treatment assignment probability using representation $Z$:
           $$
           A_{t} = Z W_{t,h} + b_{t,h}, \quad H_t = \tanh(A_{t})
           $$
           $$
           E_{\text{logit}} = H_t W_{t,o} + b_{t,o}, \quad \hat{e} = \sigma(E_{\text{logit}})
           $$
           where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid activation function.

        The total joint optimization objective function minimizes the combined loss:
        $$
        L(Y, T, X) = L_{\text{outcome}}(Y, T, \Phi(X)) + \alpha \cdot L_{\text{propensity}}(T, \Phi(X)) + L_{\text{reg}}
        $$
        where:
        - Outcome Loss:
          $$
          L_{\text{outcome}} = \frac{1}{N} \sum_{i=1}^N \left[ (1 - T_i)(Y_i - \hat{y}_{0,i})^2 + T_i(Y_i - \hat{y}_{1,i})^2 \right]
          $$
        - Propensity Cross-Entropy Loss (with clipping $\epsilon$ bounds):
          $$
          \hat{e}_i^{\text{clipped}} = \text{clip}(\hat{e}_i, \epsilon, 1 - \epsilon)
          $$
          $$
          L_{\text{propensity}} = -\frac{1}{N} \sum_{i=1}^N \left[ T_i \log(\hat{e}_i^{\text{clipped}}) + (1 - T_i) \log(1 - \hat{e}_i^{\text{clipped}}) \right]
          $$
        - L2 Regularization Penalty (applied only to weight matrices $\mathcal{W}$):
          $$
          L_{\text{reg}} = \frac{\lambda}{2} \sum_{W \in \mathcal{W}} \|W\|_F^2
          $$

    ### Gradient Backpropagation & Optimization

    ??? mathbox "Backpropagation Equations"

        Using the chain rule, gradients with respect to output predictions are computed at each training step:

        - Control outcome gradient:
          $$
          \delta_{\hat{y}_0} = -\frac{2}{N} (1 - T) \odot (Y - \hat{y}_0)
          $$
        - Treatment outcome gradient:
          $$
          \delta_{\hat{y}_1} = -\frac{2}{N} T \odot (Y - \hat{y}_1)
          $$
        - Propensity logit gradient (via sigmoid cross-entropy cancellation):
          $$
          \delta_{E_{\text{logit}}} = \frac{\alpha}{N} (\hat{e}^{\text{clipped}} - T)
          $$

        Each head propagates its respective gradients back to the shared representation layer:
        - Outcome 0 head representation gradient:
          $$
          \delta_Z^{(0)} = \left[ (\delta_{\hat{y}_0} W_{0,o}^T) \odot (1 - H_0^2) \right] W_{0,h}^T
          $$
        - Outcome 1 head representation gradient:
          $$
          \delta_Z^{(1)} = \left[ (\delta_{\hat{y}_1} W_{1,o}^T) \odot (1 - H_1^2) \right] W_{1,h}^T
          $$
        - Propensity head representation gradient:
          $$
          \delta_Z^{(t)} = \left[ (\delta_{E_{\text{logit}}} W_{t,o}^T) \odot (1 - H_t^2) \right] W_{t,h}^T
          $$

        The total representation layer gradient is the sum:
        $$
        \delta_Z = \delta_Z^{(0)} + \delta_Z^{(1)} + \delta_Z^{(t)}
        $$
        which is backpropagated to the inputs through:
        $$
        \delta_{Z_1} = \delta_Z \odot (1 - Z^2)
        $$

    Attributes:
        shared_hidden_dim (int): Dimensionality of the shared representation layer.
        outcome_hidden_dim (int): Dimensionality of outcome head hidden layers.
        propensity_hidden_dim (int): Dimensionality of propensity head hidden layer.
        alpha (float): Scaling hyperparameter balancing propensity cross-entropy.
        lambda_reg (float): L2 regularization weight penalty coefficient.
        learning_rate (float): Initial learning rate for parameters updates.
        optimizer (str): Parameter solver algorithm ('adam' or 'sgd').
        momentum (float): Exponential decay rate for SGD momentum updates.
        batch_size (int): Size of training batches (uses full batch if None).
        epochs (int): Maximum training epochs.
        tol (float): Convergence tolerance threshold for early stopping.
        clip_epsilon (float): Propensity score prediction clipping bounds.
        params (Dict[str, np.ndarray]): Dictionary containing network parameters weights/biases.
    """

    def __init__(
        self,
        shared_hidden_dim: int = 64,
        outcome_hidden_dim: int = 32,
        propensity_hidden_dim: int = 32,
        alpha: float = 1.0,
        lambda_reg: float = 1e-4,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        momentum: float = 0.9,
        batch_size: Optional[int] = 64,
        epochs: int = 200,
        tol: float = 1e-6,
        clip_epsilon: float = 1e-7,
        random_state: Optional[int] = None,
    ) -> None:
        self.shared_hidden_dim = shared_hidden_dim
        self.outcome_hidden_dim = outcome_hidden_dim
        self.propensity_hidden_dim = propensity_hidden_dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.optimizer = optimizer.lower()
        self.momentum = momentum
        self.batch_size = batch_size
        self.epochs = epochs
        self.tol = tol
        self.clip_epsilon = clip_epsilon
        self.random_state = random_state

        if self.optimizer not in ("adam", "sgd"):
            raise ValueError(f"Unsupported optimizer: {optimizer}. Must be 'adam' or 'sgd'.")

        self.params: Dict[str, np.ndarray] = {}
        self._is_fitted = False

        # Adam moment tracking variables
        self._m: Dict[str, np.ndarray] = {}
        self._v: Dict[str, np.ndarray] = {}
        self._t = 0  # Timestep tracker

        # SGD Momentum tracking variables
        self._velocity: Dict[str, np.ndarray] = {}

    def _initialize_weights(self, input_dim: int, rng: np.random.Generator) -> None:
        """Initializes network weights using Xavier/Glorot Normal initialization."""
        dims = {
            "W_phi": (input_dim, self.shared_hidden_dim),
            "b_phi": (1, self.shared_hidden_dim),
            
            "W_0h": (self.shared_hidden_dim, self.outcome_hidden_dim),
            "b_0h": (1, self.outcome_hidden_dim),
            "W_0o": (self.outcome_hidden_dim, 1),
            "b_0o": (1, 1),
            
            "W_1h": (self.shared_hidden_dim, self.outcome_hidden_dim),
            "b_1h": (1, self.outcome_hidden_dim),
            "W_1o": (self.outcome_hidden_dim, 1),
            "b_1o": (1, 1),
            
            "W_th": (self.shared_hidden_dim, self.propensity_hidden_dim),
            "b_th": (1, self.propensity_hidden_dim),
            "W_to": (self.propensity_hidden_dim, 1),
            "b_to": (1, 1),
        }

        for name, shape in dims.items():
            if name.startswith("W_"):
                # Xavier Normal variance: 2 / (fan_in + fan_out)
                fan_in, fan_out = shape[0], shape[1]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                self.params[name] = rng.normal(0.0, std, size=shape).astype(np.float64)
            else:
                self.params[name] = np.zeros(shape, dtype=np.float64)

            # Initialize optimizer variables
            self._m[name] = np.zeros_like(self.params[name])
            self._v[name] = np.zeros_like(self.params[name])
            self._velocity[name] = np.zeros_like(self.params[name])

        self._t = 0

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    def _forward(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Runs the forward propagation pass and returns intermediate activations."""
        activations: Dict[str, np.ndarray] = {}

        # 1. Shared layer
        activations["Z1"] = X @ self.params["W_phi"] + self.params["b_phi"]
        activations["Z"] = np.tanh(activations["Z1"])

        # 2. Control Outcome Head
        activations["A0"] = activations["Z"] @ self.params["W_0h"] + self.params["b_0h"]
        activations["H0"] = np.tanh(activations["A0"])
        activations["y0_hat"] = activations["H0"] @ self.params["W_0o"] + self.params["b_0o"]

        # 3. Treatment Outcome Head
        activations["A1"] = activations["Z"] @ self.params["W_1h"] + self.params["b_1h"]
        activations["H1"] = np.tanh(activations["A1"])
        activations["y1_hat"] = activations["H1"] @ self.params["W_1o"] + self.params["b_1o"]

        # 4. Propensity Head
        activations["At"] = activations["Z"] @ self.params["W_th"] + self.params["b_th"]
        activations["Ht"] = np.tanh(activations["At"])
        activations["e_logit"] = activations["Ht"] @ self.params["W_to"] + self.params["b_to"]
        activations["e_hat"] = self._sigmoid(activations["e_logit"])

        return activations

    def _backward(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        activations: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Computes accurate gradients for all parameters via backpropagation."""
        N = X.shape[0]
        grads: Dict[str, np.ndarray] = {}

        # Output predictions and activations
        y0_hat = activations["y0_hat"]
        y1_hat = activations["y1_hat"]
        e_hat = activations["e_hat"]
        e_clipped = np.clip(e_hat, self.clip_epsilon, 1.0 - self.clip_epsilon)

        H0 = activations["H0"]
        H1 = activations["H1"]
        Ht = activations["Ht"]
        Z = activations["Z"]

        # 1. Gradients from Outcome 0 Head
        delta_y0 = -2.0 / N * (1.0 - T) * (Y - y0_hat)
        grads["W_0o"] = H0.T @ delta_y0 + self.lambda_reg * self.params["W_0o"]
        grads["b_0o"] = np.sum(delta_y0, axis=0, keepdims=True)

        delta_H0 = delta_y0 @ self.params["W_0o"].T
        delta_A0 = delta_H0 * (1.0 - H0**2)
        grads["W_0h"] = Z.T @ delta_A0 + self.lambda_reg * self.params["W_0h"]
        grads["b_0h"] = np.sum(delta_A0, axis=0, keepdims=True)

        delta_Z_0 = delta_A0 @ self.params["W_0h"].T

        # 2. Gradients from Outcome 1 Head
        delta_y1 = -2.0 / N * T * (Y - y1_hat)
        grads["W_1o"] = H1.T @ delta_y1 + self.lambda_reg * self.params["W_1o"]
        grads["b_1o"] = np.sum(delta_y1, axis=0, keepdims=True)

        delta_H1 = delta_y1 @ self.params["W_1o"].T
        delta_A1 = delta_H1 * (1.0 - H1**2)
        grads["W_1h"] = Z.T @ delta_A1 + self.lambda_reg * self.params["W_1h"]
        grads["b_1h"] = np.sum(delta_A1, axis=0, keepdims=True)

        delta_Z_1 = delta_A1 @ self.params["W_1h"].T

        # 3. Gradients from Propensity Head (Sigmoid Binary Cross-Entropy)
        delta_e_logit = self.alpha / N * (e_clipped - T)
        grads["W_to"] = Ht.T @ delta_e_logit + self.lambda_reg * self.params["W_to"]
        grads["b_to"] = np.sum(delta_e_logit, axis=0, keepdims=True)

        delta_Ht = delta_e_logit @ self.params["W_to"].T
        delta_At = delta_Ht * (1.0 - Ht**2)
        grads["W_th"] = Z.T @ delta_At + self.lambda_reg * self.params["W_th"]
        grads["b_th"] = np.sum(delta_At, axis=0, keepdims=True)

        delta_Z_t = delta_At @ self.params["W_th"].T

        # 4. Shared Representation Backpropagation
        delta_Z = delta_Z_0 + delta_Z_1 + delta_Z_t
        delta_Z1 = delta_Z * (1.0 - Z**2)

        grads["W_phi"] = X.T @ delta_Z1 + self.lambda_reg * self.params["W_phi"]
        grads["b_phi"] = np.sum(delta_Z1, axis=0, keepdims=True)

        return grads

    def _update_parameters(self, grads: Dict[str, np.ndarray]) -> None:
        """Updates parameters weights using either Adam or SGD-Momentum solver."""
        if self.optimizer == "adam":
            self._t += 1
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            for name in self.params:
                # Update moments
                self._m[name] = beta1 * self._m[name] + (1.0 - beta1) * grads[name]
                self._v[name] = beta2 * self._v[name] + (1.0 - beta2) * (grads[name] ** 2)
                
                # Bias correction
                m_corrected = self._m[name] / (1.0 - beta1 ** self._t)
                v_corrected = self._v[name] / (1.0 - beta2 ** self._t)
                
                # Parameter update
                self.params[name] -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + eps)
        
        elif self.optimizer == "sgd":
            for name in self.params:
                # Update velocity
                self._velocity[name] = self.momentum * self._velocity[name] + self.learning_rate * grads[name]
                # Parameter update
                self.params[name] -= self._velocity[name]

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> "DragonNet":
        """Fits the DragonNet neural network joint representation model.

        Args:
            X (np.ndarray): Covariate matrix of shape (N, d).
            treatment (np.ndarray): Binary treatment indicators of shape (N,) or (N, 1).
            y (np.ndarray): Continuous outcome variable of shape (N,) or (N, 1).

        Returns:
            DragonNet: The fitted estimator instance.
        """
        # Validate data shapes and bounds
        X_arr = np.asarray(X, dtype=np.float64)
        T_arr = np.asarray(treatment, dtype=np.float64).reshape(-1, 1)
        Y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        N, d = X_arr.shape

        if N == 0:
            raise ValueError("DragonNet cannot fit on an empty dataset (N=0).")
        if N <= 5:
            raise ValueError(f"Insufficient samples for training DragonNet neural network (N={N}).")
        if not np.all(np.isin(T_arr, [0.0, 1.0])):
            raise ValueError("Treatment indicator T must contain only binary values (0 or 1).")
        if np.any(np.isnan(X_arr)) or np.any(np.isnan(T_arr)) or np.any(np.isnan(Y_arr)):
            raise ValueError("Input arrays contain invalid NaN values.")
        if np.any(np.isinf(X_arr)) or np.any(np.isinf(T_arr)) or np.any(np.isinf(Y_arr)):
            raise ValueError("Input arrays contain infinite (inf) values.")

        # Guard against single-class treatment assignment
        unique_t = np.unique(T_arr)
        if len(unique_t) < 2:
            raise ValueError("DragonNet requires variation in treatment assignment; only one treatment arm was found.")

        # Shared/Head dimensions validation
        if self.shared_hidden_dim <= 0 or self.outcome_hidden_dim <= 0 or self.propensity_hidden_dim <= 0:
            raise ValueError("Network layer dimensionalities must be strictly positive integers.")

        # Initialize pseudo-random generator
        rng = np.random.default_rng(self.random_state)
        self._initialize_weights(d, rng)

        # Learning Loop
        prev_loss = np.inf
        
        for epoch in range(self.epochs):
            # Batch division
            indices = np.arange(N)
            rng.shuffle(indices)

            X_shuffled = X_arr[indices]
            T_shuffled = T_arr[indices]
            Y_shuffled = Y_arr[indices]

            batch_size = self.batch_size if self.batch_size is not None and self.batch_size > 0 else N
            batch_size = min(batch_size, N)

            epoch_losses = []
            
            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                X_batch = X_shuffled[start_idx:end_idx]
                T_batch = T_shuffled[start_idx:end_idx]
                Y_batch = Y_shuffled[start_idx:end_idx]

                # Forward pass
                activations = self._forward(X_batch)

                # Compute gradients via joint backpropagation
                grads = self._backward(X_batch, T_batch, Y_batch, activations)

                # Update weights
                self._update_parameters(grads)

                # Calculate intermediate batch losses
                y0_hat = activations["y0_hat"]
                y1_hat = activations["y1_hat"]
                e_hat = activations["e_hat"]
                e_clipped = np.clip(e_hat, self.clip_epsilon, 1.0 - self.clip_epsilon)

                loss_y = np.mean((1.0 - T_batch) * (Y_batch - y0_hat)**2 + T_batch * (Y_batch - y1_hat)**2)
                loss_e = -np.mean(T_batch * np.log(e_clipped) + (1.0 - T_batch) * np.log(1.0 - e_clipped))
                
                # Weight decay regularization calculation
                l2_penalty = 0.0
                for name, w in self.params.items():
                    if name.startswith("W_"):
                        l2_penalty += np.sum(w**2)
                l2_penalty *= 0.5 * self.lambda_reg

                batch_total_loss = loss_y + self.alpha * loss_e + l2_penalty
                epoch_losses.append(batch_total_loss)

            current_loss = np.mean(epoch_losses)

            # Check convergence tolerance
            if abs(prev_loss - current_loss) < self.tol:
                logger.info(f"DragonNet training converged early at epoch {epoch + 1}.")
                break
                
            prev_loss = current_loss

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts potential outcomes under control and treatment arms.

        Args:
            X (np.ndarray): Covariates matrix of shape (N, d).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - y0_pred: Predicted outcomes under control (N,).
                - y1_pred: Predicted outcomes under treatment (N,).
        """
        if not self._is_fitted:
            raise PhaseOrderError("DragonNet instance is not fitted yet. Call .fit() first.")

        X_arr = np.asarray(X, dtype=np.float64)
        activations = self._forward(X_arr)
        
        y0_pred = activations["y0_hat"].squeeze(axis=-1)
        y1_pred = activations["y1_hat"].squeeze(axis=-1)

        return y0_pred, y1_pred

    def predict_propensity(self, X: np.ndarray) -> np.ndarray:
        """Predicts treatment propensity scores.

        Args:
            X (np.ndarray): Covariates matrix of shape (N, d).

        Returns:
            np.ndarray: Predicted propensity scores of shape (N,).
        """
        if not self._is_fitted:
            raise PhaseOrderError("DragonNet instance is not fitted yet. Call .fit() first.")

        X_arr = np.asarray(X, dtype=np.float64)
        activations = self._forward(X_arr)
        
        return activations["e_hat"].squeeze(axis=-1)

    def estimate_effect(self, X: np.ndarray) -> np.ndarray:
        r"""Estimates individual treatment effects (ITE) / uplift.

        Computes the Conditional Average Treatment Effect (CATE) for each unit:
        $$
        \hat{\tau}(X) = \hat{y}_1(X) - \hat{y}_0(X)
        $$

        Args:
            X (np.ndarray): Covariates matrix of shape (N, d).

        Returns:
            np.ndarray: Predicted treatment effect (ITE/CATE) of shape (N,).
        """
        y0_pred, y1_pred = self.predict(X)
        return y1_pred - y0_pred
