import numpy as np

class Node:
    def __init__(self, feature:int|None = None, threshold:float|None = None, left_child:Node|None = None, right_child:Node|None = None, value:int|None = None) -> None:
        """
        feature: The feature used for splitting at this node. 
        threshold: The threshold used for splitting at this node. 
        left: The left child node. 
        right: The right child node. 
        value: If this node is a leaf node, this attribute represents the predicted value for the target variable.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left_child
        self.right = right_child
        self.value = value
        

class DecisionTree:
    def __init__(self, max_depth:int|None = None, criterion:str = "entropy") -> None:
        """
        max_depth: The maximum depth of the tree. None means that the tree grows until another stopping condition is reached.
        criterion: The impurity measure used to calculate information gain. Can be "entropy" or "gini".
        """
        self.root = None
        self.max_depth = max_depth
        self.criterion = criterion

        if criterion not in ["entropy", "gini"]:
            raise ValueError("criterion must be 'entropy' or 'gini'")

    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        X: The input data to classify.
        Returns: An array containing the predicted class for each data point.
        """

        # Predict the class for each data point
        predictions = []

        for sample in X:
            node = self.root

            while node.value is None:

                # Go left or right depending on the feature value
                if sample[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            predictions.append(node.value)

        return np.array(predictions)

    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """
        X: The training data used to build the tree.
        y: The target class labels for the training data.
        """

        # Build the decision tree with ID3 algorithm
        self.root = self.id_3(X, y, 0)

    def id_3(self, X:np.ndarray, y:np.ndarray, depth:int) -> Node:
        """
        X: The data used to build the current part of the tree.
        y: The target class labels for the data.
        depth: The current depth of the tree.
        Returns: The root node of the tree built from the given data.
        """

        # Stop if all samples have the same class
        if len(np.unique(y)) == 1:
            return Node(value=y[0])
        
        # Stop if all feature values are identical
        if np.all(X == X[0]):
            return Node(value=self.most_common(y))

        # Stop if maximum depth is reached
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=self.most_common(y))

        # Find the best split
        # Start with -inf so the first valid gain is always selected
        best_threshold = None
        best_feature = None
        best_gain = -np.inf

        for feature in range(len(X[0])):

            # Use the median as the threshold
            threshold = np.median(X[:, feature])

            # Split the data into two groups
            left = X[:, feature] <= threshold
            right = X[:, feature] > threshold

            # Skip this split if it doesn't actually divide the data
            # (can happen when many samples share the median value)
            if not np.any(left) or not np.any(right):
                continue

            # Calculate information gain
            gain = self.information_gain(y, y[left], y[right])

            # Save the best split
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

        if best_feature is None:
            return Node(value=self.most_common(y))

        # Split the data
        left = X[:, best_feature] <= best_threshold
        right = X[:, best_feature] > best_threshold

        # Recursively apply the ID3 algorithm to build the left and right subtrees
        left_node = self.id_3(X[left], y[left], depth + 1)

        right_node = self.id_3(X[right], y[right], depth + 1)

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left_child=left_node,
            right_child=right_node
        )


    def most_common(self, y:np.ndarray) -> int:
        """
        y: The class labels to find the most common class from.
        Returns: The class label that occurs most frequently.
        """
        # Find the most common class
        values, counts = np.unique(y, return_counts=True)

        return values[np.argmax(counts)]
    

    def entropy(self, y:np.ndarray) -> float:
        """
        y: The class labels used to calculate entropy.
        Returns: The entropy of the given class labels.
        """
        # Calculate entropy
        values, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        return -np.sum(probabilities * np.log2(probabilities))

    def gini(self, y:np.ndarray) -> float:
        """
        y: The class labels used to calculate Gini impurity.
        Returns: The Gini impurity of the given class labels.
        """
        # Calculate Gini impurity
        values, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        return np.sum(probabilities * (1 - probabilities))

    def information_gain(self, y:np.ndarray, left_y:np.ndarray, right_y:np.ndarray) -> float:
        """
        y: The class labels before the split.
        left_y: The class labels in the left child after the split.
        right_y: The class labels in the right child after the split.
        Returns: The information gain produced by the split.
        """
        # Calculate information gain
        if self.criterion == "entropy":
            parent = self.entropy(y)
            left = self.entropy(left_y)
            right = self.entropy(right_y)

        else:
            parent = self.gini(y)
            left = self.gini(left_y)
            right = self.gini(right_y)

        left_weight = len(left_y) / len(y)
        right_weight = len(right_y) / len(y)

        return parent - (left_weight * left + right_weight * right)


    def print_tree(self, node: Node | None = None, space: str = "") -> None:  
        """
        node: The node to print. If None, the root node is used.
        space: The indentation used when printing the tree.
        """
        if node is None:
            node = self.root

        # Print the class if this is a leaf
        if node.value is not None:
            print(space + "Class:", node.value)
            return

        # Print the decision
        print(space + "Feature", node.feature,
            "<=", node.threshold)

        # Print the left and right branches
        print(space + "Left:")
        self.print_tree(node.left, space + "  ")

        print(space + "Right:")
        self.print_tree(node.right, space + "  ")

def permutation_importance(model, X:np.ndarray, y:np.ndarray, metric, n_repeats:int, seed:int) -> dict:
    """
    model: The trained model to evaluate.
    X: The input data.
    y: The true labels.
    metric: A function that takes the true labels and predicted labels as input and returns a performance score.
    n_repeats: Number of times to permute each feature.
    seed: Random seed for reproducibility.
    Returns: dict with 'importances_mean', 'importances_std', 'importances' (raw, shape (n_features, n_repeats))
    """
    rng = np.random.RandomState(seed)

    # Baseline score with the original, unpermuted data
    reference_score = metric(y, model.predict(X))

    n_features = X.shape[1]
    importances = np.zeros((n_features, n_repeats))

    # For each feature j
    for feature in range(n_features):

        # For each repetition k
        for repeat in range(n_repeats):

            # Create a corrupted version of the dataset
            X_permuted = X.copy()

            # Randomly shuffle feature j
            permuted_indices = rng.permutation(len(X_permuted))
            X_permuted[:, feature] = X_permuted[permuted_indices, feature]

            # Compute the score s_kj
            permuted_score = metric(y, model.predict(X_permuted))

            # Store s - s_kj
            importances[feature, repeat] = (
                reference_score - permuted_score
            )

    # e_j = s - (1/K) * sum(s_kj)
    importance_mean = np.mean(importances, axis=1)

    return {
        "importances_mean": importance_mean,
        "importances_std": np.std(importances, axis=1),
        "importances": importances
    }