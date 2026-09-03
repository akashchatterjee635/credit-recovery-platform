import os, textwrap

# 1. Update Feature Contract
fc_path = "backend/engine/feature_contract.py"
with open(fc_path, "r") as f:
    fc_content = f.read()

fc_content = fc_content.replace(
    "cost_weight: float = 1.0",
    "cost_weight: float = 1.0\n    domain: str = 'continuous'"
)
fc_content = fc_content.replace(
    "'DAYS_BIRTH', 'IMMUTABLE', False,",
    "'DAYS_BIRTH', 'IMMUTABLE', False, domain='integer',"
)
fc_content = fc_content.replace(
    "'DAYS_EMPLOYED', 'TIME_EVOLVING', False,",
    "'DAYS_EMPLOYED', 'TIME_EVOLVING', False, domain='integer',"
)
fc_content = fc_content.replace(
    "'BUREAU_ACTIVE_COUNT', 'CONDITIONALLY_ACTIONABLE', True, min_val=0,",
    "'BUREAU_ACTIVE_COUNT', 'CONDITIONALLY_ACTIONABLE', True, min_val=0, domain='integer',"
)
fc_content = fc_content.replace(
    "'PREV_APP_COUNT', 'HISTORICAL_IMMUTABLE', False,",
    "'PREV_APP_COUNT', 'HISTORICAL_IMMUTABLE', False, domain='integer',"
)
with open(fc_path, "w") as f:
    f.write(fc_content)
print("Updated feature_contract.py")


# 2. Update Validator (V_manifold instead of V_durability, fail closed, k-NN fixed)
val_path = "backend/engine/validator.py"
with open(val_path, "r") as f:
    val_content = f.read()

val_content = val_content.replace("V_durability", "V_manifold")
val_content = val_content.replace("_check_durability", "_check_manifold")
val_content = val_content.replace("durability_k", "manifold_k")
val_content = val_content.replace("durability_percentile", "manifold_percentile")

# Replace _check_manifold logic
new_manifold = """
    def _check_manifold(self, cand, v):
        if self.training_data is None or len(self.training_data) == 0:
            return True  # skip if no training data available
        try:
            numeric_cols = [c for c in cand.columns
                           if c in self.training_data.columns
                           and self.training_data[c].dtype in ('float64', 'float32', 'int64', 'int32')]
            if not numeric_cols:
                return True

            if self._train_numeric is None or self._knn_threshold is None:
                train_num = self.training_data[numeric_cols].copy()
                # Use mean imputation for reference distribution
                self._ref_means = train_num.mean()
                train_num = train_num.fillna(self._ref_means)
                
                if len(train_num) == 0:
                    return True
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaled = scaler.fit_transform(train_num)
                from sklearn.neighbors import NearestNeighbors
                # use k+1 because we will evaluate training points against themselves
                nn = NearestNeighbors(n_neighbors=self.manifold_k + 1, metric='euclidean')
                nn.fit(scaled)
                dists, _ = nn.kneighbors(scaled)
                # discard self-distance (the 0th neighbor)
                mean_dists = dists[:, 1:].mean(axis=1)
                self._knn_threshold = float(np.percentile(mean_dists, self.manifold_percentile))
                self._scaler = scaler
                self._nn = nn
                self._durability_cols = numeric_cols

            cand_vals = cand[self._durability_cols].fillna(self._ref_means).values
            cand_scaled = self._scaler.transform(cand_vals)
            # For candidate, we just use k neighbors (since it's not in the training set)
            dists, _ = self._nn.kneighbors(cand_scaled, n_neighbors=self.manifold_k)
            mean_dist = float(dists.mean())

            if mean_dist > self._knn_threshold:
                v.append(f'V_manifold FAILED: kNN distance {mean_dist:.2f} > threshold {self._knn_threshold:.2f} (out of distribution)')
                return False
            return True
        except Exception as e:
            v.append(f'V_manifold UNKNOWN: Guard crashed ({e})')
            return False # Fail closed
"""

import re
val_content = re.sub(r'    def _check_manifold\(self, cand, v\):.*?(?=\n    def|\Z)', new_manifold, val_content, flags=re.DOTALL)
with open(val_path, "w") as f:
    f.write(val_content)
print("Updated validator.py")


# 3. Update SLSQPSolver to round integers
slsqp_path = "backend/engine/solvers/slsqp_solver.py"
with open(slsqp_path, "r") as f:
    slsqp_content = f.read()

# Add rounding right after minimize
replace_target = """        cand = applicant.copy()
        for i, f in enumerate(actionable):
            cand[f] = res.x[i]"""

replace_with = """        cand = applicant.copy()
        for i, f in enumerate(actionable):
            val = res.x[i]
            if self.feature_contract[f].domain == 'integer':
                val = round(val)
            cand[f] = val"""
            
slsqp_content = slsqp_content.replace(replace_target, replace_with)
with open(slsqp_path, "w") as f:
    f.write(slsqp_content)
print("Updated slsqp_solver.py")
