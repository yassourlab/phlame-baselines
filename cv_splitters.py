"""Cross-validation splitter helpers for benchmarking tools."""


def build_stratified_group_cv(n_splits=5, random_state=17):
    """Create a stratified grouped cross-validator."""
    from sklearn.model_selection import StratifiedGroupKFold

    return StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def build_group_cv(n_splits=5):
    """Create a grouped cross-validator without stratification."""
    from sklearn.model_selection import GroupKFold

    return GroupKFold(n_splits=n_splits)
