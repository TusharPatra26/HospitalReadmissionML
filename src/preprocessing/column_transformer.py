"""ColumnTransformer assembly with feature name preservation for SHAP explainability."""

from typing import List, Optional
from sklearn.compose import ColumnTransformer
from src.preprocessing.numerical_scaler import get_numerical_scaler
from src.preprocessing.categorical_encoder import get_categorical_encoder
from src.feature_engineering.feature_builder import get_feature_columns
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_column_transformer(
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    scaler_type: str = "standard",
) -> ColumnTransformer:
    """Constructs a ColumnTransformer combining numerical scaling and one-hot encoding.

    Args:
        numerical_cols: List of numerical column names.
        categorical_cols: List of categorical column names.
        scaler_type: 'standard', 'robust', 'minmax', or 'passthrough'.

    Returns:
        Configured ColumnTransformer.
    """
    if numerical_cols is None or categorical_cols is None:
        canon_num, canon_cat = get_feature_columns()
        numerical_cols = numerical_cols or canon_num
        categorical_cols = categorical_cols or canon_cat

    num_transformer = get_numerical_scaler(scaler_type=scaler_type)
    cat_transformer = get_categorical_encoder(sparse_output=False, handle_unknown="ignore")

    transformer = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_cols),
            ("cat", cat_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    logger.info(
        "Built ColumnTransformer with %d numerical columns (%s scaler) and %d categorical columns.",
        len(numerical_cols),
        scaler_type,
        len(categorical_cols),
    )
    return transformer

