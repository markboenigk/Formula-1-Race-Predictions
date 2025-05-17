"""Model training, evaluation, and prediction utilities for regression tasks."""

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline


def get_k_best_features(x, y, k=10):
    """Selects the top k features using univariate regression tests.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        k (int): Number of top features to select.

    Returns:
        List[str]: List of selected feature names.
    """
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(x, y)
    selected_mask = selector.get_support()
    selected_features = x.columns[selected_mask]
    return selected_features.tolist()


def make_prediction(df_predict,  results_summary_df, results, target_column):
    """Makes predictions using the best-performing model from results.

    Args:
        df_predict (pd.DataFrame): Data to predict on.
        top_features_df (pd.DataFrame): Top selected features per model.
        results_summary_df (pd.DataFrame): Summary of model results.
        results (dict): Dictionary of trained models and metadata.
        target_column (str): Name of the target variable.

    Returns:
        pd.DataFrame: DataFrame with predictions and metadata columns.
    """
    best_model_variant = results_summary_df['Model_Variant'].iloc[0]
    model = results[best_model_variant]['fitted_model']
    model_features = results[best_model_variant]['features']

    x_predict = df_predict[model_features].fillna(0)
    y_pred = model.predict(x_predict)
    y_pred_df = pd.DataFrame(y_pred, columns=[f'predicted_{target_column}'], index=df_predict.index)

    return pd.concat([df_predict[['seasonyear', 'eventname', 'driver']], y_pred_df],
                      axis=1).sort_values(
                      by=f'predicted_{target_column}')


def evaluate_model(name, model, x_train, x_test, y_train, y_test, feature_names=None):
    """Trains and evaluates a model, returning key metrics and feature importances.

    Args:
        name (str): Model name.
        model: Sklearn estimator.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target.
        feature_names (List[str], optional): Names of features used.

    Returns:
        dict: Evaluation results including model, MAE, R², and feature importances.
    """
    model.fit(x_train, y_train)
    try:
        y_pred = model.predict(x_test)
    except ValueError as e:
        print(f"Error during prediction: {e}")
        return None

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    importances = None
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        importances = dict(zip(feature_names, model.feature_importances_))
    elif hasattr(model, 'coef_') and feature_names is not None:
        importances = dict(zip(feature_names, model.coef_))

    return {
        'model_name': name,
        'fitted_model': model,
        'y_pred': y_pred,
        'mae': mae,
        'r2': r2,
        'feature_importances': importances,
        'features': feature_names,
        'num_features': x_train.shape[1],
    }


def summarize_model_results(results_dict):
    """Summarizes metrics and feature importances from multiple models.

    Args:
        results_dict (dict): Output from `evaluate_model`.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Summary and importance DataFrames.
    """
    summary = []
    features = []

    for name, res in results_dict.items():
        summary.append({
            "Model_Variant": name,
            "MAE": res["mae"],
            "R2": res["r2"]
        })
        if res.get("feature_importances"):
            features.extend({
                "Model_Variant": name,
                "Feature": feat,
                "Importance": score
            } for feat, score in res["feature_importances"].items())

    return pd.DataFrame(summary).sort_values("R2", ascending=False), pd.DataFrame(features)


def extract_top_features(feature_importance_df, top_n=10):
    """Extracts top N features for each model variant.

    Args:
        feature_importance_df (pd.DataFrame): DataFrame from `summarize_model_results`.
        top_n (int): Number of top features per model.

    Returns:
        pd.DataFrame: Top features per model variant.
    """
    return (
        feature_importance_df
        .sort_values("Importance", ascending=False)
        .groupby("Model_Variant")
        .head(top_n)
    )


def train_and_evaluate_models(df_train, target_column, features_dict, models):
    """Trains and evaluates multiple models using different feature sets.

    Args:
        df_train (pd.DataFrame): Training data.
        df_predict (pd.DataFrame): Prediction data.
        target_column (str): Target variable.
        features_dict (dict): Dict mapping feature set names to column lists.
        models (dict): Dict mapping model names to sklearn estimators.

    Returns:
        dict: Model evaluation results keyed by model variant.
    """
    results = {}
    for feature_option, features in features_dict.items():
        x = df_train[features].copy()
        y = df_train[target_column].copy()

        if feature_option == 'all_features':
            param_grid = {'select__k': range(5, 31)}
            for name, model in models.items():
                pipeline = Pipeline([
                    ('select', SelectKBest(score_func=f_regression)),
                    ('regressor', model)
                ])
                grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
                grid.fit(x, y)

                best_k = grid.best_params_['select__k']
                selected = x.columns[grid.best_estimator_['select'].get_support()]
                print(f"Best k for {name}: {best_k}")
                print(f"Selected features for {name}: {selected.tolist()}")

                x_train, x_test, y_train, y_test = train_test_split(
                    x[selected].fillna(0), y, test_size=0.2, random_state=42
                )
                result_key = f"{name} (KBest) - {feature_option}"
                results[result_key] = evaluate_model(result_key, model,
                                                    x_train, x_test,
                                                    y_train, y_test,
                                                    x_train.columns)

        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x.fillna(0), y, test_size=0.2, random_state=42
            )
            for name, model in models.items():
                result_key = f"{name} (Raw) - {feature_option}"
                results[result_key] = evaluate_model(result_key, model,
                                                    x_train, x_test,
                                                    y_train, y_test,
                                                    x_train.columns)

    return results


def summarize_and_predict(results, df_predict, top_n, target_column):
    """Generates model summary, selects top features, and runs prediction.

    Args:
        results (dict): Model evaluation results.
        df_predict (pd.DataFrame): Prediction data.
        top_n (int): Number of top features to use.
        target_column (str): Name of target variable.

    Returns:
        Tuple[dict, pd.DataFrame, pd.DataFrame]: Metadata, predictions, and top features.
    """
    summary_df, importance_df = summarize_model_results(results)
    top_features_df = extract_top_features(importance_df, top_n)

    base_model = summary_df[summary_df['Model_Variant'].str.endswith("base_feature")].iloc[0]
    best_model = summary_df.iloc[0]

    metadata = {
        "best_model": best_model['Model_Variant'],
        "best_model_r2": best_model['R2'],
        "base_model_r2": base_model['R2'],
        "target_column": target_column,
    }

    prediction_df = make_prediction(df_predict, summary_df, results, target_column)
    return metadata, prediction_df, top_features_df
