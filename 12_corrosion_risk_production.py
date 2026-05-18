#!/usr/bin/env python3
"""
Pipeline Corrosion Risk Ranking with Machine Learning
Production script for ranking pipeline joints by corrosion failure risk.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_pipeline_corrosion_data(n_joints=5000, random_seed=42):
    """
    Generate synthetic joint-level corrosion data.
    Features represent real integrity management data sources:
    - ILI metal loss measurements
    - CP survey potentials
    - Soil resistivity tests
    - Coating type and age
    - Environmental and consequence factors
    Returns:
        DataFrame with features and corrosion failure labels
    """
    rng = np.random.default_rng(random_seed)
    # Generate feature distributions matching field data
    df = pd.DataFrame(
        {
            "age_years": rng.integers(1, 60, n_joints),
            "soil_resistivity": rng.normal(3000, 800, n_joints).clip(200, 8000),  # ohm-cm
            "cp_potential": rng.normal(-0.95, 0.08, n_joints),  # V vs Cu/CuSO4
            "coating": rng.choice(
                ["FBE", "PE", "CoalTar", "Tape"], n_joints, p=[0.4, 0.3, 0.2, 0.1]
            ),
            "near_water": rng.choice([0, 1], n_joints, p=[0.8, 0.2]),
            "hca_distance_m": rng.exponential(1500, n_joints),  # High Consequence Area
            "pressure_psig": rng.normal(800, 60, n_joints),
            "temp_c": rng.normal(18, 8, n_joints),
            "ili_metal_loss": rng.beta(1.5, 10, n_joints) * 100,  # percent wall thickness
        }
    )
    # Generate failure labels using realistic corrosion physics
    coating_degradation_map = {
        "FBE": 0.0,  # Fusion Bonded Epoxy - best durability
        "PE": 0.3,  # Polyethylene - good
        "CoalTar": 0.6,  # Coal Tar - moderate (legacy)
        "Tape": 0.9,  # Tape wrap - poor (legacy)
    }
    risk_logit = (
        0.04 * df["age_years"]
        + -0.001 * df["soil_resistivity"]
        + -4.0 * (df["cp_potential"] + 0.85)
        + df["near_water"] * 1.0
        + df["coating"].map(coating_degradation_map).fillna(0) * 1.5
        + 0.025 * df["ili_metal_loss"]
        - 2.5  # Offset to get ~10-15% failure rate
    )
    prob = 1 / (1 + np.exp(-risk_logit))
    df["corrosion_fail"] = (rng.random(n_joints) < prob).astype(int)
    logger.info(f"Generated {n_joints} pipeline joints:")
    logger.info(f"  Age range: {df['age_years'].min()} - {df['age_years'].max()} years")
    logger.info(
        f"  Soil resistivity: {df['soil_resistivity'].min():.0f} - {df['soil_resistivity'].max():.0f} ohm-cm"
    )
    logger.info(
        f"  CP potential: {df['cp_potential'].min():.3f} - {df['cp_potential'].max():.3f} V"
    )
    logger.info(
        f"  ILI metal loss: {df['ili_metal_loss'].min():.1f}% - {df['ili_metal_loss'].max():.1f}%"
    )
    logger.error(f"  Failure rate: {df['corrosion_fail'].mean():.1%}")
    logger.info("  Coating distribution:")
    for coating, count in df["coating"].value_counts().items():
        logger.info(f"    {coating}: {count} joints ({count / len(df) * 100:.1f}%)")

    return df


def prepare_features(df):
    """
    Split features and target, identify numeric and categorical columns.
    Returns:
        X, y, numeric_cols, categorical_cols
    """
    y = df["corrosion_fail"]
    X = df.drop(columns=["corrosion_fail"])
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = ["coating"]
    logger.info("\nFeature preparation:")
    logger.info(f"  Numeric features ({len(numeric_cols)}): {', '.join(numeric_cols)}")
    logger.info(f"  Categorical features ({len(categorical_cols)}): {', '.join(categorical_cols)}")
    return X, y, numeric_cols, categorical_cols


def train_corrosion_risk_model(X, y, numeric_cols, categorical_cols):
    """
    Train gradient boosting classifier to predict corrosion failure risk.
    Returns:
        Trained pipeline, test predictions, metrics
    """
    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                categorical_cols,
            ),
        ]
    )
    # Build full pipeline
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(
                    max_depth=4, learning_rate=0.08, n_estimators=400, random_state=42
                ),
            ),
        ]
    )
    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    logger.info("\nTraining corrosion risk classifier:")
    logger.info(f"  Training set: {len(X_train)} joints")
    logger.info(f"  Test set: {len(X_test)} joints")
    logger.error(f"  Positive class (failures) in test: {y_test.sum()} ({y_test.mean():.1%})")
    # Train
    model.fit(X_train, y_train)
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    # Evaluate
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    # Calculate precision/recall at different thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    # Find optimal threshold (maximize F1)
    f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-8)
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]
    optimal_precision = precisions[optimal_idx]
    optimal_recall = recalls[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    logger.info("\nModel Performance:")
    logger.info(f"  ROC AUC: {roc_auc:.3f}")
    logger.info(f"  Average Precision: {avg_precision:.3f}")
    logger.info(f"  Optimal Threshold: {optimal_threshold:.3f}")
    logger.info(f"  Precision @ Optimal: {optimal_precision:.3f}")
    logger.info(f"  Recall @ Optimal: {optimal_recall:.3f}")
    logger.info(f"  F1 Score @ Optimal: {optimal_f1:.3f}")
    return (
        model,
        X_test,
        y_test,
        y_pred_proba,
        {
            "roc_auc": roc_auc,
            "avg_precision": avg_precision,
            "optimal_threshold": optimal_threshold,
            "optimal_precision": optimal_precision,
            "optimal_recall": optimal_recall,
            "optimal_f1": optimal_f1,
        },
    )


def analyze_feature_importance(model, X, numeric_cols, categorical_cols):
    """
    Extract and analyze feature importance from gradient boosting model.
    Returns:
        DataFrame with feature importances
    """
    # Get feature names after preprocessing
    cat_encoder = model.named_steps["preprocessor"].named_transformers_["cat"]
    cat_features = list(cat_encoder.get_feature_names_out(categorical_cols))
    all_features = numeric_cols + cat_features
    # Get importance scores
    importances = model.named_steps["classifier"].feature_importances_
    # Create DataFrame
    importance_df = pd.DataFrame({"feature": all_features, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    logger.info("\nFeature Importance (Top 10):")
    for row in importance_df.head(10).itertuples():
        logger.info(f"  {row.feature:<25} {row.importance:.3f}")

    # Group by category
    logger.info("\nImportance by Category:")
    logger.info(
        f"  ILI Data (metal_loss):        {importance_df[importance_df['feature'] == 'ili_metal_loss']['importance'].sum():.3f}"
    )
    logger.info(
        f"  CP Data (cp_potential):       {importance_df[importance_df['feature'] == 'cp_potential']['importance'].sum():.3f}"
    )
    logger.info(
        f"  Soil (soil_resistivity):      {importance_df[importance_df['feature'] == 'soil_resistivity']['importance'].sum():.3f}"
    )
    logger.info(
        f"  Age:                          {importance_df[importance_df['feature'] == 'age_years']['importance'].sum():.3f}"
    )
    coating_importance = importance_df[
        importance_df["feature"].str.contains("coating", case=False)
    ]["importance"].sum()
    logger.info(f"  Coating Type:                 {coating_importance:.3f}")
    return importance_df


def create_work_list(model, X_test, y_test, y_pred_proba, budget_joints=50):
    """
    Rank joints by risk and create prioritized work list.
    Returns:
        DataFrame with top priority joints
    """
    # Create risk-scored dataset
    risk_df = X_test.copy()
    risk_df["risk_score"] = y_pred_proba
    risk_df["actual_failure"] = y_test.values
    # Estimate work costs
    base_cost = 15000
    hca_multiplier = np.maximum(0, 100 - risk_df["hca_distance_m"] / 20)
    risk_df["work_cost"] = base_cost + 100 * hca_multiplier
    # Estimate risk value
    risk_df["risk_value"] = 100000 * risk_df["risk_score"]
    # Value per dollar
    risk_df["value_per_dollar"] = risk_df["risk_value"] / risk_df["work_cost"]
    # Create work list
    work_list = risk_df.sort_values("value_per_dollar", ascending=False).head(budget_joints)
    # Calculate capture rate
    total_failures = y_test.sum()
    captured_failures = work_list["actual_failure"].sum()
    capture_rate = captured_failures / total_failures
    logger.info("\nWork List Summary:")
    logger.info(f"  Budget: {budget_joints} joints")
    logger.info(f"  Total joints: {len(X_test)}")
    logger.info(f"  Budget utilization: {budget_joints / len(X_test) * 100:.1f}%")
    logger.error(f"  Total failures in test set: {total_failures}")
    logger.error(f"  Failures captured in work list: {captured_failures}")
    logger.info(f"  Capture rate: {capture_rate:.1%}")
    logger.info(f"  Average risk score (top 50): {work_list['risk_score'].mean():.3f}")
    logger.info(f"  Average risk score (full set): {risk_df['risk_score'].mean():.3f}")
    logger.info(f"  Total work cost: ${work_list['work_cost'].sum():,.0f}")
    logger.info(f"  Average cost per joint: ${work_list['work_cost'].mean():,.0f}")
    # Display top 10
    logger.info("\nTop 10 Priority Joints:")
    for idx, row in enumerate(work_list.head(10).itertuples(), 1):
        i = row.Index
        logger.info(f"\n  Joint #{idx} (ID: {i}):")
        logger.info(f"    Risk Score: {row.risk_score:.3f}")
        logger.info(f"    Value/Cost: ${row.value_per_dollar:.2f} per $1")
        logger.info(f"    Age: {row.age_years} years, Coating: {row.coating}")
        logger.info(f"    CP: {row.cp_potential:.3f} V, Soil: {row.soil_resistivity:.0f} ohm-cm")
        logger.info(
            f"    Metal Loss: {row.ili_metal_loss:.1f}%, HCA Distance: {row.hca_distance_m:.0f} m"
        )

    return work_list, risk_df


def analyze_business_value(work_list, risk_df, y_test):
    """
    Calculate business value of ML-driven prioritization vs alternatives.
    """
    total_joints = len(risk_df)
    total_failures = y_test.sum()
    # Strategy 1: ML Model (top 50 by value/dollar)
    ml_captured = work_list["actual_failure"].sum()
    ml_cost = work_list["work_cost"].sum()
    # Strategy 2: Sort by ILI metal loss (traditional)
    ili_sorted = risk_df.sort_values("ili_metal_loss", ascending=False).head(50)
    ili_captured = ili_sorted["actual_failure"].sum()
    ili_cost = ili_sorted["work_cost"].sum()
    # Strategy 3: Random sampling (baseline)
    random_sample = risk_df.sample(50, random_state=42)
    random_captured = random_sample["actual_failure"].sum()
    random_cost = random_sample["work_cost"].sum()
    # Strategy 4: Age-based (legacy approach)
    age_sorted = risk_df.sort_values("age_years", ascending=False).head(50)
    age_captured = age_sorted["actual_failure"].sum()
    age_cost = age_sorted["work_cost"].sum()
    logger.info("=== BUSINESS VALUE ANALYSIS ===")
    logger.error(
        f"\nTotal Network: {total_joints} joints, {total_failures} failures ({total_failures / total_joints * 100:.1f}%)"
    )
    logger.info("Inspection Budget: 50 joints (4.0% of network)")
    logger.info()
    strategies = [
        ("ML Model (Value/Cost)", ml_captured, ml_cost),
        ("ILI Metal Loss Sort", ili_captured, ili_cost),
        ("Age-Based Sort", age_captured, age_cost),
        ("Random Sampling", random_captured, random_cost),
    ]
    logger.error(
        f"{'Strategy':<25} {'Failures Captured':<20} {'Capture Rate':<15} {'Cost':<15} {'Cost/Failure'}"
    )
    for strategy, captured, cost in strategies:
        capture_rate = captured / total_failures
        cost_per_failure = cost / captured if captured > 0 else float("inf")
        logger.error(
            f"{strategy:<25} {captured:>8}/{total_failures:<10} {capture_rate:>14.1%} ${cost:>13,.0f} ${cost_per_failure:>12,.0f}"
        )

    # Calculate lift
    ml_lift_vs_ili = ((ml_captured - ili_captured) / ili_captured * 100) if ili_captured > 0 else 0
    ml_lift_vs_age = ((ml_captured - age_captured) / age_captured * 100) if age_captured > 0 else 0
    ml_lift_vs_random = (
        ((ml_captured - random_captured) / random_captured * 100) if random_captured > 0 else 0
    )
    logger.info("\nML Model Lift:")
    logger.error(f"  vs ILI Sort:      +{ml_lift_vs_ili:.1f}% failures captured")
    logger.error(f"  vs Age Sort:      +{ml_lift_vs_age:.1f}% failures captured")
    logger.error(f"  vs Random:        +{ml_lift_vs_random:.1f}% failures captured")
    # Estimate prevented failures
    failure_consequence = 100000
    ml_prevented_cost = ml_captured * failure_consequence
    ili_prevented_cost = ili_captured * failure_consequence
    value_gain = ml_prevented_cost - ili_prevented_cost
    logger.info("\nEstimated Value (vs ILI Sort):")
    logger.error(f"  Additional failures prevented: {ml_captured - ili_captured}")
    logger.error(f"  Value of prevented failures: ${value_gain:,.0f}")
    logger.info(f"  ROI: {value_gain / ml_cost:.1f}x inspection cost")


def main():
    """Complete pipeline corrosion risk ranking pipeline."""
    logger.info("PIPELINE CORROSION RISK RANKING WITH MACHINE LEARNING")
    logger.info()
    # 1. Generate data
    df = generate_pipeline_corrosion_data(n_joints=5000, random_seed=42)
    # 2. Prepare features
    X, y, numeric_cols, categorical_cols = prepare_features(df)
    # 3. Train model
    model, X_test, y_test, y_pred_proba, metrics = train_corrosion_risk_model(
        X, y, numeric_cols, categorical_cols
    )
    # 4. Analyze feature importance
    importance_df = analyze_feature_importance(model, X_test, numeric_cols, categorical_cols)
    # 5. Create work list
    work_list, risk_df = create_work_list(model, X_test, y_test, y_pred_proba, budget_joints=50)
    # 6. Business value analysis
    analyze_business_value(work_list, risk_df, y_test)
    logger.info("=== Pipeline complete! ===")
    return {
        "model": model,
        "work_list": work_list,
        "metrics": metrics,
        "importance": importance_df,
    }


if __name__ == "__main__":
    results = main()
