#!/usr/bin/env python3
import signalplot

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

"""
Generate visualizations for Blog 12: Corrosion Risk Ranking
"""

import numpy as np
import matplotlib.pyplot as plt
import importlib.util



# Import production module
spec = importlib.util.spec_from_file_location(
    "corrosion_risk_production",
    "/Users/k.jones/Desktop/blogs/blog_posts/12_corrosion_risk_production.py"
)
production_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production_module)

# Import functions
generate_pipeline_corrosion_data = production_module.generate_pipeline_corrosion_data
prepare_features = production_module.prepare_features
train_corrosion_risk_model = production_module.train_corrosion_risk_model
create_work_list = production_module.create_work_list

def create_risk_visualizations(risk_df, work_list, metrics, plot: bool = False):
    """
    Generate comprehensive risk analysis visualizations.
    """
    if plot:
        plt.figure(figsize=(12, 10))
    
    # Panel 1: Risk score distribution
        ax1 = plt.subplot(2, 2, 1)
        ax1.hist(risk_df['risk_score'], bins=50, color='white', edgecolor='black', linewidth=1.5)
        ax1.axvline(x=metrics['optimal_threshold'], color='gray', linestyle='--', linewidth=2,
                    label=f"Optimal Threshold ({metrics['optimal_threshold']:.3f})")
    
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_position(('outward', 5))
        ax1.spines['bottom'].set_position(('outward', 5))
        ax1.set_title('Corrosion Risk Score Distribution', fontsize=12, fontweight='bold', loc='left')
        ax1.set_xlabel('Risk Score', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.legend(frameon=False, fontsize=9)
    
    # Panel 2: Risk vs Metal Loss
        ax2 = plt.subplot(2, 2, 2)
    
    # Separate failures and non-failures
        failures = risk_df[risk_df['actual_failure'] == 1]
        non_failures = risk_df[risk_df['actual_failure'] == 0]
    
        ax2.scatter(non_failures['ili_metal_loss'], non_failures['risk_score'], 
                    c='white', s=20, edgecolors='gray', linewidths=0.5, alpha=0.3, label='No Failure')
        ax2.scatter(failures['ili_metal_loss'], failures['risk_score'], 
                    c='black', s=40, marker='X', linewidths=1.5, label='Actual Failure')
    
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_position(('outward', 5))
        ax2.spines['bottom'].set_position(('outward', 5))
        ax2.set_title('Risk Score vs ILI Metal Loss', fontsize=12, fontweight='bold', loc='left')
        ax2.set_xlabel('ILI Metal Loss (%)', fontsize=10)
        ax2.set_ylabel('Predicted Risk Score', fontsize=10)
        ax2.legend(frameon=False, fontsize=9, loc='lower right')
    
    # Panel 3: CP Potential vs Risk (by coating type)
        ax3 = plt.subplot(2, 2, 3)
    
        coating_colors = {'FBE': 'white', 'PE': 'lightgray', 'CoalTar': 'gray', 'Tape': 'black'}
    
        for coating in ['FBE', 'PE', 'CoalTar', 'Tape']:
            coating_data = risk_df[risk_df['coating'] == coating]
            ax3.scatter(coating_data['cp_potential'], coating_data['risk_score'],
                       c=coating_colors[coating], s=15, edgecolors='black', 
                       linewidths=0.5, alpha=0.5, label=coating)
    
        ax3.axvline(x=-0.85, color='gray', linestyle='--', linewidth=1.5, 
                    label='NACE Criterion')
    
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['left'].set_position(('outward', 5))
        ax3.spines['bottom'].set_position(('outward', 5))
        ax3.set_title('CP Potential vs Risk by Coating', fontsize=12, fontweight='bold', loc='left')
        ax3.set_xlabel('CP Potential (V vs Cu/CuSO4)', fontsize=10)
        ax3.set_ylabel('Predicted Risk Score', fontsize=10)
        ax3.legend(frameon=False, fontsize=8, loc='upper left')
    
    # Panel 4: Work List Value
        ax4 = plt.subplot(2, 2, 4)
    
        top_50_sorted = work_list.sort_values('value_per_dollar', ascending=True)
        y_pos = np.arange(len(top_50_sorted))
    
        ax4.barh(y_pos[::5], top_50_sorted['value_per_dollar'].values[::5], 
                        color='white', edgecolor='black', linewidth=1.5)
    
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_position(('outward', 5))
        ax4.spines['bottom'].set_position(('outward', 5))
        ax4.set_title('Work List: Value per Dollar (Every 5th Joint)', fontsize=12, 
                      fontweight='bold', loc='left')
        ax4.set_xlabel('Value per Dollar Spent', fontsize=10)
        ax4.set_ylabel('Joint Rank', fontsize=10)
        ax4.set_yticks([])
    
        plt.tight_layout()
        plt.savefig('/Users/k.jones/Desktop/blogs/blog_posts/12_corrosion_risk_main.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("✓ Created: 12_corrosion_risk_main.png")

def main():
    """Generate all visualizations for blog 12."""
    signalplot.apply(font_family='serif')
    logger.info("BLOG 12 VISUALIZATION GENERATION")
    logger.info()
    
    # Generate data and train model
    df = generate_pipeline_corrosion_data(n_joints=5000, random_seed=42)
    X, y, numeric_cols, categorical_cols = prepare_features(df)
    model, X_test, y_test, y_pred_proba, metrics = train_corrosion_risk_model(
        X, y, numeric_cols, categorical_cols
    )
    work_list, risk_df = create_work_list(model, X_test, y_test, y_pred_proba, budget_joints=50)
    
    # Create visualizations
    create_risk_visualizations(risk_df, work_list, metrics)
    
    logger.info()
    logger.info("Visualization generation complete!")

if __name__ == "__main__":
    main()

