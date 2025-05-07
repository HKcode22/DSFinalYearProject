import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from sklearn.calibration import calibration_curve

def load_anomaly_results(results_dir=None):
    """
    Load anomaly detection results from the specified directory.
    
    Args:
        results_dir (str): Directory containing anomaly detection results
        
    Returns:
        tuple: (anomaly_report, anomaly_results, anomaly_summary)
    """
    if results_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, 'output', 'integrated_analysis')
    
    try:
        # Load anomaly report
        report_path = os.path.join(results_dir, 'anomaly_report.csv')
        anomaly_report = pd.read_csv(report_path)
        
        # Load anomaly results
        results_path = os.path.join(results_dir, 'anomaly_results.json')
        with open(results_path, 'r') as f:
            anomaly_results = json.load(f)
            
        # Load anomaly summary
        summary_path = os.path.join(results_dir, 'anomaly_summary.txt')
        with open(summary_path, 'r') as f:
            anomaly_summary = f.read()
            
        # Load integrated results if available
        integrated_path = os.path.join(results_dir, 'integrated_results.csv')
        if os.path.exists(integrated_path):
            integrated_results = pd.read_csv(integrated_path)
        else:
            integrated_results = None
            
        return anomaly_report, anomaly_results, anomaly_summary, integrated_results
        
    except Exception as e:
        print(f"Error loading anomaly results: {str(e)}")
        return None, None, None, None

def plot_anomaly_severity_distribution(anomaly_data, ax=None):
    """
    Plot the distribution of anomaly severity scores.
    
    Args:
        anomaly_data (DataFrame): Anomaly data with severity scores
        ax (Axes, optional): Matplotlib axes for plotting
        
    Returns:
        Axes: The plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check if data has anomaly severity
    if 'anomaly_severity' not in anomaly_data.columns:
        ax.text(0.5, 0.5, 'No anomaly severity data available', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        return ax
    
    # All companies
    sns.histplot(anomaly_data['anomaly_severity'], 
                 kde=True, 
                 color='blue',
                 alpha=0.6,
                 label='All Data',
                 ax=ax)
    
    # Highlight anomalies if is_anomaly field exists
    if 'is_anomaly' in anomaly_data.columns:
        anomalies = anomaly_data[anomaly_data['is_anomaly'] == True]
        if len(anomalies) > 0:
            sns.histplot(anomalies['anomaly_severity'], 
                         kde=True, 
                         color='red',
                         alpha=0.6,
                         label='Anomalies',
                         ax=ax)
    
    ax.set_title('Distribution of Anomaly Severity Scores')
    ax.set_xlabel('Anomaly Severity (higher = more anomalous)')
    ax.set_ylabel('Count')
    ax.legend()
    
    return ax

def plot_funding_vs_severity(anomaly_data, ax=None):
    """
    Plot funding amount vs anomaly severity.
    
    Args:
        anomaly_data (DataFrame): Anomaly data
        ax (Axes, optional): Matplotlib axes for plotting
        
    Returns:
        Axes: The plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check if required columns exist
    if 'anomaly_severity' not in anomaly_data.columns:
        ax.text(0.5, 0.5, 'No anomaly severity data available', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        return ax
    
    # Ensure funding_amount_numeric is available
    if 'funding_amount_numeric' not in anomaly_data.columns and 'funding_amount' in anomaly_data.columns:
        # Simple conversion for visualization
        anomaly_data['funding_amount_numeric'] = pd.to_numeric(
            anomaly_data['funding_amount'].str.replace('$', '').str.replace(',', ''), 
            errors='coerce'
        )
    
    # Filter out NaN values
    plot_data = anomaly_data.dropna(subset=['funding_amount_numeric', 'anomaly_severity'])
    
    # Determine colors based on is_anomaly field if available
    if 'is_anomaly' in plot_data.columns:
        colors = plot_data['is_anomaly'].map({True: 'red', False: 'blue'})
    else:
        # Use anomaly_severity threshold as a substitute
        threshold = plot_data['anomaly_severity'].quantile(0.95)  # Top 5% as anomalies
        colors = (plot_data['anomaly_severity'] > threshold).map({True: 'red', False: 'blue'})
    
    # Create scatter plot
    scatter = ax.scatter(
        plot_data['funding_amount_numeric'],
        plot_data['anomaly_severity'],
        c=colors,
        alpha=0.7,
        s=50
    )
    
    # Add company names as annotations for anomalies
    if 'name' in plot_data.columns:
        # Annotate high severity companies
        high_severity = plot_data[plot_data['anomaly_severity'] > threshold] if 'is_anomaly' not in plot_data.columns else plot_data[plot_data['is_anomaly'] == True]
        for _, row in high_severity.iterrows():
            ax.annotate(
                row['name'],
                xy=(row['funding_amount_numeric'], row['anomaly_severity']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
    
    # Add custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Anomaly/High Severity'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Normal')
    ]
    ax.legend(handles=legend_elements)
    
    ax.set_title('Funding Amount vs. Anomaly Severity')
    ax.set_xlabel('Funding Amount ($)')
    ax.set_ylabel('Anomaly Severity (higher = more anomalous)')
    ax.set_xscale('log')
    
    return ax

def plot_anomaly_types(anomaly_report, ax=None):
    """
    Plot the distribution of anomaly types.
    
    Args:
        anomaly_report (DataFrame): Anomaly report with types
        ax (Axes, optional): Matplotlib axes for plotting
        
    Returns:
        Axes: The plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'anomaly_type' not in anomaly_report.columns:
        ax.text(0.5, 0.5, 'No anomaly type data available', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        return ax
    
    # Extract primary anomaly type (first in list if multiple)
    anomaly_report['primary_type'] = anomaly_report['anomaly_type'].str.split(', ').str[0]
    
    # Count anomalies by type
    type_counts = anomaly_report['primary_type'].value_counts()
    
    # Plot
    type_counts.plot(kind='bar', ax=ax, color='coral')
    
    ax.set_title('Distribution of Anomaly Types')
    ax.set_xlabel('Anomaly Type')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    return ax

def plot_funding_stage_vs_anomalies(integrated_results, ax=None):
    """
    Plot relationship between funding stage and anomalies.
    
    Args:
        integrated_results (DataFrame): Integrated results
        ax (Axes, optional): Matplotlib axes for plotting
        
    Returns:
        Axes: The plot axes
    """
    if integrated_results is None or 'predicted_stage' not in integrated_results.columns:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No integrated stage prediction data available', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        return ax
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate anomaly severity by funding stage
    stage_severity = integrated_results.groupby('predicted_stage')['anomaly_severity'].agg(['mean', 'count']).reset_index()
    stage_severity = stage_severity.sort_values('mean', ascending=False)
    
    # Create plot
    bars = ax.bar(stage_severity['predicted_stage'], stage_severity['mean'], alpha=0.7)
    
    # Add count labels
    for bar, count in zip(bars, stage_severity['count']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'n={count}', ha='center', va='bottom', fontsize=8)
    
    ax.set_title('Average Anomaly Severity by Funding Stage')
    ax.set_xlabel('Funding Stage')
    ax.set_ylabel('Average Anomaly Severity')
    ax.set_ylim(0, max(stage_severity['mean']) * 1.2)  # Add some space for labels
    
    return ax

def plot_success_prediction(integrated_results, ax=None):
    """
    Plot success prediction distribution.
    
    Args:
        integrated_results (DataFrame): Integrated results
        ax (Axes, optional): Matplotlib axes for plotting
        
    Returns:
        Axes: The plot axes
    """
    if integrated_results is None or 'predicted_success' not in integrated_results.columns:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No integrated success prediction data available', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        return ax
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Count success predictions
    success_counts = integrated_results['predicted_success'].value_counts()
    
    # Create pie chart
    ax.pie(
        success_counts, 
        labels=['Success', 'Failure'], 
        autopct='%1.1f%%',
        colors=['#5cb85c', '#d9534f'],
        explode=[0, 0.1],
        shadow=True,
        startangle=90
    )
    
    ax.set_title('Predicted Success/Failure Distribution')
    
    return ax

def plot_success_score_distribution(integrated_results, ax=None):
    """
    Plot distribution of success scores.
    
    Args:
        integrated_results (DataFrame): Integrated results
        ax (Axes, optional): Matplotlib axes for plotting
        
    Returns:
        Axes: The plot axes
    """
    if integrated_results is None or 'success_score' not in integrated_results.columns:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No integrated success score data available', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        return ax
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    sns.histplot(
        integrated_results['success_score'],
        kde=True,
        ax=ax
    )
    
    # Add vertical line at threshold
    ax.axvline(x=50, color='red', linestyle='--', label='Success Threshold')
    
    ax.set_title('Distribution of Success Scores')
    ax.set_xlabel('Success Score')
    ax.set_ylabel('Count')
    ax.legend()
    
    return ax

def plot_calibration_curve(integrated_results, ax=None):
    """
    Plot calibration curve for anomaly detection predictions.
    
    Args:
        integrated_results (DataFrame): Integrated results with predicted probabilities
        ax (Axes, optional): Matplotlib axes for plotting
        
    Returns:
        Axes: The plot axes
    """
    if integrated_results is None or 'anomaly_severity' not in integrated_results.columns:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No anomaly severity data available', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        return ax
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert anomaly_severity to probability-like scale if not already
    # Typically, higher severity = higher probability of being anomaly
    if 'is_anomaly' in integrated_results.columns:
        # Normalize severity scores to [0, 1] range for calibration
        prob_scores = integrated_results['anomaly_severity'] / integrated_results['anomaly_severity'].max()
        y_true = integrated_results['is_anomaly'].astype(int)
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_true, prob_scores, n_bins=10, strategy='uniform')
        
        # Plot calibration curve
        ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration curve')
        
        # Plot perfect calibration reference line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        # Calculate and display Brier score
        from sklearn.metrics import brier_score_loss
        brier_score = brier_score_loss(y_true, prob_scores)
        ax.text(0.05, 0.95, f'Brier score: {brier_score:.3f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title('Calibration Plot (Reliability Curve)')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives (Empirical probability)')
        ax.legend(loc='best')
        ax.grid(True)
    else:
        # If no is_anomaly field exists, use a threshold to create binary labels
        threshold = integrated_results['anomaly_severity'].quantile(0.95)  # Top 5% as anomalies
        y_true = (integrated_results['anomaly_severity'] > threshold).astype(int)
        prob_scores = integrated_results['anomaly_severity'] / integrated_results['anomaly_severity'].max()
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_true, prob_scores, n_bins=10, strategy='uniform')
        
        # Plot calibration curve
        ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration curve')
        
        # Plot perfect calibration reference line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        ax.set_title('Calibration Plot (Based on threshold)')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives (Empirical probability)')
        ax.legend(loc='best')
        ax.grid(True)
        ax.text(0.05, 0.95, f'Using {int(threshold*100)/100} threshold',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return ax

def main():
    # Set up matplotlib for better visualization
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Load anomaly results
    base_dir = os.path.dirname(os.path.abspath(__file__))
    integrated_dir = os.path.join(base_dir, 'output', 'integrated_analysis')
    anomaly_dir = os.path.join(base_dir, 'output', 'anomaly_detection')
    
    # Try integrated results first, fall back to anomaly_detection
    if os.path.exists(os.path.join(integrated_dir, 'anomaly_report.csv')):
        results_dir = integrated_dir
    else:
        results_dir = anomaly_dir
    
    print(f"Loading results from: {results_dir}")
    
    # Create output directory for new visualizations
    viz_dir = os.path.join(base_dir, 'output', 'outputAnomalyDetection')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load results
    anomaly_report, anomaly_results, anomaly_summary, integrated_results = load_anomaly_results(results_dir)
    
    if anomaly_report is None:
        print("Error: Could not load anomaly results.")
        return
    
    # Print summary information
    print("\nANOMALY DETECTION SUMMARY")
    print("=======================")
    print(anomaly_summary)
    
    # Prepare anomaly data from the report data
    anomaly_data = anomaly_report.copy()
    anomaly_data['is_anomaly'] = True  # All rows in the report are anomalies
    
    # Create comprehensive visualization dashboard
    fig = plt.figure(figsize=(20, 24))  # Made taller to accommodate new plots
    
    # Create a 4x2 grid of subplots (one more row for calibration plot)
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Anomaly Severity Distribution 
    ax1 = fig.add_subplot(gs[0, 0])
    # Use the anomaly report which contains just the anomalies
    plot_anomaly_severity_distribution(pd.DataFrame(anomaly_results['anomalies']), ax1)
    
    # Plot 2: Funding vs Severity
    ax2 = fig.add_subplot(gs[0, 1])
    plot_funding_vs_severity(pd.DataFrame(anomaly_results['anomalies']), ax2)
    
    # Plot 3: Anomaly Types
    ax3 = fig.add_subplot(gs[1, 0])
    plot_anomaly_types(anomaly_report, ax3)
    
    # Plot 4: Funding Stage vs Anomalies
    ax4 = fig.add_subplot(gs[1, 1])
    plot_funding_stage_vs_anomalies(integrated_results, ax4)
    
    # Plot 5: Success Prediction
    ax5 = fig.add_subplot(gs[2, 0])
    plot_success_prediction(integrated_results, ax5)
    
    # Plot 6: Success Score Distribution
    ax6 = fig.add_subplot(gs[2, 1])
    plot_success_score_distribution(integrated_results, ax6)
    
    # Plot 7: Calibration Plot
    ax7 = fig.add_subplot(gs[3, 0])
    plot_calibration_curve(integrated_results, ax7)
    
    # Additional plots for model performance metrics
    # Check if performance metrics exist and copy them to outputAnomalyDetection
    model_calib_path = os.path.join(results_dir, 'model_calibration.png')
    perf_metrics_path = os.path.join(results_dir, 'performance_metrics.png')
    conf_matrix_path = os.path.join(results_dir, 'confusion_matrix.png')
    
    # Copy performance visualizations if they exist
    for src_path, dst_name in [
        (model_calib_path, 'model_calibration.png'),
        (perf_metrics_path, 'performance_metrics.png'),
        (conf_matrix_path, 'confusion_matrix.png')
    ]:
        if os.path.exists(src_path):
            # Use matplotlib to load and save the image to avoid direct file copy
            try:
                img = plt.imread(src_path)
                plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.savefig(os.path.join(viz_dir, dst_name), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Copied {dst_name} to output directory")
            except Exception as e:
                print(f"Error copying {dst_name}: {str(e)}")
    
    # Try to load performance metrics JSON
    perf_metrics_json = os.path.join(results_dir, 'model_performance.json')
    if os.path.exists(perf_metrics_json):
        try:
            with open(perf_metrics_json, 'r') as f:
                metrics = json.load(f)
                print("\nMODEL PERFORMANCE METRICS")
                print("========================")
                for metric, value in metrics.items():
                    print(f"{metric.upper()}: {value:.4f}")
                print()
        except Exception as e:
            print(f"Error loading performance metrics: {str(e)}")

    # Add title
    fig.suptitle('Startup Funding Anomaly Detection & Success Prediction Analysis', fontsize=16)
    
    # Save the dashboard
    dashboard_path = os.path.join(viz_dir, 'anomaly_detection_dashboard.png')
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    
    print(f"\nDashboard saved to: {dashboard_path}")
    
    # Also create individual plots
    
    # Anomaly Severity Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_anomaly_severity_distribution(pd.DataFrame(anomaly_results['anomalies']), ax)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'anomaly_severity_distribution.png'), dpi=300)
    
    # Funding vs Severity
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_funding_vs_severity(pd.DataFrame(anomaly_results['anomalies']), ax)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'funding_vs_severity.png'), dpi=300)
    
    # Anomaly Types
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_anomaly_types(anomaly_report, ax)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'anomaly_types.png'), dpi=300)

    # If integrated results are available, create additional plots
    if integrated_results is not None:
        # Funding Stage vs Anomalies
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_funding_stage_vs_anomalies(integrated_results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'funding_stage_vs_anomalies.png'), dpi=300)
        
        # Success Prediction
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_success_prediction(integrated_results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'success_prediction.png'), dpi=300)
        
        # Success Score Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_success_score_distribution(integrated_results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'success_score_distribution.png'), dpi=300)
        
        # Calibration Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_calibration_curve(integrated_results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'calibration_plot.png'), dpi=300)
    
    print(f"Individual plots saved to: {viz_dir}")
    
    # Create top anomalies table
    top_anomalies = anomaly_report.sort_values('anomaly_severity', ascending=False).head(10)
    
    print("\nTOP 10 ANOMALIES")
    print("===============")
    for i, row in top_anomalies.iterrows():
        print(f"{i+1}. {row['name']} - {row['funding_type']} - ${row['funding_amount']} - Severity: {row['anomaly_severity']:.4f}")
        if 'anomaly_type' in row and not pd.isna(row['anomaly_type']):
            print(f"   Type: {row['anomaly_type']}")
        print(f"   Industry: {row['industry']}")
        print()
    
    print(f"Visualization complete. All plots saved to {viz_dir}")

if __name__ == "__main__":
    main() 