import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import numpy as np
import joblib
import os

def perform_clustering(features, n_clusters=3):
    """
    Perform KMeans clustering on the provided feature data.
    
    Args:
    - features (pd.DataFrame): The feature data used for clustering.
    - n_clusters (int): Number of clusters.
    
    Returns:
    - kmeans (KMeans): The trained KMeans model.
    - labels (np.array): The cluster labels for each sample.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    
    return kmeans, labels

def calculate_revenue(window):
    """
    Calculate revenue percentages for all buy-sell combinations in a window.

    Args:
    - window (np.array): A single window of raw stock prices.
    
    Returns:
    - revenues (np.array): Revenue percentages for each buy-sell combination in the window.
    """
    revenues = []
    for buy_idx in range(len(window) - 1):
        for sell_idx in range(buy_idx + 1, len(window)):
            buy_price = window[buy_idx]
            sell_price = window[sell_idx]
            revenue = (sell_price - buy_price) / buy_price * 100  # Revenue as a percentage
            revenues.append(revenue)
    
    return np.array(revenues)

def analyze_revenue_per_cluster(raw_data, labels, n_clusters):
    """
    Analyze the revenue for each cluster by simulating buy-sell trades within each window.
    Calculate the fraction of points in each window where a specific revenue percentage can be achieved.
    
    Args:
    - raw_data (pd.DataFrame): DataFrame containing the raw data windows.
    - labels (pd.Series): Cluster labels for each window.
    - n_clusters (int): Number of clusters.
    
    Returns:
    - df (pd.DataFrame): A DataFrame containing the analysis results per cluster.
    - buy_points_distribution (dict): Dictionary containing the number of valid buying points per cluster and threshold.
    """
    revenue_thresholds = range(1, 11)  # From 1% to 10% revenue
    results = []
    buy_points_distribution = {cluster_id: {threshold: [] for threshold in revenue_thresholds} for cluster_id in range(n_clusters)}

    for cluster_id in range(n_clusters):
        cluster_windows = raw_data[labels == cluster_id]
        num_weeks = len(cluster_windows)  # Number of weeks (windows) in the cluster
        
        # Initialize metrics storage for each revenue threshold
        threshold_metrics = {threshold: {'point_fraction': [], 'sell_point_fraction': []} for threshold in revenue_thresholds}
        
        for window in cluster_windows.values:
            window_size = len(window)
            
            # Iterate over each threshold (1% to 10%)
            for threshold in revenue_thresholds:
                valid_buy_points = 0
                
                # Iterate through each point in the window as a potential buy point
                for buy_idx in range(window_size - 1):
                    sell_points_for_buy = 0
                    buy_price = window[buy_idx]
                    
                    # Check if any sell points yield the desired revenue
                    for sell_idx in range(buy_idx + 1, window_size):
                        sell_price = window[sell_idx]
                        revenue = (sell_price - buy_price) / buy_price * 100  # Revenue as a percentage
                        
                        if revenue >= threshold:
                            sell_points_for_buy += 1
                    
                    # If the buy point has valid sell points
                    if sell_points_for_buy > 0:
                        valid_buy_points += 1
                
                # Store the number of valid buy points for this cluster and threshold
                buy_points_distribution[cluster_id][threshold].append(valid_buy_points)
                
                # Calculate the fraction of valid buy points
                point_fraction = valid_buy_points / window_size if window_size > 0 else 0
                
                # Store the metrics for this window and threshold
                threshold_metrics[threshold]['point_fraction'].append(point_fraction)
        
        # Aggregate the metrics for each threshold across all windows in the cluster
        for threshold in revenue_thresholds:
            mean_point_fraction = np.mean(threshold_metrics[threshold]['point_fraction'])
            
            # Store the results for this cluster and threshold
            results.append({
                'cluster': cluster_id,
                'num_weeks': num_weeks,
                'threshold': threshold,
                'mean_point_fraction': mean_point_fraction
            })

    # Convert the results to a pandas DataFrame
    df = pd.DataFrame(results)
    return df, buy_points_distribution

def plot_and_save_histograms(buy_points_distribution, output_dir):
    """
    Plot and save histograms for the number of buying points per threshold for each cluster.
    
    Args:
    - buy_points_distribution (dict): Dictionary containing the number of valid buying points per cluster and threshold.
    - output_dir (str): Directory to save the histogram plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for cluster_id, thresholds in buy_points_distribution.items():
        for threshold, buy_points in thresholds.items():
            plt.figure(figsize=(10, 6))
            plt.hist(buy_points, bins=20, alpha=0.7, color='blue', edgecolor='black')
            plt.title(f'Cluster {cluster_id} - Revenue Threshold {threshold}%: Buy Points Distribution')
            plt.xlabel('Number of Valid Buy Points')
            plt.ylabel('Frequency')
            
            # Save the plot as a file
            file_name = f"cluster_{cluster_id}_threshold_{threshold}_histogram.png"
            plt.savefig(os.path.join(output_dir, file_name))
            plt.close()  # Close the plot to prevent overlap in the next iteration

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Clustering, revenue percentage analysis per cluster, and histogram plotting.')
    parser.add_argument('--features_csv', type=str, required=True, help='Input CSV file with processed features')
    parser.add_argument('--raw_data_csv', type=str, required=True, help='Input CSV file with raw data windows')
    parser.add_argument('--n_clusters', type=int, default=3, help='Number of clusters (default: 3)')
    parser.add_argument('--save_model', action='store_true', help='Save the trained clustering model')
    parser.add_argument('--model_file', type=str, default='kmeans_model.pkl', help='Path to save the clustering model (default: kmeans_model.pkl)')
    parser.add_argument('--output_dir', type=str, default='output_histograms', help='Directory to save histogram plots')
    
    args = parser.parse_args()

    # Load the processed features and raw data using pandas
    features = pd.read_csv(args.features_csv, header=None)
    raw_data_windows = pd.read_csv(args.raw_data_csv, header=None)
    
    # Perform clustering
    kmeans, labels = perform_clustering(features, n_clusters=args.n_clusters)
    
    # Convert labels to a pandas Series for easier handling
    labels = pd.Series(labels)
    
    # Analyze revenue per cluster and get the buy points distribution
    analysis_df, buy_points_distribution = analyze_revenue_per_cluster(raw_data_windows, labels, args.n_clusters)
    
    # Display the DataFrame with results
    print(analysis_df)
    
    # Plot and save histograms for each cluster
    plot_and_save_histograms(buy_points_distribution, args.output_dir)
    
    # Save the trained model if required
    if args.save_model:
        joblib.dump(kmeans, args.model_file)
        print(f"Clustering model saved to {args.model_file}")

if __name__ == "__main__":
    main()
