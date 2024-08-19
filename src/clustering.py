import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import numpy as np
import joblib

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

def plot_centroids(kmeans, feature_names=None):
    """
    Plot the centroids of the clusters.
    
    Args:
    - kmeans (KMeans): The trained KMeans model with centroids.
    - feature_names (list): Optional list of feature names for labeling the x-axis.
    """
    centroids = kmeans.cluster_centers_
    
    plt.figure(figsize=(10, 6))
    for i, centroid in enumerate(centroids):
        plt.plot(centroid, label=f'Cluster {i} Centroid')
    
    plt.title('Cluster Centroids')
    plt.xlabel('Feature Index' if feature_names is None else 'Features')
    plt.ylabel('Centroid Value')
    plt.legend()
    plt.show()

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
            if sell_price > buy_price:
                revenue = (sell_price - buy_price) / buy_price * 100  # Revenue as a percentage
                revenues.append(revenue)
    
    return np.array(revenues)

def analyze_revenue_per_cluster(raw_data, labels, n_clusters):
    """
    Analyze the revenue for each cluster by simulating buy-sell trades within each window.

    Args:
    - raw_data (pd.DataFrame): DataFrame containing the raw data windows.
    - labels (pd.Series): Cluster labels for each window.
    - n_clusters (int): Number of clusters.
    """
    revenue_stats = []

    for cluster_id in range(n_clusters):
        cluster_windows = raw_data[labels == cluster_id]
        
        all_revenues = []
        for window in cluster_windows.values:
            revenues = calculate_revenue(window)
            all_revenues.extend(revenues)
        
        # Calculate statistics for the cluster
        mean_revenue = np.mean(all_revenues)
        median_revenue = np.median(all_revenues)
        prob_positive_revenue = np.mean(np.array(all_revenues) > 0)
        
        revenue_stats.append({
            'cluster': cluster_id,
            'mean_revenue': mean_revenue,
            'median_revenue': median_revenue,
            'prob_positive_revenue': prob_positive_revenue * 100
        })

    # Display results for each cluster
    for stats in revenue_stats:
        print(f"Cluster {stats['cluster']}:")
        print(f"  Mean Revenue: {stats['mean_revenue']:.2f}%")
        print(f"  Median Revenue: {stats['median_revenue']:.2f}%")
        print(f"  Probability of Positive Revenue: {stats['prob_positive_revenue']:.2f}%")
        print("")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Clustering, centroid plotting, and revenue analysis per cluster.')
    parser.add_argument('--features_csv', type=str, required=True, help='Input CSV file with processed features')
    parser.add_argument('--raw_data_csv', type=str, required=True, help='Input CSV file with raw data windows')
    parser.add_argument('--n_clusters', type=int, default=3, help='Number of clusters (default: 3)')
    parser.add_argument('--save_model', action='store_true', help='Save the trained clustering model')
    parser.add_argument('--model_file', type=str, default='kmeans_model.pkl', help='Path to save the clustering model (default: kmeans_model.pkl)')
    
    args = parser.parse_args()

    # Load the processed features and raw data using pandas
    features = pd.read_csv(args.features_csv, header=None)
    raw_data_windows = pd.read_csv(args.raw_data_csv, header=None)
    
    # Perform clustering
    kmeans, labels = perform_clustering(features, n_clusters=args.n_clusters)
    
    # Convert labels to a pandas Series for easier handling
    labels = pd.Series(labels)
    
    # Analyze revenue per cluster
    analyze_revenue_per_cluster(raw_data_windows, labels, args.n_clusters)
    
    # Save the trained model if required
    if args.save_model:
        joblib.dump(kmeans, args.model_file)
        print(f"Clustering model saved to {args.model_file}")
    
    # Plot the centroids of the clusters
    plot_centroids(kmeans)

if __name__ == "__main__":
    main()
