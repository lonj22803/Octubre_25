"""
Example demonstrating the solution to the PDF weight issue.
Shows how center influence is increased compared to peripheral clusters.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Objets.City.City import CityBase


def create_sample_clusters():
    """Create sample cluster data for demonstration"""
    # Simulate a city center at (40.4, -3.7)
    center = (40.4, -3.7)
    
    # Create 3 clusters: one near center, two peripheral
    clusters_info = {
        0: {  # Cluster near center
            'mean': (40.41, -3.71),
            'cov': np.array([[0.0001, 0], [0, 0.0001]]),
            'weight': 0.3,  # 30% of points
            'size': 30
        },
        1: {  # Peripheral cluster with many points
            'mean': (40.45, -3.65),
            'cov': np.array([[0.0002, 0], [0, 0.0002]]),
            'weight': 0.5,  # 50% of points - dominates by size
            'size': 50
        },
        2: {  # Another peripheral cluster
            'mean': (40.38, -3.75),
            'cov': np.array([[0.0001, 0], [0, 0.0001]]),
            'weight': 0.2,  # 20% of points
            'size': 20
        }
    }
    
    return clusters_info, center


def compare_weight_strategies():
    """Compare cluster weights with and without center boost"""
    clusters_info, center = create_sample_clusters()
    
    print("=" * 70)
    print("COMPARISON: Cluster Weights With and Without Center Influence")
    print("=" * 70)
    print(f"\nCity Center: {center}")
    print("\nCluster Information:")
    for cluster_id, info in clusters_info.items():
        dist_to_center = np.linalg.norm(
            np.array(info['mean']) - np.array(center)
        )
        print(f"  Cluster {cluster_id}:")
        print(f"    - Mean: {info['mean']}")
        print(f"    - Size: {info['size']} points")
        print(f"    - Base weight: {info['weight']:.3f}")
        print(f"    - Distance to center: {dist_to_center:.4f}")
    
    # Strategy 1: No center boost (original problem)
    print("\n" + "-" * 70)
    print("Strategy 1: NO CENTER BOOST (Original Problem)")
    print("-" * 70)
    weights_no_boost = CityBase.calculate_cluster_weights(
        clusters_info,
        center,
        center_weight_multiplier=1.0  # No boost
    )
    print("Resulting weights:")
    for cluster_id, weight in weights_no_boost.items():
        print(f"  Cluster {cluster_id}: {weight:.3f} ({weight*100:.1f}%)")
    print("\n⚠️  Problem: Cluster 1 dominates with 50% despite being far from center!")
    
    # Strategy 2: With center boost (solution)
    print("\n" + "-" * 70)
    print("Strategy 2: WITH CENTER BOOST (Solution)")
    print("-" * 70)
    weights_with_boost = CityBase.calculate_cluster_weights(
        clusters_info,
        center,
        center_weight_multiplier=2.5  # 2.5x boost for center
    )
    print("Resulting weights:")
    for cluster_id, weight in weights_with_boost.items():
        print(f"  Cluster {cluster_id}: {weight:.3f} ({weight*100:.1f}%)")
    print("\n✓ Solution: Cluster 0 (near center) now dominates with boosted weight!")
    
    return weights_no_boost, weights_with_boost, clusters_info, center


def visualize_pdf_comparison():
    """Visualize PDF values with and without center influence"""
    clusters_info, center = create_sample_clusters()
    
    # Create a grid of points
    lat_range = np.linspace(40.35, 40.50, 50)
    lon_range = np.linspace(-3.80, -3.60, 50)
    lat_grid, lon_grid = np.meshgrid(lat_range, lon_range)
    grid_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    
    # Calculate PDF without center boost
    pdf_no_boost = CityBase.calculate_pdf_values(
        clusters_info,
        grid_points,
        center,
        center_influence_factor=1.0  # No boost
    )
    pdf_no_boost = pdf_no_boost.reshape(lat_grid.shape)
    
    # Calculate PDF with center boost
    pdf_with_boost = CityBase.calculate_pdf_values(
        clusters_info,
        grid_points,
        center,
        center_influence_factor=3.0  # 3x boost for center
    )
    pdf_with_boost = pdf_with_boost.reshape(lat_grid.shape)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 6))
    
    # Plot 1: No center boost
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(lon_grid, lat_grid, pdf_no_boost, cmap='viridis', alpha=0.8)
    ax1.scatter([center[1]], [center[0]], [np.max(pdf_no_boost)], 
               color='red', s=100, label='City Center')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_zlabel('PDF Value')
    ax1.set_title('PDF WITHOUT Center Boost\n(Original Problem)')
    ax1.legend()
    
    # Plot 2: With center boost
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(lon_grid, lat_grid, pdf_with_boost, cmap='viridis', alpha=0.8)
    ax2.scatter([center[1]], [center[0]], [np.max(pdf_with_boost)], 
               color='red', s=100, label='City Center')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_zlabel('PDF Value')
    ax2.set_title('PDF WITH Center Boost\n(Solution - 3x factor)')
    ax2.legend()
    
    # Plot 3: Difference
    ax3 = fig.add_subplot(133, projection='3d')
    diff = pdf_with_boost - pdf_no_boost
    ax3.plot_surface(lon_grid, lat_grid, diff, cmap='RdBu', alpha=0.8)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_zlabel('Difference')
    ax3.set_title('Difference\n(Solution - Original)')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'Objets', 'pdf_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Show the plot (may not display in headless environment)
    try:
        plt.show()
    except:
        pass


def demonstrate_entity_distribution():
    """Show how entities are distributed with different strategies"""
    clusters_info, center = create_sample_clusters()
    
    # Create sample stations based on cluster centers
    stations = ['Station_A', 'Station_B', 'Station_C']
    station_coords = [
        clusters_info[0]['mean'],
        clusters_info[1]['mean'],
        clusters_info[2]['mean']
    ]
    
    print("\n" + "=" * 70)
    print("ENTITY DISTRIBUTION COMPARISON")
    print("=" * 70)
    print(f"\nTotal entities to distribute: 100")
    
    # Without center boost
    weights_no_boost = CityBase.calculate_cluster_weights(
        clusters_info, center, center_weight_multiplier=1.0
    )
    station_weights_no_boost = CityBase.assign_weights_to_entities(
        stations, station_coords, clusters_info, weights_no_boost
    )
    
    print("\nWithout center boost:")
    for station, weight in zip(stations, station_weights_no_boost):
        count = int(weight * 100)
        print(f"  {station}: {count} entities ({weight*100:.1f}%)")
    
    # With center boost
    weights_with_boost = CityBase.calculate_cluster_weights(
        clusters_info, center, center_weight_multiplier=2.5
    )
    station_weights_with_boost = CityBase.assign_weights_to_entities(
        stations, station_coords, clusters_info, weights_with_boost
    )
    
    print("\nWith center boost (2.5x):")
    for station, weight in zip(stations, station_weights_with_boost):
        count = int(weight * 100)
        print(f"  {station}: {count} entities ({weight*100:.1f}%)")
    
    print("\n✓ Notice how Station_A (near center) gets more entities with the boost!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Solution to PDF Weight Issue")
    print("Lines 453 and 474: center_pdf * center_weight vs cluster_pdf * cluster_weight")
    print("=" * 70)
    
    # Run demonstrations
    compare_weight_strategies()
    demonstrate_entity_distribution()
    
    print("\n" + "=" * 70)
    print("Creating 3D visualization...")
    print("=" * 70)
    visualize_pdf_comparison()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The solution to the PDF weight issue involves TWO key parameters:

1. center_weight_multiplier (in calculate_cluster_weights):
   - Multiplies the weight of the cluster closest to city center
   - Recommended range: 1.5 - 5.0
   - Default: 2.0

2. center_influence_factor (in calculate_pdf_values):
   - Boosts the PDF contribution of the city center
   - Recommended range: 2.0 - 10.0
   - Default: 3.0
   - THIS DIRECTLY ADDRESSES LINES 453 AND 474

Usage in classes:
- Hotels: center_weight_multiplier=2.5
- Restaurants: center_weight_multiplier=3.0
- TouristPlace: center_weight_multiplier=4.0

The architecture is now generalized and reusable across all entity types!
    """)
    print("=" * 70)
