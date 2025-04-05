import heapq
import math
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Implementation a: adj matrix + array
def dijkstra_matrix_array(graph, start):
    n = len(graph)
    dist = [float('inf')] * n
    pi = [None] * n
    visited = [False] * n
    dist[start] = 0
    
    for _ in range(n):
        min_dist = float('inf')
        min_vertex = -1
        
        for v in range(n):
            if not visited[v] and dist[v] < min_dist:
                min_dist = dist[v]
                min_vertex = v
        
        if min_vertex == -1:
            break
        
        visited[min_vertex] = True
        
        for v in range(n):
            if not visited[v] and graph[min_vertex][v] != float('inf'):
                if dist[min_vertex] + graph[min_vertex][v] < dist[v]:
                    pi[v] = min_vertex
                    dist[v] = dist[min_vertex] + graph[min_vertex][v]
    
    return dist

# Implementation b: adj list + heap
def dijkstra_list_heap(graph, start):
    n = len(graph)
    dist = [float('inf')] * n
    pi = [None] * n
    visited = [False] * n
    dist[start] = 0

    priority_queue = [(0, start)]
    
    while priority_queue:
        d_u, u = heapq.heappop(priority_queue)
        
        if visited[u]:
            continue

        visited[u] = True

        for v, weight in graph[u]:
            if not visited[v] and dist[v] > dist[u] + weight:
                dist[v] = dist[u] + weight
                pi[v] = u
                heapq.heappush(priority_queue, (dist[v], v))

    return dist

# Function to generate random graphs with different densities
def generate_random_graph(n, density, weight_range=(1, 10)):
    # Create adjacency matrix
    adj_matrix = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        adj_matrix[i][i] = 0  # Set diagonal to 0
    
    # Create adjacency list
    adj_list = [[] for _ in range(n)]
    
    # Add random edges
    edge_count = 0
    max_edges = n * (n - 1)
    target_edges = int(max_edges * density)
    
    while edge_count < target_edges:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        
        # Skip self-loops and existing edges
        if u == v or adj_matrix[u][v] != float('inf'):
            continue
        
        weight = random.randint(weight_range[0], weight_range[1])
        adj_matrix[u][v] = weight
        adj_list[u].append((v, weight))
        edge_count += 1
    
    return adj_matrix, adj_list

# Find the crossover density for specific vertex sizes
def find_crossover_density(vertex_sizes, density_range, num_trials=3):
    results = {}
    
    for vertices in vertex_sizes:
        crossovers = []
        
        for density in tqdm(density_range, desc=f"Testing V={vertices}"):
            matrix_times = []
            list_times = []
            
            for _ in range(num_trials):
                adj_matrix, adj_list = generate_random_graph(vertices, density)
                start_vertex = random.randint(0, vertices - 1)
                
                # Time matrix implementation
                start_time = time.time()
                dijkstra_matrix_array(adj_matrix, start_vertex)
                end_time = time.time()
                matrix_times.append(end_time - start_time)
                
                # Time list implementation
                start_time = time.time()
                dijkstra_list_heap(adj_list, start_vertex)
                end_time = time.time()
                list_times.append(end_time - start_time)
            
            avg_matrix_time = sum(matrix_times) / num_trials
            avg_list_time = sum(list_times) / num_trials
            
            # If matrix is faster, we've found our crossover
            if avg_matrix_time <= avg_list_time:
                crossovers.append(density)
        
        if crossovers:
            results[vertices] = min(crossovers)  # The minimum density where matrix is better
        else:
            results[vertices] = None  # No crossover found
    
    return results

# Analyze runtime behavior at specific densities
def analyze_runtime_behavior(vertex_sizes, densities):
    results = {
        'matrix': {d: [] for d in densities},
        'list': {d: [] for d in densities}
    }
    
    for density in densities:
        for vertices in tqdm(vertex_sizes, desc=f"Testing density={density}"):
            matrix_times = []
            list_times = []
            
            # Run 3 trials for more stable results
            for _ in range(3):
                adj_matrix, adj_list = generate_random_graph(vertices, density)
                start_vertex = random.randint(0, vertices - 1)
                
                # Time matrix implementation
                start_time = time.time()
                dijkstra_matrix_array(adj_matrix, start_vertex)
                end_time = time.time()
                matrix_times.append(end_time - start_time)
                
                # Time list implementation
                start_time = time.time()
                dijkstra_list_heap(adj_list, start_vertex)
                end_time = time.time()
                list_times.append(end_time - start_time)
            
            # Store average times
            results['matrix'][density].append(sum(matrix_times) / len(matrix_times))
            results['list'][density].append(sum(list_times) / len(list_times))
    
    return results

# Plot the crossover points
def plot_crossover_points(crossover_results):
    vertices = list(crossover_results.keys())
    densities = [crossover_results[v] if crossover_results[v] is not None else 1.0 for v in vertices]
    
    plt.figure(figsize=(10, 6))
    plt.plot(vertices, densities, 'o-', linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Density = 0.5')
    
    # Fill areas
    plt.fill_between(vertices, 0, densities, alpha=0.2, color='green', label='List+Heap better')
    plt.fill_between(vertices, densities, 1, alpha=0.2, color='blue', label='Matrix+Array better')
    
    plt.xlabel('Number of Vertices (V)')
    plt.ylabel('Crossover Density')
    plt.title('Density Threshold for Choosing Between Implementations')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('dijkstra_crossover_points.png')
    plt.close()

# Plot the detailed runtime behavior
def plot_runtime_behavior(vertex_sizes, behavior_results, densities):
    fig, axs = plt.subplots(len(densities), 1, figsize=(12, 5*len(densities)))
    
    for i, density in enumerate(densities):
        ax = axs[i] if len(densities) > 1 else axs
        
        matrix_times = behavior_results['matrix'][density]
        list_times = behavior_results['list'][density]
        
        # Main plot
        ax.plot(vertex_sizes, matrix_times, 'o-', linewidth=2, label='Matrix+Array', color='blue')
        ax.plot(vertex_sizes, list_times, 'o-', linewidth=2, label='List+Heap', color='green')
        
        # Find intersection points
        intersections = []
        for j in range(1, len(vertex_sizes)):
            if (matrix_times[j-1] - list_times[j-1]) * (matrix_times[j] - list_times[j]) <= 0:
                # Linear interpolation to find the exact crossing point
                v1, v2 = vertex_sizes[j-1], vertex_sizes[j]
                m1, m2 = matrix_times[j-1], matrix_times[j]
                l1, l2 = list_times[j-1], list_times[j]
                
                # Solve for the intersection point
                if m2 - m1 != l2 - l1:  # Avoid division by zero
                    x_intersect = v1 + (v2 - v1) * (l1 - m1) / ((m2 - m1) - (l2 - l1))
                    if v1 <= x_intersect <= v2:
                        # Linear interpolation for y value
                        y_intersect = m1 + (m2 - m1) * (x_intersect - v1) / (v2 - v1)
                        intersections.append((x_intersect, y_intersect))
        
        # Mark intersection points
        for x, y in intersections:
            ax.plot(x, y, 'ro', markersize=8)
            ax.annotate(f'Crossover at V≈{int(x)}', 
                        xy=(x, y), xytext=(x+50, y*1.2),
                        arrowprops=dict(arrowstyle='->'))
        
        # Add ratio subplot (list time / matrix time)
        ax2 = ax.twinx()
        ratios = [l/m if m > 0 else 0 for l, m in zip(list_times, matrix_times)]
        ax2.plot(vertex_sizes, ratios, 'r--', alpha=0.5, label='List/Matrix Ratio')
        ax2.axhline(y=1, color='k', linestyle='--', alpha=0.3)
        ax2.set_ylabel('List/Matrix Time Ratio', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Set labels and legend
        ax.set_xlabel('Number of Vertices (V)')
        ax.set_ylabel('Runtime (seconds)')
        ax.set_title(f'Runtime Behavior at Density = {density}')
        ax.grid(True)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('dijkstra_runtime_behavior.png')
    plt.close()

# Generate a theoretical explanation graph
def plot_theoretical_explanation():
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Runtime vs Density for fixed V
    densities = np.linspace(0, 1, 100)
    v_fixed = 500  # Example size
    
    # Theoretical runtime models (simplified)
    matrix_times = [v_fixed**2 * 0.00001 for _ in densities]  # O(V²)
    list_times = [(v_fixed + v_fixed * d * v_fixed) * np.log2(v_fixed) * 0.00001 for d in densities]  # O((V+E)logV) where E=d*V²
    
    ax1.plot(densities, matrix_times, label='Matrix+Array (O(V²))', color='blue')
    ax1.plot(densities, list_times, label='List+Heap (O((V+E)logV))', color='green')
    
    # Find the intersection point
    intersect_idx = np.argmin(np.abs(np.array(matrix_times) - np.array(list_times)))
    intersect_density = densities[intersect_idx]
    
    ax1.axvline(x=intersect_density, color='r', linestyle='--')
    ax1.annotate(f'Crossover at d≈{intersect_density:.2f}', 
                xy=(intersect_density, matrix_times[intersect_idx]),
                xytext=(intersect_density+0.1, matrix_times[intersect_idx]*1.2),
                arrowprops=dict(arrowstyle='->'))
    
    ax1.set_xlabel('Density (d)')
    ax1.set_ylabel('Theoretical Runtime')
    ax1.set_title(f'Theoretical Runtime vs Density (V={v_fixed})')
    ax1.legend()
    ax1.grid(True)
    
    # Subplot 2: Crossover density vs V
    vertices = np.linspace(100, 1000, 10)
    
    # Theoretical model for crossover density
    # For larger V, the list implementation gets better for dense graphs
    # The crossover density should decrease as V increases
    crossover_densities = [1/(1 + np.log2(v)/v) for v in vertices]
    
    ax2.plot(vertices, crossover_densities, 'o-', color='purple')
    ax2.fill_between(vertices, 0, crossover_densities, alpha=0.2, color='green', label='List+Heap better')
    ax2.fill_between(vertices, crossover_densities, 1, alpha=0.2, color='blue', label='Matrix+Array better')
    
    ax2.set_xlabel('Number of Vertices (V)')
    ax2.set_ylabel('Crossover Density')
    ax2.set_title('Theoretical Crossover Density vs Vertices')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('dijkstra_theoretical_explanation.png')
    plt.close()

def main():
    # Test parameters
    vertex_sizes_crossover = list(range(100, 5001, 98))
    density_range = np.linspace(0.1, 0.9, 9)  # 0.1, 0.2, ..., 0.9
    
    vertex_sizes_behavior = list(range(100, 1001, 100))  # 100, 200, ..., 1000
    behavior_densities = [0.3, 0.5, 0.7]  # Examining the behavior at these specific densities
    
    print("Finding crossover densities...")
    crossover_results = find_crossover_density(vertex_sizes_crossover, density_range)
    print("Crossover densities found:", crossover_results)
    
    print("Analyzing runtime behavior...")
    # behavior_results = analyze_runtime_behavior(vertex_sizes_behavior, behavior_densities)
    
    print("Generating plots...")
    plot_crossover_points(crossover_results)
    # plot_runtime_behavior(vertex_sizes_behavior, behavior_results, behavior_densities)
    # plot_theoretical_explanation()
    
    print("Analysis complete! Check the generated PNG files.")

if __name__ == "__main__":
    main()