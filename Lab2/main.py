import random
from dijkstra import printSolution, dijkstra_matrix_array, dijkstra_list_heap
import time
import numpy as np
import matplotlib.pyplot as plt

def generate_graph(n, density=0.5, max_weight=10):
    """
    Generates a random weighted graph.

    Returns:
        - adj_matrix: Adjacency matrix (2D list)
        - adj_list: Array of adjacency lists (list of lists)
    """

    adj_matrix = [[float('inf')] * n for _ in range(n)]
    adj_list = [[] for _ in range(n)]  # nodes with no connections are simply not represented
    edge_count = 0

    for i in range(n):
        adj_matrix[i][i] = 0 

        for j in range(i + 1, n):
            if random.random() < density:
                weight = random.randint(1, max_weight)
                
                # Populate adjacency matrix
                adj_matrix[i][j] = weight # directed graph

                # Populate adjacency list (array of lists)
                adj_list[i].append((j, weight))
                edge_count += 1

    return adj_matrix, adj_list, edge_count


def theoretical_time(n, e, implementation):
    """Calculate theoretical time complexity"""
    if implementation == "matrix_array":
        return n**2  # O(|V|²)
    else:  # list_heap
        return (e + n) * np.log2(n)  # O((|E|+|V|)log|V|)

if __name__ == "__main__":
    # """Measure performance of adjacency matrix + array implementation"""
    sizes = list(range(50, 2001, 50))  # Vertex counts from 50 to 2000
    times_matrix_array = []
    theoretical_results_matrix_array = []

    for n in sizes:
        graph, adj_list, edge_count = generate_graph(n)
        theoretical_results_matrix_array.append(theoretical_time(n, edge_count, "matrix_array"))

        # Measure time
        start_time = time.perf_counter()
        dijkstra_matrix_array(graph, 0)
        end_time = time.perf_counter()

        times_matrix_array.append(end_time-start_time)
        print(f"Matrix+Array: |V|={n}, |E|={edge_count}, Time={end_time-start_time:.6f}s")

    scale_matrix = np.mean(np.array(times_matrix_array) / np.array(theoretical_results_matrix_array))
    theoretical_results_matrix_array = [t * scale_matrix for t in theoretical_results_matrix_array]

    # Plot the results
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(sizes, times_matrix_array, 'bo-', label="Empirical (Matrix+Array)")
    plt.plot(sizes, theoretical_results_matrix_array, 'b--', label="Theoretical O(|V|²)")
    plt.title("Adjacency Matrix + Array PQ: Empirical vs Theoretical Time Complexity")
    plt.xlabel("|V| (Number of Vertices)")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.grid(True)

    """Measure performance of adjacency list + heap implementation"""
    times_list_heap = []
    theoretical_results_list_heap = []

    for n in sizes:
        graph, adj_list, edge_count = generate_graph(n)
        theoretical_results_list_heap.append(theoretical_time(n, edge_count, "list_heap"))

        # Measure time
        start_time = time.perf_counter()
        dijkstra_list_heap(adj_list, 0)
        end_time = time.perf_counter()

        times_list_heap.append(end_time-start_time)

        print(f"List+Heap: |V|={n}, |E|={edge_count}, Time={end_time-start_time:.6f}s")

    scale_matrix = np.mean(np.array(times_list_heap) / np.array(theoretical_results_list_heap))
    theoretical_results = [t * scale_matrix for t in theoretical_results_list_heap]

    # Plot the results
    plt.subplot(2, 1, 2)
    plt.plot(sizes, times_list_heap, 'ro-', label="Empirical (List+Heap)")
    plt.plot(sizes, theoretical_results, 'r--', label="Theoretical O((|E|+|V|)log|V|)")
    plt.title("Adjacency List + Heap PQ: Empirical vs Theoretical Time Complexity")
    plt.xlabel("|V| (Number of Vertices)")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()


    # Test different densities with fixed size
    n = 50
    densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    density_results = {
        "matrix_array": {"times": [], "edges": []},
        "list_heap": {"times": [], "edges": []}
    }

    density_theoretical = {
        "matrix_array": [],
        "list_heap": []
    }

    # Run density tests
    print("\nTesting with different graph densities (|V|=10000):")
    for density in densities:
        matrix, adj_list, edge_count = generate_graph(n, density=density)

        # Measure matrix+array implementation
        start_time = time.perf_counter()
        dijkstra_matrix_array(matrix, 0)
        matrix_time = time.perf_counter() - start_time

        # Measure list+heap implementation
        start_time = time.perf_counter()
        dijkstra_list_heap(adj_list, 0)
        list_time = time.perf_counter() - start_time

        # Record results
        density_results["matrix_array"]["times"].append(matrix_time)
        density_results["matrix_array"]["edges"].append(edge_count)
        density_results["list_heap"]["times"].append(list_time)
        density_results["list_heap"]["edges"].append(edge_count)

        # Calculate theoretical time
        density_theoretical["matrix_array"].append(theoretical_time(n, edge_count, "matrix_array"))
        density_theoretical["list_heap"].append(theoretical_time(n, edge_count, "list_heap"))

        print(f"Density={density}, |E|={edge_count}:")
        print(f"  Matrix+Array: {matrix_time:.6f}s")
        print(f"  List+Heap: {list_time:.6f}s")

    # Scale theoretical results for density comparison
    scale_matrix = np.mean(np.array(density_results["matrix_array"]["times"]) /
                           np.array(density_theoretical["matrix_array"]))
    scale_list = np.mean(np.array(density_results["list_heap"]["times"]) /
                         np.array(density_theoretical["list_heap"]))

    density_theoretical["matrix_array"] = [t * scale_matrix for t in density_theoretical["matrix_array"]]
    density_theoretical["list_heap"] = [t * scale_list for t in density_theoretical["list_heap"]]

    # Plot comparison for different densities
    plt.figure(figsize=(15, 10))

    # Plot both implementations vs edge count
    edge_counts = density_results["matrix_array"]["edges"]

    plt.subplot(2, 1, 1)
    plt.plot(edge_counts, density_results["matrix_array"]["times"], 'bo-', label="Empirical (Matrix+Array)")
    plt.plot(edge_counts, density_theoretical["matrix_array"], 'b--', label="Theoretical O(|V|²)")
    plt.title("Effect of Edge Density on Matrix+Array Implementation (|V|=200)")
    plt.xlabel("|E| (Number of Edges)")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(edge_counts, density_results["list_heap"]["times"], 'ro-', label="Empirical (List+Heap)")
    plt.plot(edge_counts, density_theoretical["list_heap"], 'r--', label="Theoretical O((|E|+|V|)log|V|)")
    plt.title("Effect of Edge Density on List+Heap Implementation (|V|=200)")
    plt.xlabel("|E| (Number of Edges)")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot comparison between implementations
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_matrix_array, 'bo-', label="Empirical (Matrix+Array)")
    plt.plot(sizes, times_list_heap, 'ro-', label="Empirical (List+Heap)")

    if any(m > l for m, l in zip(times_matrix_array, times_list_heap)):
        # Find crossover point by interpolation
        for i in range(1, len(sizes)):
            if (times_matrix_array[i - 1] <= times_list_heap[i - 1] and
                    times_matrix_array[i] >= times_list_heap[i]):
                # Interpolate crossover point
                x1, y1 = sizes[i - 1], times_matrix_array[i - 1]
                x2, y2 = sizes[i], times_matrix_array[i]
                x3, y3 = sizes[i - 1], times_list_heap[i - 1]
                x4, y4 = sizes[i], times_list_heap[i]

                # Linear interpolation
                m1 = (y2 - y1) / (x2 - x1)
                m2 = (y4 - y3) / (x4 - x3)
                b1 = y1 - m1 * x1
                b2 = y3 - m2 * x3

                x_cross = (b2 - b1) / (m1 - m2)
                y_cross = m1 * x_cross + b1

                plt.plot(x_cross, y_cross, 'go', markersize=8)
                plt.annotate(f"Crossover: |V|≈{int(x_cross)}",
                             (x_cross, y_cross),
                             xytext=(x_cross + 20, y_cross * 1.1),
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.title("Empirical Comparison: Matrix+Array vs List+Heap (density=0.5)")
    plt.xlabel("|V| (Number of Vertices)")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()

    sizes = list(range(10, 200, 10))  # Vertex counts from 50 to 2000
    times_matrix_array = []
    times_list_heap = []

    for n in sizes:
        graph, adj_list, edge_count = generate_graph(n, density=0.5)

        # Measure time
        start_time = time.perf_counter()
        dijkstra_matrix_array(graph, 0)
        end_time = time.perf_counter()

        times_matrix_array.append(end_time - start_time)
        print(f"Matrix+Array: |V|={n}, |E|={edge_count}, Time={end_time - start_time:.6f}s")

        start_time = time.perf_counter()
        dijkstra_list_heap(adj_list, 0)
        end_time = time.perf_counter()

        times_list_heap.append(end_time - start_time)

        print(f"List+Heap: |V|={n}, |E|={edge_count}, Time={end_time - start_time:.6f}s")


    # Plot the results
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(sizes, times_matrix_array, 'bo-', label="Empirical (Matrix+Array)")
    plt.plot(sizes, times_list_heap, 'b--', label="Empirical (List+Heap)")
    plt.title("Adjacency Matrix + Array PQ vs List + Heap (density=0.5)")
    plt.xlabel("|V| (Number of Vertices)")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.grid(True)

    times_matrix_array = []
    times_list_heap = []

    for n in sizes:
        graph, adj_list, edge_count = generate_graph(n, density=0.9)

        # Measure time
        start_time = time.perf_counter()
        dijkstra_matrix_array(graph, 0)
        end_time = time.perf_counter()

        times_matrix_array.append(end_time - start_time)
        print(f"Matrix+Array: |V|={n}, |E|={edge_count}, Time={end_time - start_time:.6f}s")

        start_time = time.perf_counter()
        dijkstra_list_heap(adj_list, 0)
        end_time = time.perf_counter()

        times_list_heap.append(end_time - start_time)

        print(f"List+Heap: |V|={n}, |E|={edge_count}, Time={end_time - start_time:.6f}s")

    # Plot the results
    plt.subplot(2, 1, 2)
    plt.plot(sizes, times_matrix_array, 'bo-', label="Empirical (Matrix+Array)")
    plt.plot(sizes, times_list_heap, 'r-', label="Empirical (List+Heap)")
    plt.title("Adjacency Matrix + Array PQ vs List + Heap (density=0.9)")
    plt.xlabel("|V| (Number of Vertices)")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # n = 6
    # adj_matrix, adj_list, edge_count = generate_graph(n)
    #
    # print("Adjacency Matrix:")
    # for row in adj_matrix:
    #     print(row)
    #
    # print("\nAdjacency List (Array of Lists):")
    # for i, neighbors in enumerate(adj_list):
    #     print(f"{i}: {neighbors}")
    #
    #
    # n_matrix, dist_matrix, pi_matrix, comparisons_matrix = dijkstra_matrix_array(adj_matrix, 0)
    # n_list, dist_list, pi_list, comparisons_list = dijkstra_list_heap(adj_list, 0)
    #
    # print("Solution using adjacency matrix and priority array representation: \n")
    # printSolution(n_matrix, dist_matrix)
    # print(f"Number of comparisons using adjacency matrix and array representation: {comparisons_matrix}")
    #
    # print("Solution using adjacency list and priority queue representation: \n")
    # printSolution(n_list, dist_list)
    # print(f"Number of comparisons using adjacency list and priority queue representation: {comparisons_list}")