import random
from dijkstra import printSolution, dijkstra_matrix_array, dijkstra_list_heap
import time
import numpy as np
import matplotlib.pyplot as plt

def generate_graph(n, density=6, max_weight=10):
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


"""Measure performance of adjacency matrix + array implementation"""
sizes = list(range(50, 2001, 200))  # Vertex counts from 50 to 2000
times = []
theoretical_results = []

for n in sizes:
    graph, adj_list, edge_count = generate_graph(n)
    theoretical_results.append(theoretical_time(n, edge_count, "matrix_array"))

    # Measure time
    start_time = time.perf_counter()
    dijkstra_matrix_array(graph, 0)
    end_time = time.perf_counter()

    times.append(end_time-start_time)
    print(f"Matrix+Array: |V|={n}, |E|={edge_count}, Time={end_time-start_time:.6f}s")

scale_matrix = np.mean(np.array(times) / np.array(theoretical_results))
theoretical_results = [t * scale_matrix for t in theoretical_results]

# Plot the results
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(sizes, times, 'bo-', label="Empirical (Matrix+Array)")
plt.plot(sizes, theoretical_results, 'b--', label="Theoretical O(|V|²)")
plt.title("Adjacency Matrix + Array PQ: Empirical vs Theoretical Time Complexity")
plt.xlabel("|V| (Number of Vertices)")
plt.ylabel("Runtime (seconds)")
plt.legend()
plt.grid(True)

"""Measure performance of adjacency list + heap implementation"""
sizes = list(range(50, 2001, 200))  # Vertex counts from 50 to 2000
times = []
theoretical_results = []

for n in sizes:
    graph, adj_list, edge_count = generate_graph(n)
    theoretical_results.append(theoretical_time(n, edge_count, "list_heap"))

    # Measure time
    start_time = time.perf_counter()
    dijkstra_list_heap(adj_list, 0)
    end_time = time.perf_counter()

    times.append(end_time-start_time)

    print(f"List+Heap: |V|={n}, |E|={edge_count}, Time={end_time-start_time:.6f}s")

scale_matrix = np.mean(np.array(times) / np.array(theoretical_results))
theoretical_results = [t * scale_matrix for t in theoretical_results]

# Plot the results
plt.subplot(2, 1, 2)
plt.plot(sizes, times, 'ro-', label="Empirical (List+Heap)")
plt.plot(sizes, theoretical_results, 'r--', label="Theoretical O((|E|+|V|)log|V|)")
plt.title("Adjacency List + Heap PQ: Empirical vs Theoretical Time Complexity")
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