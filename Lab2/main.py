import random
from dijkstra import printSolution, dijkstra_matrix_array, dijkstra_list_heap

def generate_graph(n, density=6, max_weight=10):
    """
    Generates a random weighted graph.

    Returns:
        - adj_matrix: Adjacency matrix (2D list)
        - adj_list: Array of adjacency lists (list of lists)
    """

    adj_matrix = [[float('inf')] * n for _ in range(n)]
    adj_list = [[] for _ in range(n)]  # nodes with no connections are simply not represented

    for i in range(n):
        adj_matrix[i][i] = 0 

        for j in range(i + 1, n):
            if random.random() < density:
                weight = random.randint(1, max_weight)
                
                # Populate adjacency matrix
                adj_matrix[i][j] = weight # directed graph

                # Populate adjacency list (array of lists)
                adj_list[i].append((j, weight))

    return adj_matrix, adj_list

n = 6
adj_matrix, adj_list = generate_graph(n)

print("Adjacency Matrix:")
for row in adj_matrix:
    print(row)

print("\nAdjacency List (Array of Lists):")
for i, neighbors in enumerate(adj_list):
    print(f"{i}: {neighbors}")


n_matrix, dist_matrix, pi_matrix, comparisons_matrix = dijkstra_matrix_array(adj_matrix, 0)
n_list, dist_list, pi_list, comparisons_list = dijkstra_list_heap(adj_list, 0)

print("Solution using adjacency matrix and priority array representation: \n")
printSolution(n_matrix, dist_matrix)
print(f"Number of comparisons using adjacency matrix and array representation: {comparisons_matrix}")

print("Solution using adjacency list and priority queue representation: \n")
printSolution(n_list, dist_list)
print(f"Number of comparisons using adjacency list and priority queue representation: {comparisons_list}")