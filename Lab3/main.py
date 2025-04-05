from knapsack import knapsack, print_matrix, unbounded_knapsack

# Example 1:
weights1 = [4, 6, 8]
profits1 = [7, 6, 9]

# Example 2:
weights2 = [5, 6, 8]
profits2 = [7, 6, 9]

unbounded_knapsack(weights1, profits1, 14)
unbounded_knapsack(weights2, profits2, 14)