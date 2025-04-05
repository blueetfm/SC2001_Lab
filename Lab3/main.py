from knapsack import knapsack, print_matrix, unbounded_knapsack

def part_4a(weights, profits, capacity, n):
    print(f"\n--- Running knapsack for capacity = {capacity}, weights = {weights}, profits = {profits} ---")
    res_matrix, max_profit = knapsack(weights, profits, capacity, n)
    print_matrix(weights, profits, res_matrix)

    print(f"The result of P(14) with weights [4, 6, 8] and profits [7, 6, 9] is {max_profit}.")

# Example 1:
weights1 = [4, 6, 8]
profits1 = [7, 6, 9]

# Example 2:
weights2 = [5, 6, 8]
profits2 = [7, 6, 9]

unbounded_knapsack(weights1, profits1, 14)
unbounded_knapsack(weights2, profits2, 14)