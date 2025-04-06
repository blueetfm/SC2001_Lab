def zero_one_knapsack(weights, profits, capacity, n):
    profit = [[0] * (n + 1) for _ in range(capacity + 1)]

    # Base Case 1: if we have 0 items to choose from, the profit is 0
    for j in range(1, capacity):
        profit[j][0] = 0

    # Base Case 2: if the knapsack capacity is 0, the profit is 0
    for i in range(0, n):
        profit[0][i] = 0
    
    
    for c in range(1, capacity + 1):
        for j in range(1, n + 1):

            # if this item fits in the current knapsack capacity
            # Note that item j has index 'j-1' in weights and profits
            if weights[j - 1] <= c:
                # Choice 1: profit[i][j-1] does not include the j-th item
                # Choice 2: profit[i - weights[j]][j-1] + profits[j] includes the j-th item, 
                # # and represents the max profit of remaining capacity and items, plus the added profit of j-th item
                profit[c][j] = max(profit[c][j-1], profit[c - weights[j - 1]][j-1] + profits[j-1])
  
    return profit, profit[capacity][n]

def unbounded_knapsack(weights, profits, capacity):
    dp_array = [0] * (capacity + 1)

    for c in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= c:
                dp_array[c] = max(dp_array[c], dp_array[c - weights[i]] + profits[i])
    
    print_unbounded_matrix(weights, profits, dp_array)
    return dp_array[capacity]


def print_matrix(weights, profits, matrix):
    n = len(matrix[0])
    capacity = len(matrix)

    # Print column headers
    print("Capacity\\Items |", end=" ")
    for j in range(n):
        if j == 0:
            print("0   ", end=" ")  # No items
        else:
            print(f"{j}({weights[j-1]}/{profits[j-1]})", end=" ")
    print("\n" + "-" * (12 + n * 8))

    # Print matrix row by row
    for i in range(capacity):
        print(f"{i:<14}|", end=" ")
        for j in range(n):
            print(f"{matrix[i][j]:<8}", end=" ")
        print()
    print()


def print_unbounded_matrix(weights, profits, dp_array):
    capacity = len(dp_array)
    n = len(weights)
    
    # Create a header with item information
    print("\nUnbounded Knapsack DP Array:")
    print(f"Items available: ", end="")
    for i in range(n):
        print(f"Item {i}(w:{weights[i]}/v:{profits[i]})", end="  ")
    print("\n" + "-" * 50)
    
    # Print capacity and corresponding max value
    print("Capacity | Max Value")
    print("-" * 20)
    
    for c in range(capacity):
        print(f"{c:<9}| {dp_array[c]}")
    
    # Optional: add a summary of the optimal solution
    max_capacity = capacity - 1
    print("\nOptimal solution for capacity", max_capacity, ":", dp_array[max_capacity])
    
    # Backtrack to show which items were selected (optional)
    remaining = max_capacity
    item_count = [0] * n
    print("\nItems selected:")
    
    while remaining > 0:
        was_item_added = False
        for i in range(n):
            if weights[i] <= remaining and dp_array[remaining] == dp_array[remaining - weights[i]] + profits[i]:
                item_count[i] += 1
                remaining -= weights[i]
                was_item_added = True
                break
        if not was_item_added:
            break
    
    for i in range(n):
        if item_count[i] > 0:
            print(f"Item {i} (w:{weights[i]}/v:{profits[i]}) × {item_count[i]}")



