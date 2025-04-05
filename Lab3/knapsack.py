# Part 1 
# def knapsack_recursive(weights, profits, capacity, n):
#     if n == 0 or capacity == 0:
#         return 0

#     if weights[n-1] <= capacity:


#     return


# Part 3
def knapsack(weights, profits, capacity, n):
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

  
    return profit[capacity][n]

