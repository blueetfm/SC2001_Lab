from knapsack import knapsack

def part_4a(capacity, n):
    weights = [4, 6, 8]
    profits = [7, 6, 9]

    res = knapsack(weights, profits, capacity, n)
    print(f"The result of P(14) with weights [4, 6, 8] and profits [7, 6, 9] is {res}.")


def part_4b(capacity, n):
    weights = [5, 6, 8]
    profits = [7, 6, 9]

    res = knapsack(weights, profits, capacity, n)
    print(f"The result of P(14) with weights [5, 6, 8] and profits [7, 6, 9] is {res}.")

part_4a(14, 3)
part_4b(14, 3)