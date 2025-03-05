import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple
import random
import sort

def generate_dataset(size: int, max_value: int) -> list:
    """Generate a random dataset of specified size"""
    return [random.randint(1, max_value) for _ in range(size)]

def compare_algorithms(input_sizes: List[int], s: int, num_trials: int = 3) -> Tuple[List[dict], List[dict]]:
    """
    Compare hybrid and original merge sort across different input sizes
    Returns two lists of dictionaries containing performance metrics
    """
    hybrid_results = []
    original_results = []
    max_value = 1000000  # Maximum value for random numbers

    for size in input_sizes:
        print(f"\nTesting input size: {size}")
        
        # Initialize accumulators for this size
        hybrid_total_time = 0
        hybrid_total_comparisons = 0
        original_total_time = 0
        original_total_comparisons = 0

        # Run multiple trials
        for trial in range(num_trials):
            # Generate same data for both algorithms
            data = generate_dataset(size, max_value)
            
            # Test hybrid sort
            hybrid_data = data.copy()
            start_time = time.time()
            hybrid_comparisons = sort.merge_sort_hybrid(hybrid_data, 0, len(hybrid_data)-1, s)
            hybrid_time = time.time() - start_time
            
            # Test original merg sort
            original_data = data.copy()
            start_time = time.time()
            original_comparisons = sort.merge_sort_ori(original_data, 0, len(original_data)-1)
            original_time = time.time() - start_time

            # Accumulate results
            hybrid_total_time += hybrid_time
            hybrid_total_comparisons += hybrid_comparisons
            original_total_time += original_time
            original_total_comparisons += original_comparisons

        # Calculate averages
        hybrid_results.append({
            'size': size,
            'time': hybrid_total_time / num_trials,
            'comparisons': hybrid_total_comparisons / num_trials
        })
        
        original_results.append({
            'size': size,
            'time': original_total_time / num_trials,
            'comparisons': original_total_comparisons / num_trials
        })

        print(f"Hybrid - Avg time: {(hybrid_total_time/num_trials)*1000:.2f}ms, "
              f"Avg comparisons: {hybrid_total_comparisons/num_trials:.0f}")
        print(f"Original - Avg time: {(original_total_time/num_trials)*1000:.2f}ms, "
              f"Avg comparisons: {original_total_comparisons/num_trials:.0f}")

    return hybrid_results, original_results

def plot_comparison(hybrid_results: List[dict], original_results: List[dict], s: int):
    """
    Create comparison plots for runtime and number of comparisons
    """
    # Extract data
    sizes = [r['size'] for r in hybrid_results]
    hybrid_times = [r['time'] * 1000 for r in hybrid_results]  # Convert to milliseconds
    original_times = [r['time'] * 1000 for r in original_results]
    hybrid_comparisons = [r['comparisons'] for r in hybrid_results]
    original_comparisons = [r['comparisons'] for r in original_results]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Runtime comparison
    ax1.plot(sizes, hybrid_times, 'b-', marker='o', label='Hybrid Sort')
    ax1.plot(sizes, original_times, 'r--', marker='s', label='Merge Sort')
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    ax1.set_xlabel('Input Size (n)')
    ax1.set_ylabel('Runtime (ms)')
    ax1.set_title(f'Runtime Comparison (S={s})')
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Comparisons comparison
    ax2.plot(sizes, hybrid_comparisons, 'b-', marker='o', label='Hybrid Sort')
    ax2.plot(sizes, original_comparisons, 'r--', marker='s', label='Merge Sort')
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    ax2.set_xlabel('Input Size (n)')
    ax2.set_ylabel('Number of Comparisons')
    ax2.set_title(f'Number of Comparisons (S={s})')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Print percentage differences
    print("\nPerformance Differences (Hybrid vs Merge):")
    for i, size in enumerate(sizes):
        time_diff = (hybrid_times[i] - original_times[i]) / original_times[i] * 100
        comp_diff = (hybrid_comparisons[i] - original_comparisons[i]) / original_comparisons[i] * 100
        print(f"\nInput size {size}:")
        print(f"Runtime difference: {time_diff:+.2f}% ({'+' if time_diff > 0 else '-'}slower)")
        print(f"Comparisons difference: {comp_diff:+.2f}% ({'+' if comp_diff > 0 else '-'}more)")

# Main execution
if __name__ == "__main__":
    # Test parameters
    input_sizes = list(range(1000, 10000001, 999900)) #for 50 data points
    fixed_s = 3
    num_trials = 3  # Number of trials for each size to average resultsß

    # Run comparison
    print(f"Running comparison with S={fixed_s}...")
    hybrid_results, original_results = compare_algorithms(input_sizes, fixed_s, num_trials)
    
    # Plot results
    plot_comparison(hybrid_results, original_results, fixed_s)