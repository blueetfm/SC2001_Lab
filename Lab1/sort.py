'''
 The idea is to set a small integer S as a threshold for the size of subarrays.
Once the size of a subarray in a recursive call of Mergesort is less than or equal to S,
the algorithm will switch to Insertion Sort, which is efficient for small-sized input.

(a) Algorithm implementation: Implement the above hybrid algorithm.
'''


def insertion_sort(arr, left, right) -> []:
    for i in range(left + 1, right + 1):
        j = i
        while j > left and arr[j] < arr[j - 1]:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j -= 1

    return arr


def merge(arr, left, mid, right):
    # Create temporary arrays
    left_sub = arr[left:mid + 1]
    right_sub = arr[mid + 1:right + 1]

    i = j = 0  # Initial index of first subarray and second subarray
    k = left  # Initial index of merged subarray

    # Merge the temp arrays
    while i < len(left_sub) and j < len(right_sub):
        if left_sub[i] <= right_sub[j]:
            arr[k] = left_sub[i]
            i += 1
        else:
            arr[k] = right_sub[j]
            j += 1
        k += 1

    # Copy the remaining elements of L[],
    # if there are any
    while i < len(left_sub):
        arr[k] = left_sub[i]
        i += 1
        k += 1

    # Copy the remaining elements of R[],
    # if there are any
    while j < len(right_sub):
        arr[k] = right_sub[j]
        j += 1
        k += 1


# Original Mergesort
def merge_sort_ori(arr, left, right):
    if left < right:
        mid = (left + right) // 2

        merge_sort_ori(arr, left, mid)
        merge_sort_ori(arr, mid + 1, right)
        merge(arr, left, mid, right)
