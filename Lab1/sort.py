'''
 The idea is to set a small integer S as a threshold for the size of subarrays.
Once the size of a subarray in a recursive call of Mergesort is less than or equal to S,
the algorithm will switch to Insertion Sort, which is efficient for small-sized input.

(a) Algorithm implementation: Implement the above hybrid algorithm.
'''


def insertion_sort(arr, left, right) -> []:
    # both are the same implementation, just not sure which one is clearer to read
    comparisons = 0

    for i in range(left + 1, right + 1):
        for j in range(i, left, -1):
            if arr[j-1] > arr[j]:
                comparisons += 1
                arr[j], arr[j-1] = arr[j-1], arr[j]
            else:
                break

    return comparisons
    # for i in range(left + 1, right + 1):
    #     j = i
    #     while j > left and arr[j] < arr[j - 1]:
    #         arr[j], arr[j - 1] = arr[j - 1], arr[j]
    #         j -= 1

    # return arr


def merge(arr, left, mid, right):

    left_sub = arr[left:mid + 1]
    right_sub = arr[mid + 1:right + 1]

    i = j = comparisons = 0  # Initial index of first subarray and second subarray
    k = left  # Initial index of merged subarray

    while i < len(left_sub) and j < len(right_sub):
        if left_sub[i] <= right_sub[j]:
            arr[k] = left_sub[i]
            i += 1
        else:
            arr[k] = right_sub[j]
            j += 1
        comparisons += 1
        k += 1

    # remaining elements of L[], if there are any
    while i < len(left_sub):
        arr[k] = left_sub[i]
        i += 1
        k += 1

    # remaining elements of R[], if there are any
    while j < len(right_sub):
        arr[k] = right_sub[j]
        j += 1
        k += 1
    
    return comparisons


def merge_sort_ori(arr, left, right):
    total_comparisons = 0

    if left < right:

        mid = (left + right) // 2

        total_comparisons += merge_sort_ori(arr, left, mid)
        total_comparisons += merge_sort_ori(arr, mid + 1, right)
        total_comparisons += merge(arr, left, mid, right)
    
    return total_comparisons


def merge_sort_hybrid(arr, left, right, s):
    total_comparisons = 0

    if left < right:

        if (right - left) <= s:
            insertion_comparisons = insertion_sort(arr, left, right)
            total_comparisons += insertion_comparisons
            
            return total_comparisons

        mid = (left + right) // 2

        total_comparisons += merge_sort_hybrid(arr, left, mid, s)
        total_comparisons += merge_sort_hybrid(arr, mid + 1, right, s)
        total_comparisons += merge(arr, left, mid, right)
    
    return total_comparisons


# We can change it such that the function itself returns the array 
arr = [41, 714, 914, 815, 198, 972, 1013, 1040, 865, 273, 886, 925, 84, 623, 963, 179, 277, 640, 415]
print(merge_sort_ori(arr, 0, 18))
print(arr)
arr = [41, 714, 914, 815, 198, 972, 1013, 1040, 865, 273, 886, 925, 84, 623, 963, 179, 277, 640, 415]
print(merge_sort_hybrid(arr, 0, 18, 4))
print(arr)