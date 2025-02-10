'''
 The idea is to set a small integer S as a threshold for the size of subarrays.
Once the size of a subarray in a recursive call of Mergesort is less than or equal to S,
the algorithm will switch to Insertion Sort, which is efficient for small-sized input.

(a) Algorithm implementation: Implement the above hybrid algorithm.
'''


def insertion_sort(arr, left, right) -> []:
    for i in range(left + 1, right + 1):
        for j in range(i, left, -1):
            if arr[j-1] > arr[j]:
                arr[j], arr[j-1] = arr[j-1], arr[j]
            else:
                break
    return arr
    # for i in range(left + 1, right + 1):
    #     j = i
    #     while j > left and arr[j] < arr[j - 1]:
    #         arr[j], arr[j - 1] = arr[j - 1], arr[j]
    #         j -= 1

    # return arr


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
def merge_sort_ori(arr, left, right, s):
    if left < right:

        if (right - left) <= s:
            insertion_sort(arr, left, right)
            return arr

        mid = (left + right) // 2

        merge_sort_ori(arr, left, mid, s)
        merge_sort_ori(arr, mid + 1, right, s)
        merge(arr, left, mid, right)
    
    return arr


print(merge_sort_ori([1, 4, 2, 6], 0, 3, 2))