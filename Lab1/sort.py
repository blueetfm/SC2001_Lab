'''
 The idea is to set a small integer S as a threshold for the size of subarrays.
Once the size of a subarray in a recursive call of Mergesort is less than or equal to S,
the algorithm will switch to Insertion Sort, which is efficient for small-sized input.

(a) Algorithm implementation: Implement the above hybrid algorithm.
'''

def insertion_sort(arr) -> []:


    return arr

def merge

def mergesort(low, high, arr, s) -> None:
    if (high - low + 1) <= s:
        insertion_sort(arr)

    mid = (low + high)/2
    if high <= low:
        return
    elif (high - low) > 1:
        mergesort(low, mid, arr, s)
        mergesort(mid + 1, high, arr, s)
    
    merge(low, high)


    return
