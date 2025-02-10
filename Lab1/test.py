'''
(b) Generate input data: Generate arrays of increasing sizes, in a range from
1,000 to 10 million. For each of the sizes, generate a random dataset of integers
in the range of [1, …, x], where x is the largest number you allow for your
datasets.

(c) Analyze time complexity: Run your program of the hybrid algorithm on the
datasets generated in Step (b). 
    - Record the number of key comparisons performed in each case.
    
    i. With the value of S fixed, plot the number of key comparisons over
    different sizes of the input list n. Compare your empirical results with
    your theoretical analysis of the time complexity.

    ii. With the input size n fixed, plot the number of key comparisons over
    different values of S. Compare your empirical results with your
    theoretical analysis of the time complexity.

    iii. Using different sizes of input datasets, study how to determine an
    optimal value of S for the best performance of this hybrid algorithm.
'''
import random


# do we want to allow repeat numbers?
def generate_arrays(max) -> [[]]:
    test_arrays = []

    # generate 100 test arrays
    for i in range(10):
        rand_size = random.randint(1000, 10000000)
        array = [random.randint(1, max) for _ in range(rand_size)]
        random.shuffle(array)

        test_arrays.append(array)
    
    return test_arrays

print(generate_arrays(1048))
