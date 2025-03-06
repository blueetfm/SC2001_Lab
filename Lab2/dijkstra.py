import heapq
import math

def printSolution(V, dist):
    print("Vertex \tDistance from Source")
    for node in range(V):
        print(node, "   \t", dist[node])
            
# Implementation a: adj matric + array
def dijkstra_matrix_array(graph, start):
    """
    Args:
        graph: Adjacency matrix where graph[i][j] is the weight from i to j
              (infinity if no edge exists)
        start: Starting vertex
    
    Returns:
        Tuple of (list of shortest distances from start to all vertices, comparison count)
    """
    n = len(graph)  # Number of vertices
    comparisons = 0
    
    # Initialize distances and visited array
    dist = [float('infinity')] * n
    pi = [None] * n
    visited = [False] * n

    dist[start] = 0
    
    for _ in range(n):
        min_dist = float('infinity')

        # Part 1: Finding the cheapest (unvisited) vertex
        # ----------------------------------------------
        # This part identifies the next vertex to process.
        # We search for the vertex with the smallest known distance that has not been visited yet.
        # This mimics the extract-min operation of a priority queue but with O(V) complexity.
        min_vertex = -1
        for v in range(n):
            if not visited[v] and dist[v] < min_dist:
                min_dist = dist[v]
                min_vertex = v
        
        comparisons += n  # V comparisons for finding minimum
        
        if min_vertex == -1:
            break
        
        # Part 2: The Core of Dijkstra’s Algorithm – Relaxation Process
        # ------------------------------------------------------------
        # Once the vertex with the smallest known distance is selected (min_vertex), we:
        # - Mark it as visited, meaning its shortest path is finalized.
        # - Update the distances of its adjacent vertices if a shorter path is found.
        visited[min_vertex] = True
        
        for v in range(n):
            if not visited[v] and graph[min_vertex][v] != float('infinity'):
                if dist[min_vertex] + graph[min_vertex][v] < dist[v]:
                    pi[v] = min_vertex
                    dist[v] = dist[min_vertex] + graph[min_vertex][v]
        
        comparisons += n  # V comparisons for updating neighbors paths
    
    printSolution(n, dist)
    return dist, comparisons


# Implementation b: adj list + heap
def dijkstra_list_heap(graph, start):
    """
    Args:
        graph: Dictionary where keys are vertices and values are lists of (neighbor, weight) tuples
        start: Starting vertex
    
    Returns:
        Tuple of (dictionary of shortest distances from start to all vertices, pi, comparison count)
    """
    n = len(graph)  # Number of vertices
    comparisons = 0
    heap_operations = 0  # count of heap operations
    edge_comparisons = 0  # count of edge relaxation comparisons
    

    # dist = {vertex: float('infinity') for vertex in graph}
    dist = [float('infinity')] * n
    pi = [None] * n
    visited = [False] * n
    dist[start] = 0

    priority_queue = [(0, start)]
    
    while priority_queue:
        d_u, u = heapq.heappop(priority_queue)
        heap_operations += 1
        comparisons += 2 * math.log2(len(priority_queue) + 1) if priority_queue else 0  # Heap extraction comparisons
        
        if visited[u]:
            continue

        visited[u] = True  

        for v in range(n):
            if not visited[v] and graph[u][v] != float('infinity'):
                comparisons += 1  # Comparison for relaxation
                
                if dist[u] + graph[u][v] < dist[v]: 
                    dist[v] = dist[u] + graph[u][v]
                    pi[v] = u
                    heapq.heappush(priority_queue, (dist[v], v))  
                    heap_operations += 1
                    comparisons += 2 * math.log2(len(priority_queue)) if priority_queue else 0  # Heap insertion comparisons
    
    
    '''while priority_queue:
        # get v with min distance
        d_u, = heapq.heappop(priority_queue)
        heap_operations += 1
        comparisons += 2 * math.log2(len(priority_queue) + 1) if priority_queue else 0
        
        # skip if better path exist
        if current_dist > dist[current_vertex]:
            continue
        
        # check all neighbours
        for neighbor, weight in graph[current_vertex]:
            distance = current_dist + weight
            
            # update dist if better path found
            edge_comparisons += 1
            if distance < dist[neighbor]:
                dist[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                heap_operations += 1
                comparisons += 2 * math.log2(len(priority_queue)) if priority_queue else 0
    
    # total comp = heap comps + edge relaxation comps
    comparisons += edge_comparisons'''
    printSolution(n, dist)
    return dist, pi, comparisons