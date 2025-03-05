import heapq
import math

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
    dist[start] = 0
    visited = [False] * n
    
    for _ in range(n):
        min_dist = float('infinity')
        min_vertex = -1
        for v in range(n):
            if not visited[v] and dist[v] < min_dist:
                min_dist = dist[v]
                min_vertex = v
        
        comparisons += n  # V comparisons for finding minimum
        
        if min_vertex == -1:
            break

        visited[min_vertex] = True
        
        for v in range(n):
            if not visited[v] and graph[min_vertex][v] != float('infinity'):
                if dist[min_vertex] + graph[min_vertex][v] < dist[v]:
                    dist[v] = dist[min_vertex] + graph[min_vertex][v]
        
        comparisons += n  # V comparisons for updating neighbors
    
    return dist, comparisons

# Implementation b: adj list + heap
def dijkstra_list_heap(graph, start):
    """
    Args:
        graph: Dictionary where keys are vertices and values are lists of (neighbor, weight) tuples
        start: Starting vertex
    
    Returns:
        Tuple of (dictionary of shortest distances from start to all vertices, comparison count)
    """
    comparisons = 0
    heap_operations = 0  # count of heap operations
    edge_comparisons = 0  # count of edge relaxation comparisons
    
    # init dist dict + priority q
    dist = {vertex: float('infinity') for vertex in graph}
    dist[start] = 0
    priority_queue = [(0, start)]
    
    while priority_queue:
        # get v with min distance
        current_dist, current_vertex = heapq.heappop(priority_queue)
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
    comparisons += edge_comparisons
    
    return dist, comparisons