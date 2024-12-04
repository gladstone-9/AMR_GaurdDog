import heapq
import math


class Vertex:
    def __init__(self, ID, coordinates):
        """
        Initializes a vertex with the given ID and coordinates.
        Also initializes the degree to 0 and the neighbors list as empty.
        """
        self.ID = ID  # Vertex identifier
        self.coordinates = coordinates  # Coordinates of the vertex (tuple, e.g., (x, y))
        self.Degree = 0  # Degree of the vertex, initially 0
        self.neighbors = []  # List to store neighboring vertices [neighbor id, compass direction, weight (length of path)]
        self.neighbor_paths = []    # Red pixel path to neighbor_i
    
    def add_neighbor(self, neighbor):
        """
        Adds a neighboring vertex and updates the degree of the vertex.
        The neighbor must be an instance of the Vertex class.
        """
        self.neighbors.append(neighbor)
        self.Degree += 1  # Increment degree as a new neighbor is added
    
    def __repr__(self):
        """
        Returns a string representation of the vertex for easy display.
        """
        return f"Vertex(ID={self.ID}, Coordinates={self.coordinates}, Degree={self.Degree}, Neighbors={len(self.neighbors)})"

    def print_neighbors(self):
        for neighbor in self.neighbors:
            print(f'{neighbor}\n')

def heuristic(coord1, coord2):
    """
    Calculate the heuristic (Euclidean distance) between two coordinates.

    Args:
        coord1 (tuple): Coordinates of the first vertex (x1, y1).
        coord2 (tuple): Coordinates of the second vertex (x2, y2).

    Returns:
        float: Euclidean distance between the two points.
    """
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def a_star(vertices, start_id, goal_id):
    """
    Perform the A* algorithm to find the shortest path between two vertices.

    Args:
        vertices (list): List of Vertex objects representing the graph.
        start_id (int): ID of the starting vertex.
        goal_id (int): ID of the goal vertex.

    Returns:
        tuple: Shortest path and its total cost.
    """
    # Create a lookup dictionary for vertices by ID
    id_to_vertex = {vertex.ID: vertex for vertex in vertices}

    # Priority queue for nodes to explore
    open_set = []
    heapq.heappush(open_set, (0, start_id))  # (priority, vertex ID)

    # Maps for tracking the shortest path
    came_from = {}  # To reconstruct the path
    g_score = {vertex.ID: float('inf') for vertex in vertices}
    g_score[start_id] = 0  # Cost from start to start is 0
    f_score = {vertex.ID: float('inf') for vertex in vertices}
    f_score[start_id] = heuristic(id_to_vertex[start_id].coordinates, id_to_vertex[goal_id].coordinates)

    while open_set:
        # Get the node with the lowest f_score
        _, current_id = heapq.heappop(open_set)

        # If the goal is reached, reconstruct the path
        if current_id == goal_id:
            path = []
            while current_id in came_from:
                path.append(current_id)
                current_id = came_from[current_id]
            path.append(start_id)
            path.reverse()
            return path, g_score[goal_id]

        # Explore neighbors
        current_vertex = id_to_vertex[current_id]
        for neighbor_id, _, weight in current_vertex.neighbors:
            tentative_g_score = g_score[current_id] + weight
            if tentative_g_score < g_score[neighbor_id]:
                # Update the best path to this neighbor
                came_from[neighbor_id] = current_id
                g_score[neighbor_id] = tentative_g_score
                f_score[neighbor_id] = g_score[neighbor_id] + heuristic(
                    id_to_vertex[neighbor_id].coordinates, id_to_vertex[goal_id].coordinates
                )
                # Add the neighbor to the open set if not already there
                if not any(neighbor_id == item[1] for item in open_set):
                    heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id))

    # If the goal is not reachable, return None
    return None, float('inf')



if __name__ == "__main__":
    # Define the vertices
    v0 = Vertex(0, (32, 136))
    v1 = Vertex(1, (94, 116))
    v2 = Vertex(2, (95, 97))
    v3 = Vertex(3, (95, 191))
    v4 = Vertex(4, (120, 142))
    v5 = Vertex(5, (123, 28))
    v6 = Vertex(6, (123, 67))
    v7 = Vertex(7, (159, 66))
    v8 = Vertex(8, (159, 68))
    v9 = Vertex(9, (181, 186))
    v10 = Vertex(10, (182, 28))
    v11 = Vertex(11, (183, 186))
    v12 = Vertex(12, (189, 65))
    v13 = Vertex(13, (189, 68))
    v14 = Vertex(14, (189, 72))
    v15 = Vertex(15, (198, 201))

    # Add neighbors
    v0.add_neighbor((1, 'S', 61))
    v0.add_neighbor((5, 'W', 205))
    v0.add_neighbor((3, 'E', 120))

    v1.add_neighbor((0, 'N', 61))
    v1.add_neighbor((2, 'W', 18))
    v1.add_neighbor((4, 'SE', 27))

    v2.add_neighbor((11, 'S', 184))
    v2.add_neighbor((6, 'W', 29))
    v2.add_neighbor((1, 'NE', 18))

    v3.add_neighbor((0, 'N', 120))
    v3.add_neighbor((9, 'S', 85))
    v3.add_neighbor((4, 'W', 48))

    v4.add_neighbor((9, 'S', 71))
    v4.add_neighbor((1, 'W', 27))
    v4.add_neighbor((3, 'E', 48))

    v5.add_neighbor((0, 'N', 205))
    v5.add_neighbor((10, 'S', 58))
    v5.add_neighbor((6, 'E', 38))

    v6.add_neighbor((8, 'S', 35))
    v6.add_neighbor((5, 'W', 38))
    v6.add_neighbor((2, 'NE', 29))

    v7.add_neighbor((8, 'E', 1))
    v7.add_neighbor((6, 'NE', 35))

    v8.add_neighbor((13, 'S', 29))
    v8.add_neighbor((7, 'W', 1))
    v8.add_neighbor((6, 'NW', 35))

    v9.add_neighbor((11, 'S', 1))
    v9.add_neighbor((4, 'W', 71))
    v9.add_neighbor((3, 'E', 86))

    v10.add_neighbor((5, 'N', 58))

    v11.add_neighbor((9, 'N', 1))
    v11.add_neighbor((2, 'W', 185))
    v11.add_neighbor((15, 'SE', 14))

    v12.add_neighbor((13, 'E', 2))

    v13.add_neighbor((8, 'N', 29))
    v13.add_neighbor((12, 'W', 2))
    v13.add_neighbor((14, 'E', 3))

    v14.add_neighbor((13, 'W', 3))

    v15.add_neighbor((11, 'NW', 14))

    # Create the list of vertices
    vertices = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15]
    start_id = 0  # Starting vertex
    goal_id = 15  # Goal vertex ID (B)
    path, cost = a_star(vertices, start_id, goal_id)

    if path:
        print(f"Shortest path: {path}")
        print(f"Total cost: {cost}")
    else:
        print("No path found between the given vertices.")