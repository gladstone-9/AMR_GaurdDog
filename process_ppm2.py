import numpy as np
import matplotlib.pyplot as plt
import subprocess
from collections import deque       # Find vertex neighbors



# Defines
GREY_PIXEL = [127, 127, 127]        # Blocked off by walls
BLACK_PIXEL = [0, 0, 0]             # Walls
WHITE_PIXEL = [255, 255, 255]       # Free Space
RED_PIXEL = [255, 0, 0]             # Skeleton Path
BLUE_PIXEL = [0, 0, 255]            # Vertex of skeleton


class Vertex:
    def __init__(self, ID, coordinates):
        """
        Initializes a vertex with the given ID and coordinates.
        Also initializes the degree to 0 and the neighbors list as empty.
        """
        self.ID = ID  # Vertex identifier
        self.coordinates = coordinates  # Coordinates of the vertex (tuple, e.g., (x, y))
        self.Degree = 0  # Degree of the vertex, initially 0
        self.neighbors = []  # List to store neighboring vertices [neighbor id, compass direction, weight]
    
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

def analyze_and_plot_ppm_p6(file_path):
    graph = []      # An adj list of vertices
    
    with open(file_path, 'rb') as file:
        # Read magic number
        magic_number = file.readline().strip()
        if magic_number != b"P6":
            raise ValueError("Unsupported PPM format. This script supports only P6.")

        # Skip comments
        line = file.readline()
        while line.startswith(b'#'):
            line = file.readline()

        # Read dimensions and max color value
        dimensions = line.strip().split()
        width, height = int(dimensions[0]), int(dimensions[1])
        max_color = int(file.readline().strip())
        
        # Check parameters
        print(f'_____Metadata_____\nFormat: {magic_number}\nWidth: {width}\nHeight: {height}\nMax Color: {max_color}\n')

        # Validate max color value
        if max_color != 255:
            raise ValueError("Unsupported max color value. This script assumes 255.")

        # Read binary pixel data
        pixel_data = file.read()

        # Create RGB Grid
        grid = np.zeros((height, width, 3), dtype=np.uint8)
        idx = 0
        for i in range(height):
            for j in range(width):
                # Each pixel has 3 bytes: R, G, B
                grid[i, j, 0] = pixel_data[idx]     # Red
                grid[i, j, 1] = pixel_data[idx + 1] # Green
                grid[i, j, 2] = pixel_data[idx + 2] # Blue
                idx += 3
        
        # Plot PPM Image Before Vertice Update
        plt.imshow(grid)
        plt.title("PPM Image Visualization")
        plt.axis("off")  # Turn off axis labels
        plt.show()
        
        ## Occupancy Grid Processing
        # grid = cluster_removal(grid)
        grid, graph = find_vertices(grid, graph)        # Find vertices and update grid
        graph = find_neighbor_vertices(grid, graph)     # Find neighbors for each vertice       
        
        
        # Print Adjacency List
        for i in range(len(graph)):
            print(f'{graph[i]}\n')
            graph[i].print_neighbors()
        
        # Plot PPM Image After Vertice Update
        plt.imshow(grid)
        plt.title("PPM Image Visualization")
        plt.axis("off")  # Turn off axis labels
        plt.show()

def find_vertices(grid, graph):
    new_grid = np.zeros_like(grid)

    height = np.size(grid, axis=0)
    width = np.size(grid, axis=1)
    
    count_grid = np.zeros((height, width, 3), dtype=np.uint8)

    for row in range(height):
        for col in range(width):
            pixel = grid[row][col]

            # Initial count of Pixels
            if np.array_equal(pixel, RED_PIXEL):
                # 9-cell window
                row_start = max(row - 1, 0)
                row_end = min(row + 1, grid.shape[0] - 1)
                col_start = max(col - 1, 0)
                col_end = min(col + 1, grid.shape[1] - 1)
                
                window = grid[row_start:row_end + 1, col_start:col_end + 1]
                # print(window)
                
                # Count red pixels in window
                red_pixel_count = np.sum(np.all(window == RED_PIXEL, axis=2))
                
                # Adjacent vertex check
                new_grid_window = new_grid[row_start:row_end + 1, col_start:col_end + 1]
                blue_pixel_count = np.sum(np.all(new_grid_window == BLUE_PIXEL, axis=2))
                if(blue_pixel_count > 0):
                    new_grid[row][col] = RED_PIXEL
                    continue
                
                # To check for adjacent cells
                if window.shape == (3,3,3):                     # Not checking bounds of map b/c lazy
                    grid_slices = []
                    grid_slices.append(window[0, :])
                    grid_slices.append(window[2, :])
                    grid_slices.append(window[:, 0])
                    grid_slices.append(window[:, 2])
                    
                    if red_pixel_count > 3:
                        for slice in grid_slices:
                            slice_cell_count = 0
                            for cell in slice:
                                if np.array_equal(cell, RED_PIXEL):
                                    slice_cell_count += 1
                            # 3 adj red pixels
                            if slice_cell_count == 3:
                                red_pixel_count -= 2
                            # 2 adj red pixels
                            elif slice_cell_count == 2 and np.array_equal(slice[1], RED_PIXEL):
                                red_pixel_count -= 1
                    elif red_pixel_count == 3:
                        for slice in grid_slices:
                            slice_cell_count = 0
                            for cell in slice:
                                if np.array_equal(cell, RED_PIXEL):
                                    slice_cell_count += 1
                        # 2 adj red pixels
                        if slice_cell_count == 2 and np.array_equal(slice[1], RED_PIXEL):
                                red_pixel_count += 1
                
                # Vertex Check
                if red_pixel_count > 3 or red_pixel_count == 2:
                    new_grid[row][col] = BLUE_PIXEL
                    graph.append(Vertex(len(graph), (row,col)))
                else:
                    new_grid[row][col] = grid[row][col]
                    # print(f'Vertex at {row}, {col}')
                    
            else:
                count_grid[row][col] = 0
                new_grid[row][col] = grid[row][col]
    
    return new_grid, graph

# All paths should be 1 pixel wide
# Implement if needed
def cluster_removal(grid):
    pass

# A BFS approach to find vertex neighbors following the red pixel path
# Note: If the red pixel path ends (unexpetedly as in some cases) a valid neighbor will not be found
# Approach:
# - For each adjacent red pixel call a BFS to the first blue pixel
# - Reuse the visited graph for further calls.
def find_neighbor_vertices(grid, graph):
    for i in range(len(graph)):
        V = graph[i]
        
        # Compass Directions Probably messed up b/c NP array
        # Fix directions
        directions = [
            (-1, 0, "N"),
            (1, 0, "S"),
            (0, -1, "W"),
            (0, 1, "E"),
            (-1, -1, "SW"),
            (1, 1, "NE"),
            (1, -1, "SE"),
            (-1, 1, "NW")
        ]
        
        # Right now I'm passing actual vertice, should be passing the red pixels around it
        visited = [[False for _ in range(len(grid[0]))] for _ in range(len(grid))]
        for dx, dy, direc in directions:

            x = V.coordinates[0]
            y = V.coordinates[1]
            nx = x+dx
            ny = y+dy
            
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and np.array_equal(grid[nx][ny], RED_PIXEL):
                neighbor_vertice, visited = bfs(V, [nx, ny], grid, visited, direc, graph)
                
        
                if neighbor_vertice != []:
                    graph[i].add_neighbor(neighbor_vertice)   
    return graph

# For each Vertice, find neighbors along the Red paths
def is_valid_move(x, y, grid, visited, start_vertex):
    # The point is:
    # Not adjacent to the start vertex
    # Is valid in the grid
    # Is RED or BLUE
    # Has not been visited
    return (abs(x - start_vertex[0]) > 1 or abs(y - start_vertex[1]) > 1) and 0 <= x < len(grid) and 0 <= y < len(grid[0]) and (np.array_equal(grid[x][y], RED_PIXEL) or np.array_equal(grid[x][y], BLUE_PIXEL)) and not visited[x][y]

def bfs(V, start, grid, visited, direction, graph):
    start_vertex = list(V.coordinates)    
    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, 1),
        (1, -1),
        (-1, 1)
    ]
    
    q = deque()  # (row, col, distance)       Placeholder N
    arr = [start[0], start[1], 0]
    q.append(arr)
    # Visited array to mark the visited positions
    visited[start[0]][start[1]] = True
    
    while q:
        x, y, dist = q.popleft()
        
        # Check if reached a vertex
        if [x,y] != start_vertex and np.array_equal(grid[x][y], BLUE_PIXEL):
            for vertex in graph:
                if vertex.coordinates == (x,y):
                    return [vertex.ID, direction, dist], visited
        
        # Explore the 4 possible directions
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # If the move is valid and not visited, add to the queue
            if is_valid_move(nx, ny, grid, visited, start_vertex):
                visited[nx][ny] = True
                q.append((nx, ny, dist + 1))
    
    # If no path is found
    return [], visited


def create_ppm(command):
    # Create The Map (ppm) with EVG-THIN from pgm
    try:
        result = subprocess.run(
            command, 
            # check=True, 
            text=True, 
            capture_output=True
        )
        
        # Output the result (stdout and stderr)
        print("Command Output:")
        print(result.stdout)
        
        if result.stderr:
            print("Command Error Output:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:                              # Return code 1, but result was a success so removed check
        print(f"An error occurred while running the command: {e}\n")
        print(f"Return code: {e.returncode}")
        print(f"Standard Output: {e.stdout}")
        print(f"Standard Error: {e.stderr}")
    except FileNotFoundError:
        print("The command or file could not be found.")


# Test Runs with Different Maps
# command = ['./openslam_evg-thin/test', 
#            '-image-file', 'openslam_evg-thin/Maps/DIAG_floor1.pgm', 
#            '-min-distance', '12',
#            '-pruning', '0',
#            '-max-distance', '100',
#            '-robot-loc', '1144', '691']
# create_ppm(command)
# analyze_and_plot_ppm_p6("openslam_evg-thin/Maps/DIAG_floor1_skeleton.ppm")


# Test: Cumberland Map
# Turning off pruning and setting a minimum distance from obstacles helped
# Complete obstacle border helped from creating skeleton lines that are in grey
# Result: Pretty Nice Graph
cumberland_test_command = ['./openslam_evg-thin/test', 
           '-image-file', 'Maps/maps/cumberland/cumberland.pgm', 
           '-min-distance', '4',
           '-pruning', '0',
        #    '-max-distance', '100',
        #    '-robot-loc', '1144', '691'
]
create_ppm(cumberland_test_command)
analyze_and_plot_ppm_p6("Maps/maps/cumberland/cumberland_skeleton.ppm")





# Important EVG-THIN Parameters
'''
-min-distance R : Bleeds obstacles by R cells before calculating
                    skeleton.  This removes branches that come too close to
                    obstacles.

// I think its -robot-loc instead of -robot_loc lol
-robot-loc X Y : This location is used to select which skeleton is
	   valid, given complex images with multiple, disjoint
	   skeletons.  By default, the "robot" is located at the
	   center of the image.
'''



# Current problems:
# For some reason the vertex is right but the direction is wrong
# The positions are also backwards (b/c NP array)