import random
from collections import deque
import matplotlib.pyplot as plt
from mazelib import Maze
from mazelib.generate.Prims import Prims
import numpy as np


class MazeGenerator:
    @staticmethod
    def gen_maze(size, n_wormholes):
        """This generates a maze using mazelib as specified in the pdf"""
        maze = Maze()
        maze.generator = Prims(size // 2, size // 2)
        maze.generate()
        maze.generate_entrances(True, True)

        """This finds all the open cells ( ones that arent walls ) and stores them in alist"""
        open_cells = [(x, y) for x in range(size) for y in range(size) if maze.grid[x][y] == 0]

        """Wee security check to see if we have more wormholes than open cells"""
        if len(open_cells) < 2 * n_wormholes:
            raise ValueError("Not enough open cells for the specified number of teleport links.")

        """Randomly pick some cells, we use 2*wormholes as each wormholes needs an exit"""
        selected_cells = random.sample(open_cells, 2 * n_wormholes)

        """This gives us a list of wormhole pairs"""
        wormholes = [(selected_cells[i], selected_cells[i + 1]) for i in range(0, 2 * n_wormholes, 2)]

        return maze, wormholes


def bfs(maze, wormholes):
    """Defines our maze information such as where bfs should start and end and the directions it can take"""
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rows, cols = len(maze.grid), len(maze.grid[0])
    start, end = maze.start, maze.end
    """Defines our empty queues and lists that will be updated as we search"""
    queue = deque([(start, 0)])
    visited = {start}
    path = {start: None}
    steps_map = {start: 0}
    wormholes_map = {source: destination for source, destination in wormholes}
    wormholes_map.update({destination: source for source, destination in wormholes})

    """While were still searching """
    while queue:
        (x, y), steps = queue.popleft()

        """If we reach the end add our path and step info to these and return it"""
        if (x, y) == end:
            final_path = []
            current = end
            while current is not None:
                final_path.append(current)
                current = path[current]
            final_path.reverse()
            return steps, final_path, steps_map

        """For directions x and y visit each unvisited node and update our path and step info"""
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (maze.grid[nx][ny] == 0 or (nx, ny) == end) and (
                    nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), steps + 1))
                path[(nx, ny)] = (x, y)
                steps_map[(nx, ny)] = steps + 1

        """If we hit a wormhole thats not visited then teleport to it and makr as visited"""
        if (x, y) in wormholes_map:
            tp_x, tp_y = wormholes_map[(x, y)]
            if (tp_x, tp_y) not in visited:
                visited.add((tp_x, tp_y))
                queue.append(((tp_x, tp_y), steps + 1))
                path[(tp_x, tp_y)] = (x, y)
                steps_map[(tp_x, tp_y)] = steps + 1

    """No path found"""
    return -1, [], steps_map


def task1_render(m, path, step_count, wormholes, steps_map):
    fig, ax = plt.subplots()
    grid = np.array(m.grid, dtype=float)

    """This maps our mazes walls and open cells for matplotlib"""
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 1:
                grid[x, y] = 1

    """Changes the colour of the path and then labels it with the step number"""
    for x, y in path:
        grid[x, y] = 0.5
        ax.text(y, x, str(steps_map[(x, y)]), ha='center', va='center', color='black', fontsize=8)

    """Changes the colour of the start and ends, decided not to label them S and E as u can tell by the step"""
    start_x, start_y = m.start
    end_x, end_y = m.end
    grid[start_x, start_y] = 0.2  # Color for the start position
    grid[end_x, end_y] = 0.8  # Color for the end position

    """Colour our wormholes"""
    for (x1, y1), (x2, y2) in wormholes:
        grid[x1, y1] = 0.3
        grid[x2, y2] = 0.3

    """Plot it"""
    ax.imshow(grid, cmap='viridis', interpolation='none')
    ax.axis('off')
    plt.title(f"Steps to Goal: {step_count}")
    plt.show()


def task2_render():
    sizes = [10, 15, 20, 25, 30]
    steps_to_goal = []

    for n in sizes:
        generator = MazeGenerator()
        maze, wormholes = generator.gen_maze(n, 2)
        steps, _, _ = bfs(maze, wormholes)  # We only need the step count here
        steps_to_goal.append(steps)
        print(f"Size: {n}x{n}, Steps to Goal: {steps}")

    plt.plot(sizes, steps_to_goal, marker='o', linestyle='-')
    plt.xlabel("Maze Size (n x n)")
    plt.ylabel("Steps to Goal")
    plt.title("Effect of Maze Size on Steps to Goal")
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    # generator = MazeGenerator()
    # maze, teleports = generator.gen_maze(7, 2)
    # steps, path, steps_map = bfs(maze, teleports)
    # task1_render(maze, path, steps, teleports, steps_map)

    task2_render()
