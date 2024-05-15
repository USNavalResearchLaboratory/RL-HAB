import numpy as np
import matplotlib.pyplot as plt
import random
import math
from matplotlib.animation import FuncAnimation

# Environment class for the flow field
class FlowFieldEnv:
    def __init__(self):
        # Define the dimensions and properties of the environment
        self.WIDTH = 100
        self.HEIGHT = 100
        self.NUM_FLOW_LEVELS = 5
        self.dt = 0.1
        # Generate a random flow field for each altitude level
        self.flow_field = np.zeros((self.NUM_FLOW_LEVELS, self.WIDTH))
        for altitude in range(self.NUM_FLOW_LEVELS):
            wind_direction = np.random.choice([-1, 1])
            wind_speed = np.random.uniform(0.5, 10)
            self.flow_field[altitude, :] = wind_direction * wind_speed

    # Calculate the horizontal flow at a given position based on altitude
    def horizontal_flow(self, position):
        altitude_level = int((position[1] / self.HEIGHT) * self.NUM_FLOW_LEVELS)
        return self.flow_field[altitude_level, 0]

    # Check if a point is within the valid bounds of the environment
    def is_point_valid(self, point):
        return point[0] >= -100 and point[0] <= 100 and point[1] >= 0 and point[1] <= self.HEIGHT

# RRT path planning algorithm class
class RRT:
    def __init__(self, env, start, goal, max_iter=1000, max_distance=5, dt=0.1, altitude_move_amount=2):
        self.env = env
        self.start = start
        self.goal = goal
        self.max_iter = max_iter
        self.max_distance = max_distance
        self.dt = dt
        self.altitude_move_amount = altitude_move_amount
        self.nodes = [start]
        self.parents = [-1]

    def plan(self):
        for _ in range(self.max_iter):
            rand_point = np.array([random.uniform(-100, 100), random.uniform(0, self.env.HEIGHT)])
            nearest_node_idx = self._find_nearest_node(rand_point)
            nearest_node = self.nodes[nearest_node_idx]
            new_node = self._extend(nearest_node, rand_point)
            if np.linalg.norm(new_node - self.goal) < self.max_distance:
                return self._construct_path(new_node)
        return None

    # Extend the tree from a given node towards a target point
    def _extend(self, from_node, to_point, stay_probability=.5):
        direction = to_point - from_node
        distance = min(np.linalg.norm(direction), self.max_distance)
        horizontal_flow = self.env.horizontal_flow(from_node)

        # Check if the agent should stay at the current altitude
        if random.random() < stay_probability:  # Adjust the probability as needed
            vertical_distance = 0  # Stay at current altitude
        else:
            vertical_distance = min(abs(to_point[1] - from_node[1]), self.altitude_move_amount) * np.sign(
                to_point[1] - from_node[1])

        new_node = np.array([from_node[0] + horizontal_flow * self.dt, from_node[1] + vertical_distance])
        if self._is_valid_edge(from_node, new_node):
            self.nodes.append(new_node)
            self.parents.append(self._find_nearest_node(from_node))
            return new_node
        return from_node

    # Find the index of the nearest node in the tree to a given point
    def _find_nearest_node(self, point):
        distances = [np.linalg.norm(node - point) for node in self.nodes]
        return np.argmin(distances)


    # Check if an edge between two nodes is valid (i.e., does not intersect obstacles)
    def _is_valid_edge(self, from_node, to_node):
        step_size = 0.1
        num_steps = int(np.linalg.norm(to_node - from_node) / step_size)
        for i in range(num_steps):
            point = from_node + (to_node - from_node) * (i / num_steps)
            if not self.env.is_point_valid(point):
                return False
        return True

    # Reconstruct the path from the end node to the start node
    def _construct_path(self, end_node):
        path = [end_node]
        current_node_idx = len(self.nodes) - 1
        while current_node_idx != 0:
            current_node = self.nodes[current_node_idx]
            parent_idx = self._find_parent(current_node_idx)
            parent_node = self.nodes[parent_idx]
            path.append(parent_node)
            current_node_idx = parent_idx
        path.append(self.start)
        return path[::-1]

    # Find the parent node index of a given node index
    def _find_parent(self, node_idx):
        return self.parents[node_idx]

class RRT_star(RRT):
    def __init__(self, env, start, goal, max_iter=1000, max_distance=5):
        super().__init__(env, start, goal, max_iter, max_distance)
        self.costs = [0]  # Cost to reach each node, initialized to 0 for start node

    def plan(self):
        for i in range(self.max_iter):
            print(i)
            rand_point = np.array([random.uniform(-100, 100), random.uniform(0, self.env.HEIGHT)])
            nearest_node_idx = self._find_nearest_node(rand_point)
            nearest_node = self.nodes[nearest_node_idx]
            new_node = self._extend(nearest_node, rand_point)
            if np.linalg.norm(new_node - self.goal) < self.max_distance:
                path = self._construct_path(new_node)
                return path
            self._rewire(new_node)
        return None

    def _extend(self, from_node, to_point, stay_probability = .5):
        direction = to_point - from_node
        distance = min(np.linalg.norm(direction), self.max_distance)
        horizontal_flow = self.env.horizontal_flow(from_node)

        # Check if the agent should stay at the current altitude
        if random.random() <stay_probability:  # Adjust the probability as needed
            vertical_distance = 0  # Stay at current altitude
        else:
            vertical_distance = min(abs(to_point[1] - from_node[1]), self.altitude_move_amount) * np.sign(
                to_point[1] - from_node[1])

        new_node = np.array([from_node[0] + horizontal_flow * self.dt, from_node[1] + vertical_distance])
        if self._is_valid_edge(from_node, new_node):
            self.nodes.append(new_node)
            self.parents.append(self._find_nearest_node(from_node))
            return new_node
        return from_node

    def _rewire(self, new_node):
        for i, node in enumerate(self.nodes):
            if node is new_node:
                continue
            if np.linalg.norm(new_node - node) < self.max_distance:
                cost = self.costs[self._find_parent(len(self.nodes) - 1)] + np.linalg.norm(new_node - node)
                if cost < self.costs[i]:
                    if self._is_valid_edge(new_node, node):
                        self.parents[i] = len(self.nodes) - 1
                        self.costs[i] = cost

    def _cost_to_node(self, node):
        return np.linalg.norm(node - self.start)

# Function to visualize the flow field and the RRT planning process
def visualize(env, rrt, animate=True, show_rrt_explored=False, show_final_path=True, smoothed_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(np.flipud(env.flow_field), cmap='coolwarm', aspect='auto', extent=[-100, 100, 0, env.HEIGHT])
    ax.plot(rrt.start[0], rrt.start[1], 'go', label='Start')
    ax.plot(rrt.goal[0], rrt.goal[1], 'bo', label='Goal')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Altitude')
    ax.set_title('RRT Path Planning in Flow Field Environment')
    #ax.invert_yaxis()

    # Add colorbar for wind speed
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Wind Speed')

    if animate:
        def update(frame):
            if frame < len(rrt.nodes):
                ax.plot(rrt.nodes[frame][0], rrt.nodes[frame][1], 'ko', markersize=1)
                if frame > 0:
                    ax.plot([rrt.nodes[frame][0], rrt.nodes[rrt._find_parent(frame)][0]],
                            [rrt.nodes[frame][1], rrt.nodes[rrt._find_parent(frame)][1]], 'k-', linewidth=0.5)
            else:
                if show_final_path:
                    if path:
                        ax.plot([point[0] for point in path], [point[1] for point in path], 'ro-', label='Original Path')
                    else:
                        ax.text(0.5, 0.5, "No path found", horizontalalignment='center',
                                verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
                if smoothed_path:
                    ax.plot([point[0] for point in smoothed_path], [point[1] for point in smoothed_path], 'go-', label='Smoothed Path')
                ani.event_source.stop()

        ani = FuncAnimation(fig, update, frames=len(rrt.nodes) + 1, interval=1)
        plt.show()
    else:
        if show_rrt_explored:
            for i in range(1, len(rrt.nodes)):
                ax.plot([rrt.nodes[i][0], rrt.nodes[rrt._find_parent(i)][0]],
                        [rrt.nodes[i][1], rrt.nodes[rrt._find_parent(i)][1]], 'k-', linewidth=0.5)
                ax.plot(rrt.nodes[i][0], rrt.nodes[i][1], 'ko', markersize=1)

        if show_final_path:
            if path:
                ax.plot([point[0] for point in path], [point[1] for point in path], 'ro-', label='Original Path')
            else:
                ax.text(0.5, 0.5, "No path found", horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')

        if smoothed_path:
            ax.plot([point[0] for point in smoothed_path], [point[1] for point in smoothed_path], color='cyan', label='Smoothed Path')

        plt.show()


def smooth_path(path, env, max_iter=1000, alpha=0.5):
    smoothed_path = path.copy()

    for _ in range(max_iter):
        new_path = smoothed_path.copy()
        for i in range(1, len(smoothed_path) - 1):
            new_path[i] = smoothed_path[i] + alpha * (smoothed_path[i - 1] - 2 * smoothed_path[i] + smoothed_path[i + 1])

        if is_path_valid(new_path, env):
            smoothed_path = new_path

    return smoothed_path

def is_path_valid(path, env):
    for point in path:
        if not env.is_point_valid(point):
            return False
    return True

# Main function to create the environment, start and goal positions, and run the RRT algorithm
if __name__ == "__main__":
    env = FlowFieldEnv()
    start = np.array([random.uniform(-80, 80), random.uniform(20, 80)])
    goal = np.array([random.uniform(-80, 80), random.uniform(20, 80)])
    rrt = RRT(env, start, goal, dt=2, altitude_move_amount=4)
    path = rrt.plan()

    # Print the path if found, otherwise print a message
    if path is None:
        print("No path found")
    else:
        print("Path found:", path)
        # Visualize the environment, RRT exploration, and final path
        smoothed_path = smooth_path(path, env)
        visualize(env, rrt, animate=False, show_rrt_explored=True, show_final_path=True, smoothed_path=None)


