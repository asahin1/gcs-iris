from pydrake.geometry.optimization import Iris, HPolyhedron, VPolytope, ComputePairwiseIntersections
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import networkx as nx
import cvxpy as cp
assert "GUROBI" in cp.installed_solvers()
from gcsopt import GraphOfConvexSets
from gcsopt.plot_utils import plot_2d_edge
from gcsopt.graph_problems.rounding.shortest_path import randomized_dfs

X_MIN = 0
X_MAX = 11
Y_MIN = 0
Y_MAX = 11

N_SAMPLES = 25

VISUALIZE_GCS = False

USE_ROUNDING = True
NUM_PATHS = 15   # needed if USE_ROUNDING is TRUE
MAX_TRIALS = 300 # needed if USE_ROUNDING is TRUE

def ordered(poly):
    """Sort 2D vertices counterclockwise around their centroid."""
    c = poly.mean(axis=0)
    angles = np.arctan2(poly[:, 1] - c[1], poly[:, 0] - c[0])
    return poly[np.argsort(angles)]

class Environment:
    """A 2D environment representation with box-shaped obstacles."""
    def __init__(self):
        """Initialize default environment."""
        self.domain = None
        self.obstacles = []
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None

    def set_domain(self, x_min, y_min, x_max, y_max):
        """Set domain boundaries."""
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.domain = HPolyhedron.MakeBox((x_min, y_min), (x_max, y_max))

    def add_obstacle(self, x_l, y_l, x_u, y_u):
        """Add a box-shaped obstacle."""
        self.obstacles.append(HPolyhedron.MakeBox((x_l, y_l), (x_u, y_u)))

    def generate_valid_sample(self, regions):
        """Generate a random sample (uniform) within the domain."""
        is_valid = False
        while not is_valid:
            x = np.random.uniform(self.x_min, self.x_max)
            y = np.random.uniform(self.y_min, self.y_max)
            is_valid = self.is_sample_valid((x, y), regions)
        return (x, y)

    def is_sample_valid(self, sample, regions):
        """
        Check if a sample is collision free and does not belong to any previous region.
        """
        for obstacle in self.obstacles:
            if obstacle.PointInSet(sample):
                return False
        for region in regions:
            if region.PointInSet(sample):
                return False
        return True

def generate_convex_regions(environment, n_samples):
    """Generate 'n_samples' convex regions."""
    seeds = []
    regions = []
    for i in tqdm(range(n_samples)):
        sample = environment.generate_valid_sample(regions)
        seeds.append(sample)
        region = Iris(obstacles=environment.obstacles,
                      sample = seeds[i], 
                      domain = environment.domain)
        regions.append(region.ReduceInequalities())
    return regions, seeds

def main():

    # Construct the environment
    environment = Environment()
    environment.set_domain(X_MIN, Y_MIN, X_MAX, Y_MAX)
    environment.add_obstacle(2, 6, 4, 8)
    environment.add_obstacle(6, 2, 8, 4)
    environment.add_obstacle(5, 5, 6, 6)

    regions, seeds = generate_convex_regions(environment, N_SAMPLES)

    directed = True
    G = GraphOfConvexSets(directed)

    # Add the GCS vertices
    for name, region in enumerate(regions):
        v = G.add_vertex(name)
        x = v.add_variable(2)
        v.add_constraint(region.A() @ x <= region.b()) # Vertex belongs to the region

    # Insert source and target vertices
    source = G.add_vertex("source")
    start = source.add_variable(2)
    start_coords = (1, 1)
    source.add_constraint(start == start_coords)
    target = G.add_vertex("target")
    goal = target.add_variable(2)
    goal_coords = (10, 10)
    target.add_constraint(goal == goal_coords)

    # Add the edges based on region intersections
    intersections, _ = ComputePairwiseIntersections(regions, [])
    for v1_name, v2_name in intersections:
        v1 = G.get_vertex(v1_name)
        v2 = G.get_vertex(v2_name)
        v1_var = v1.variables[0]
        v2_var = v2.variables[0]
        e = G.add_edge(v1, v2)
        e.add_constraint(regions[v1_name].A() @ v2_var <= regions[v1_name].b()) # Head of the edge belongs to tails region
        e.add_cost(cp.norm2(v1_var - v2_var))

    # Connect source and target to the rest of GCS
    for name, region in enumerate(regions):
        v = G.get_vertex(name)
        var = v.variables[0]
        if region.PointInSet(start_coords):
            e = G.add_edge(source, v)
            e.add_cost(cp.norm2(start - var))
        if region.PointInSet(goal_coords):
            e = G.add_edge(v, target)
            e.add_cost(cp.norm2(var - goal))

    if USE_ROUNDING:
        def rounding_fn(G, source, target):
            return randomized_dfs(G, source, target, NUM_PATHS, MAX_TRIALS)
        G.solve_shortest_path_with_rounding(source, target, rounding_fn, solver=cp.GUROBI)
        print("Rounding status: ", G.status)
        print("Rounding value: ", G.value)
    else:
        G.solve_shortest_path(source, target, solver=cp.GUROBI, verbose=False)
        print("Problem status:", G.status)
        print("Problem optimal value:", G.value)
        for v in G.vertices:
            x = v.variables[0]
            print(f"Variable {v.name} optimal value:", x.value)

    # Visualize GCS using graphviz (requires graphviz).
    if VISUALIZE_GCS:
        dot = G.graphviz()
        dot.view()

    fig, ax = plt.subplots()

    vertices_on_GCS_path = []
    for v in G.vertices:
        if v.binary_variable.value is not None and v.binary_variable.value > 1e-4:
            vertices_on_GCS_path.append(v.name)
            var = v.variables[0]
            ax.scatter(*var.value, fc='w', ec='k', zorder=3)
    edges_on_GCS_path = []
    for e in G.edges:
        if e.binary_variable.value is not None and e.binary_variable.value > 1e-4:
            edges_on_GCS_path.append((e.tail.name, e.head.name))
            tail = e.tail.variables[0].value
            head = e.head.variables[0].value
            ax.plot([tail[0], head[0]], [tail[1], head[1]],  color='blue')

    # For visualization
    polygons = []
    for region in regions:
        poly = ordered(VPolytope(region).vertices().T)
        polygons.append(poly)

    labeled = False
    for obstacle in environment.obstacles:
        obst_poly = ordered(VPolytope(obstacle).vertices().T)
        ax.fill(obst_poly[:,0], obst_poly[:,1], color="red", alpha=1.0, label="obstacle" if not labeled else None)
        labeled = True
        
    labeled = False
    for poly, seed in zip(polygons, seeds):
        ax.fill(poly[:,0], poly[:,1], color="green", alpha=0.5, label="IRIS region" if not labeled else None)
        ax.plot(seed[0], seed[1], "r*", markersize=12, label="seed" if not labeled else None)
        labeled = True
  
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")
    ax.legend()
    plt.show()
    

if __name__ == "__main__":
    main()
