import numpy as np
from models.local_model import LocalModel
from privacy.dp_noise import add_noise
from federated.aggregation import fedavg
from planner.semantic_astar import astar

def run():
    grid = np.zeros((10,10))
    grid[3:5,3:5] = 1

    start = (0,0)
    goal = (9,9)

    # simple model training
    model = LocalModel()
    X = np.random.rand(20,4)
    y = np.random.rand(20)
    model.train(X,y)

    w_noisy, b_noisy = add_noise(model.w, model.b)

    w_global, b_global = fedavg([w_noisy],[b_noisy])

    path = astar(grid, start, goal)
    print("Path:", path)
