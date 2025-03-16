import random 
import os
import pickle as pkl
import numpy as np
import math
import pandas as pd

main_file = os.path.join(os.path.dirname(__file__))

# def generate_random_points(n):
#     points = [random.uniform(0, 1) for _ in range(n)]
#     return points
def generate_random_points(n, start=0, end=1036.8, min_dist=40, max_dist=60):
    points = []
    current_point = random.uniform(start, start + max_dist)
    points.append(current_point)
    while len(points) < n:
        step = random.uniform(min_dist, max_dist)
        next_point = points[-1] + step
        print(next_point)
        
        if next_point > end:
            break
        points.append(next_point)
    print(points)

    return points

def map_to_unit_interval(points, max_value):
    return [point / max_value for point in points]

data = {
    'total_dist': {},
    'longitude': {},
    'latitude': {}
}
data['total_dist'] = 1036.8
data['longitude'] = [
    24.85673,
    24.855898,
    24.853698,
    24.853605,
    24.853559,
    24.8535205,
    24.853499,
    24.8534829,
    24.8534572,
    24.8533031,
    24.853212,
    24.8530468,
    24.8529641,
    24.852933,
    24.85265,
    24.852488,
    24.8523532,
    24.8521998,
    24.8519978,
    24.8518524,
    24.8515883,
    24.85146,
    24.8515503,
    24.8516121,
    24.8512859,
    24.8512847,
    24.8511633,
    24.8510769,
    24.8510608,
    24.8510495,
    24.8509955,
    24.8509464,
    24.850902,
    24.8508618,
    24.850858,
    24.8507426,
    24.8506799,
    24.8506349,
    24.8506221,
    24.8505972,
    24.8505862,
    24.850568,
    24.8505083,
    24.8502303,
    24.8502053,
    24.8500988,
    24.8500149,
    24.8499571,
    24.8499205,
    24.8498451,
    24.8497922,
    24.8497678,
    24.8497283,
    24.8495676,
    24.84929,
    24.848993,
    24.8489324,
    24.8487365,
    24.8486689,
    24.8486145,
    24.8486017,
    24.8485661
]

data['latitude'] = [
    121.800161,
    121.801116,
    121.804258,
    121.804203,
    121.804206,
    121.8042687,
    121.8043151,
    121.8043563,
    121.804387,
    121.8046549,
    121.80479,
    121.8049825,
    121.8051042,
    121.8051702,
    121.804973,
    121.804969,
    121.8048866,
    121.8048336,
    121.8047297,
    121.804615,
    121.8043766,
    121.8042378,
    121.8041366,
    121.8040262,
    121.8037456,
    121.8037047,
    121.8036125,
    121.8035545,
    121.8035398,
    121.8035445,
    121.8036217,
    121.8037056,
    121.8036953,
    121.8036536,
    121.8036192,
    121.8035327,
    121.8034925,
    121.8034677,
    121.803467,
    121.8035019,
    121.8035093,
    121.8035079,
    121.8034321,
    121.8032055,
    121.8031907,
    121.8033215,
    121.8034455,
    121.8034911,
    121.8034858,
    121.8034154,
    121.8033497,
    121.8033356,
    121.8033798,
    121.8036179,
    121.80397,
    121.804471,
    121.8044849,
    121.8047924,
    121.8048819,
    121.8049791,
    121.805021,
    121.8050639
]

length = data['total_dist']
print(length)

# N = [6, 12, 18]
# for point in N:
#     random_points = generate_random_points(n = point, end = length)
#     random_points = map_to_unit_interval(random_points, length)

#     rand_path = os.path.join(main_file, 'random_storage', str(point)+'.pkl')
#     print('--------------------------------')
#     print("mapping back:", random_points)
#     print(np.asarray(random_points).shape)
#     print('--------------------------------')
#     with open(rand_path, 'wb') as f:
#         pkl.dump(random_points, f)

all_point_cnt = 20
random_points = generate_random_points(n = all_point_cnt, end = length)
random_points = map_to_unit_interval(random_points, length)
# random_points = np.asarray(random_points)

N = [6, 12, 18]
for point in N:
    random.sample(random_points, point)

    rand_path = os.path.join(main_file, 'random_storage', str(point)+'.pkl')
    print('--------------------------------')
    print("mapping back:", random_points)
    print(np.asarray(random_points).shape)
    print('--------------------------------')
    with open(rand_path, 'wb') as f:
        pkl.dump(random_points, f)
