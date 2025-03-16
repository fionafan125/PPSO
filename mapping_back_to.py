import os
import matplotlib.pyplot as plt
import seaborn
import pickle as pkl
import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd
import src
import re
from src import *
'''
    "earned_money": earned_money,
    "final_ans": final_ans,
    "record_cost":fitnessCurve
'''

all_data = {
    'spend': [],
    'earned_money':[],
    'pole':[],
    'machine':[],
    'final_ans':[],
    'final_ans_2d':[]
}

def get_maximum(earned_money, machine_amount):
    earned_money, machine_amount = np.asarray(earned_money), np.asarray(machine_amount)
    idx = np.argmax(earned_money)
    max_earned_money, max_machine = earned_money[idx], machine_amount[idx]


    return max_earned_money, max_machine, idx
    

pattern_machine = r'machine_amount_(\d+)'
pattern_pole = r'wire_pole_(\d+)'
pattern_spend = r'money_([\d\.eE]+)'

main_file = os.path.join(os.path.dirname(__file__),'result_file')
file_name = 'storage_name.pkl'
#original settings

for money in os.listdir(main_file):
    money_folder = os.path.join(main_file, money)

    for wire_pole in os.listdir(money_folder):
        wire_pole_folder = os.path.join(money_folder, wire_pole)

        earned_money = []
        machine_amount = []
        final_ans_stack = []
        for machine in os.listdir(wire_pole_folder):
            read_folder = os.path.join(wire_pole_folder, machine, file_name)

            with open(read_folder, 'rb') as f:
                data = pkl.load(f)

            earned_money.append(data['earned_money'])
            final_ans_stack.append(np.asarray(data['final_ans']))

            match_machine = re.search(pattern_machine, machine)
            machine_amount_value = int(match_machine.group(1))
            machine_amount.append(machine_amount_value)

            

        max_earned_money, max_machine, idx = get_maximum(earned_money, machine_amount)
        all_data['earned_money'].append(max_earned_money)
        all_data['machine'].append(max_machine)
        all_data['final_ans'].append(final_ans_stack[idx])


        match_pole = re.search(pattern_pole, wire_pole)
        pole_amount_value = int(match_pole.group(1))
        all_data['pole'].append(pole_amount_value)

        
        match_spend = re.search(pattern_spend, money)
        spend_value = match_spend.group(1)
        all_data['spend'].append(spend_value)
        

#讀取原本excel
#原本的值mapping 回去變成點位

store_final_data = 'store_final_data.pkl'
#==settings =================================
#假設發電15年 設5瓦

store_all_folder = 'store_all'
if not os.path.exists(store_all_folder):
    os.mkdir(store_all_folder)

pkl_file_storage = os.path.join(store_all_folder, store_final_data)
record_cost = []
KW_volumn = 5.1
#===gain


meters_per_degree_latitude = 111000 #經度換算公尺

#with open('point.xlsx', 'rb') as f:
 #   data = pd.read_excel(f)

data = {
    'total_dist': {},
    'longitude': {},
    'latitude': {}
}
data['total_dist'] = [1036.8]
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

actual_length = data['total_dist'][0]  # [0]
#===================================================

points = np.vstack((data['longitude'], data['latitude'])).T #原始座標
dist_points_meters = src.get_compared_dist(points, meters_per_degree_latitude) #相對座標
accumulated_points = src.get_acc_distance(dist_points_meters) #累計相對座標
actual_length = data['total_dist'][0]

Straight_distance, accumulated_straight_dist = src.get_straight_dist(dist_points_meters, actual_length) #直線距離，累積距離





# 打印每个键对应的值
for idx in range(0, len(all_data['earned_money'])):
    point_back = src.map_back_to_curve(all_data['final_ans'][idx], points, actual_length)
    
    all_data['final_ans_2d'].append(point_back)
    print("++++++++++++++++++++++++++++++++")
    print("================================")

    for key in all_data:
        print(f"{key}: {all_data[key][idx]}")

    print("================================")
    print("++++++++++++++++++++++++++++++++")
    print("")


with open(pkl_file_storage, 'wb') as f:
    pkl.dump(all_data, f)

#===open rand=============
N = [6, 12, 18]
for wire_rand in N:
    with open(os.path.join('./random_storage', str(wire_rand) + '.pkl'), 'rb') as f:
        rand_point = pkl.load(f)
    rand_point = np.asarray(rand_point)
    rand_point = map_points_to_range(rand_point, actual_length)
    print("================================")
    print(wire_rand)
    for idx, rand_pt in enumerate(rand_point):
        if rand_pt < 16.723:
            rand_point[idx] = 16.724
    print("================================")
    rand_point2D = map_back_to_curve(rand_point, points, actual_length)
    print(rand_point2D)
    with open(os.path.join(store_all_folder, 'rand_' + str(wire_rand) + 'pkl'), 'wb') as f:
        pkl.dump(wire_rand, f)