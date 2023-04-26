import os
import numpy as np

if __name__ == "__main__":
    my_avgs = []
    for filename in os.listdir("data/loopfinal"):
        mypath = os.path.join("data/loopfinal", filename)
        file = open(mypath, 'r')
        my_info = file.readlines()
        my_nums = []
        for line in my_info:
            my_nums.append(float(line.split(" ")[5]))
        my_nums = np.array(my_nums)
        my_avgs.append(my_nums[16])
        # print(my_info)
    print(my_avgs)
    print(np.average(my_avgs))
