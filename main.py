import numpy as np
import cv2
from utils import *
from ioc_maxent import *
import timeit

ft_maps = load_feat_maps('~/walk_feat/')

trajs = load_demo_trajs('~/walk_traj/')

model = MaxEntHIOC(trajs, ft_maps, (384,216))

model.compute_empirical_stats()

converge = 0
tic=timeit.default_timer()
while converge != 1:
    model.backward_pass()
    model.forward_pass()
    model.update_gradient()
    np.savetxt('test.out', model.theta_best, delimiter=',')
    converge = model.converged
    epoch += 1
toc=timeit.default_timer()
print "######### Time taken to converge: ", (toc - tic), " ############"
