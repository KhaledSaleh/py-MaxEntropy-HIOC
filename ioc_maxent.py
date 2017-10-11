import numpy as np
import cv2
import itertools

class MaxEntHIOC(object):

    def __init__(self, demo_trajs, feat_maps, state_dim):
        """The constructor initializes all the parameters needed for the
        MaxEnt IRL framework.

        inputs:
            demo_trajs: List of the agent demonstrated trajectories, each
            item in the list is a dict of each (x,y) pos of each trajectory.
            feat_maps:  List of the the feature maps for each trajectory, each
            item in the list is a 2D numpy array of the total number of
            features that represents the state space height and width.
            state_dim: tuple (height, width) of state space dimensionality
        """
        # minimum improvement in loglikelihood
        self.delta = 0.01
        # initial step size
        self._lambda = 0.01
        # bad parameters
        self.error = 0
        # convirgence flag
        self.converged = 0
        # minimum float32 value
        self.min_flt = np.finfo(np.float32).min
        self.flt_min_bnd = 1.17549e-38
        # loglikelihood
        self.loglikelihood = 0
        # min loglikelihood
        self.minloglikelihood = self.min_flt
        # feature maps
        self.feat_maps = feat_maps
        # state space height
        self.state_height, self.state_width = state_dim
        # demonstrated trajectories
        self.demo_trajs = demo_trajs
        # list of starting point for each trajectory
        self.start_states = [traj[0] for traj in demo_trajs]
        # list of terminal point for each trajectory
        self.end_states = [traj[-1] for traj in demo_trajs]
        # reward params
        self.theta = np.ones((feat_maps[0].shape[1], ), dtype=np.float32) * 0.5
        self.theta_best = np.zeros((feat_maps[0].shape[1], ), dtype=np.float32) * 0.5
        # state visitation distribution
        self.state_visit = np.zeros((self.state_width, self.state_height), dtype=np.float32)
        # reward function
        self.reward = [0] * len(self.demo_trajs)
        # soft value function
        self.value_f = [0] * len(self.demo_trajs)
        # policy
        self.policy = [0] * len(self.demo_trajs)
        # number of actions [3x3]
        self.num_actions = 9
        # empirical feature count
        self.feature_empirical = np.zeros((feat_maps[0].shape[1]), dtype=np.float32)
        # expected feature count
        self.feature_expected = np.zeros((feat_maps[0].shape[1]), dtype=np.float32)
        # gradient
        self.feature_gradient = np.zeros((feat_maps[0].shape[1]), dtype=np.float32)

    def step2idx(self, step):
        """Convert (x,y) step to index."""
        return step[0] + step[1] * self.state_height

    def idx2step(self, idx):
        """Convert index to (x,y) step."""
        return (idx % self.state_height, idx / self.state_width)

    def backward_pass(self):
        """The backward pass of the MaxEnt HIOC framework."""
        print "Backward Pass..."
        self.error = 0
        self.loglikelihood = 0
        for idx in range(len(self.demo_trajs)):
            self.compute_reward(idx)
            self.compute_soft_value(idx)
            self.compute_policy(idx)
            self.compute_traj_likelihood(idx)
            if self.loglikelihood <= self.min_flt:
                break
        print "LogLikelihood SUM:", self.loglikelihood

    def forward_pass(self):
        """The forward pass of the MaxEnt HIOC framework."""
        print "Forward Pass..."
        if self.error != 0 or self.loglikelihood < self.minloglikelihood:
            print "Skip."
            return
        for idx in range(len(self.demo_trajs)):
            self.compute_state_visitation_freq(idx)
            self.compute_expec_feature_count(idx)
        print "Mean expected feature count:", self.feature_expected

    def compute_empirical_stats(self):
        """Compute the mean empirical feature count.
        
        Args:
            self.demo_trajs: List of the agent demonstrated trajectories.
            self.feat_maps: List of the the feature maps for each trajectory.
        Returns:
            List of the computed 37 empirical features count.
        """
        print "Computing empirical stats..."
        for traj, fmap in itertools.izip(self.demo_trajs, self.feat_maps):
            for step in traj:
                self.feature_empirical += fmap[self.step2idx(step), :]
        self.feature_empirical = self.feature_empirical / len(self.demo_trajs)

    def compute_reward(self, idx):
        """Compute the reward function given the feature map & the updated weights.
        
        Args:
            self.feat_maps[idx]: the idx-th feature map.
            self.theta: reward parameters
        Returns:
            The computed reward for the idx-th feature map.
        """
        print "  Computing reward function for trajectory:", idx
        self.reward[idx] *= 0
        self.reward[idx] = np.dot(self.feat_maps[idx], self.theta)

    def softmax_over_actions(self, v_pos, value_func, sub_region, sub_pos):
        """Compute softmax over possible actions, i.e 8."""
        sub_region_value = sub_region[sub_pos[0], sub_pos[1]]
        minv = min(value_func[v_pos[0], v_pos[1]], sub_region_value)
        maxv = max(value_func[v_pos[0], v_pos[1]], sub_region_value)
        softmax = maxv + np.log(1.0 + np.exp(minv - maxv))
        return softmax

    def compute_soft_value(self, idx):
        """Compute the value function (softer version)."""
        if self.error:
            return
        print "  Computing soft value function for trajectory:", idx
        v_actual = np.full((self.state_width, self.state_height),
                           self.min_flt)
        v_temp = v_actual.copy()
        # number of value iterations
        n = 0
        while True:
            if n%100 == 0: 
                print "   Iteration:", n
            v_padded = cv2.copyMakeBorder(v_temp, 1, 1, 1, 1, cv2.BORDER_CONSTANT,
                                          value=float(self.min_flt))

            for col in range(v_padded.shape[1] - 2):
                for row in range(v_padded.shape[0] - 2):
                    sub_region = v_padded[row:row + 3, col:col + 3]
                    if np.amax(sub_region) == np.finfo(v_actual.dtype).min:
                        continue
                    for y in range(sub_region.shape[1]):
                        for x in range(sub_region.shape[0]):
                            if y == 1 and x == 1:
                                continue
                            v_temp[row, col] = self.softmax_over_actions((row,
                                                                          col),
                                                                         v_temp,
                                                                         sub_region,
                                                                         (y, x))
                    # asyncronus updates
                    v_temp[row, col] += self.reward[idx][self.step2idx((col, row)), ]
                    if v_temp[row, col] > 0:
                        self.error = 1
                        return
            # reset goal value to 0
            v_temp[self.end_states[idx][1], self.end_states[idx][0]] = 0.0
            # convergence criteria
            residual = cv2.absdiff(v_temp, v_actual)
            v_actual = v_temp.copy()
            if np.amax(residual) < 0.9:
                break
            n += 1
            if n > 1000:
                print "ERROR: Max number of iterations."
                self.error = 1
                return
        self.value_f[idx] = v_temp

    def compute_policy(self, idx):
        """Compute the policy."""
        if self.error:
            return
        print "  Computing policy for trajectory:", idx
        policy_actual = np.zeros((self.state_width, self.state_height,
                                  self.num_actions))
        v_padded = cv2.copyMakeBorder(self.value_f[idx], 1, 1, 1, 1,
                                      cv2.BORDER_CONSTANT, value=-np.inf)
        for col in range(v_padded.shape[1] - 2):
            for row in range(v_padded.shape[0] - 2):
                sub_region = v_padded[row:row + 3, col:col + 3]
                maxv = np.amax(sub_region)
                # log rescaling
                policy_temp = sub_region - maxv
                # Z(x,a) - probability space
                policy_temp = np.exp(policy_temp, policy_temp)
                # zero out center
                policy_temp[1, 1] = 0.0
                # sum (denominator)
                denom = np.sum(policy_temp)
                if denom > 0.0:
                    # normalize (compute policy(x|a))
                    policy_temp = policy_temp / denom
                else:
                    # uniform distribution
                    policy_temp[...] = 1.0 / (self.num_actions - 1.0)
                # update policy
                policy_actual[row, col, :] = policy_temp.ravel()
        self.policy[idx] = policy_actual

    def compute_traj_likelihood(self, idx):
        """Compute trajectory likelhood."""
        print "  Computing likelihood for trajectory :", idx
        ll_val = 0.0
        traj = self.demo_trajs[idx]
        for ix, step in enumerate(traj[:-1]):
            dx = traj[(ix + 1)][0] - step[0]
            dy = traj[(ix + 1)][1] - step[1]
            action = -1
            # check if action was NW
            if dx == -1 and dy == -1:
                action = 0
            # check if action was N
            elif dx == 0 and dy == -1:
                action = 1
            # check if action was NE
            elif dx == 1 and dy == -1:
                action = 2
            # check if action was W
            elif dx == -1 and dy == 0:
                action = 3
            # check if action was E
            elif dx == 1 and dy == 0:
                action = 5
            # check if action was SW
            elif dx == -1 and dy == 1:
                action = 6
            # check if action was S
            elif dx == 0 and dy == 1:
                action = 7
            # check if action was SE
            elif dx == 1 and dy == 1:
                action = 8
            # otherwise (i.e stopping), don't account for it
            else:
                action = -1
            if action < 0:
                print "ERROR: Invalid action", ix, dx, dy
                print "Preprocess trajectory data properly"
                return
            log_val = np.log(self.policy[idx][step[1], step[0], action])
            if log_val < np.finfo(np.float32).min:
                ll_val = np.finfo(np.float32).min
                break
            ll_val += log_val
        print "    loglikelihood: ", ll_val
        self.loglikelihood += ll_val

    def compute_state_visitation_freq(self, idx):
        """Compute state visitation frequency."""
        print "  Computing State Visitation Frequency for traj:", idx
        # current state visitation distribution
        state_visit_curr = np.zeros(self.state_visit.shape)
        # running calculated state visitation distribution
        state_visit_run = np.zeros(self.state_visit.shape)
        # start and end states for trajectory[idx]
        start_x, start_y = self.start_states[idx]
        end_x, end_y = self.end_states[idx]
        # initialize start state
        state_visit_curr[start_y, start_x] = 1.0
        self.state_visit = self.state_visit * 0.0
        self.state_visit = self.state_visit + state_visit_curr
        # number of iterations
        n = 0
        while True:
            if n%100 ==0:
                print "   Iteration:", n
            state_visit_run = state_visit_run * 0.0
            for col in range(state_visit_curr.shape[1]):
                for row in range(state_visit_curr.shape[0]):
                    # absorption state
                    if row == end_y and col == end_x:
                        continue
                    # ignore small probabilities
                    if state_visit_curr[row, col] > self.flt_min_bnd:
                        num_cols = state_visit_run.shape[1] - 1
                        num_rows = state_visit_run.shape[0] - 1
                        # action NW
                        if col > 0 and row > 0:
                            state_visit_run[(row - 1), (col - 1)] += (state_visit_curr[row, col] * self.policy[idx][row, col, 0])
                        # action N
                        if row > 0:
                            state_visit_run[(row - 1), (col - 0)] += (state_visit_curr[row, col] * self.policy[idx][row, col, 1])
                        # action NE
                        if col < num_cols and row > 0:
                            state_visit_run[(row - 1), (col + 1)] += (state_visit_curr[row, col] * self.policy[idx][row, col, 2])
                        # action W
                        if col > 0:
                            state_visit_run[(row - 0), (col - 1)] += (state_visit_curr[row, col] * self.policy[idx][row, col, 3])
                        # action E
                        if col < num_cols:
                            state_visit_run[(row - 0), (col + 1)] += (state_visit_curr[row, col] * self.policy[idx][row, col, 5])
                        # action SW
                        if col > 0 and row < num_rows:
                            state_visit_run[(row + 1), (col - 1)] += (state_visit_curr[row, col] * self.policy[idx][row, col, 6])
                        # action S
                        if row < num_rows:
                            state_visit_run[(row + 1), (col - 0)] += (state_visit_curr[row, col] * self.policy[idx][row, col, 7])
                        # action SE
                        if col < num_cols and row < num_rows:
                            state_visit_run[(row + 1), (col + 1)] += (state_visit_curr[row, col] * self.policy[idx][row, col, 8])
            # absorption state
            state_visit_run[end_y, end_x] = 0.0
            # swap the two distributions
            state_visit_curr, state_visit_run = state_visit_run, state_visit_curr
            # update state visitation distribution
            self.state_visit = self.state_visit + state_visit_curr
            n += 1
            if n > 301:
                break

    def compute_expec_feature_count(self, idx):
        """Compute accumlated expected feature count."""
        print "Computing Accumlated Exepected Feature Count"
        self.feature_expected += (self.feat_maps[idx].T.dot(self.state_visit.flatten()) /
                                  len(self.demo_trajs))

    def update_gradient(self):
        """Update weights based on gradients."""
        print "  Gradients updating..."
        if self.error:
            print "  ERROR. Increase step size."
            self.theta *= 2.0
        train_improve = self.loglikelihood - self.minloglikelihood
        if train_improve > self.delta:
            self.minloglikelihood = self.loglikelihood
        elif train_improve < self.delta and train_improve >= 0.0:
            train_improve = 0
        print "Improved by:", train_improve
        # update parameters (standard line search)
        if train_improve < 0.0:
            print "  ===> NO IMPROVEMENT: decrease step size and redo."
            self._lambda *= 0.5
            self.theta = self.theta_best * np.exp(self._lambda * self.feature_gradient)
        elif train_improve > 0:
            print "  ***> IMPROVEMENT: increase step size."
            self.theta_best = self.theta
            self._lambda *= 2.0
            self.feature_gradient = self.feature_empirical - self.feature_expected
            self.theta = self.theta_best * np.exp(self._lambda * self.feature_gradient)
        elif train_improve == 0:
            print "  CONVERGED."
            self.converged = 1
        print " _lambda: ", self._lambda
