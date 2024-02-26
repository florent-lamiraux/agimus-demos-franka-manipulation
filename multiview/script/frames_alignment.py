import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares


class FrameAlignmentPbe:

    def __init__(self, T_wc_lst, T_bc_lst, T_bw0) -> None:
        self.T_wc_lst = T_wc_lst
        self.T_bc_lst = T_bc_lst
        self.T_cb_lst = [T.inverse() for T in self.T_bc_lst]

        self.N = len(self.T_wc_lst)

        # Initial guess for relative transform
        self.T_bw0 = pin.SE3(T_bw0)

    def func_se3_res(self, x: np.ndarray):
        T_bw = self.T_bw0*pin.exp6(x)
        res = np.zeros(6*self.N)

        for i in range(self.N):
            T_wc = self.T_wc_lst[i]
            T_cb = self.T_cb_lst[i]
            T_bb_delta = T_bw * T_wc * T_cb  # identiy if perfect data and optimum
            res[6*i:6*(i+1)] = pin.log(T_bb_delta).vector

        return res

    def func_r3_res(self, x: np.ndarray):
        """
        x: 6d array, local se3 tangent space delta so that
        T_bw = T_bw0*Exp6(x)
        """
        T_bw = self.T_bw0*pin.exp6(x)
        res = np.zeros(3*self.N)

        for i in range(self.N):
            T_wc = self.T_wc_lst[i]
            T_bc = self.T_bc_lst[i]
            res[3*i:3*(i+1)] = T_bc.translation - T_bw * T_wc.translation

        return res

def align_robot_fk_cosy_multi(T_WC, T_BC, use_r3=True):
    """
    T_WC: world (cosy multi) camera poses
    T_WC: robot (Forward Kinematics) camera poses
    """

    T_wc_lst = [pin.SE3(T) for T in T_WC]
    T_bc_lst = [pin.SE3(T) for T in T_BC]
    assert len(T_wc_lst) == len(T_bc_lst)

    # Initial guess
    T_bw0 = T_bc_lst[0] * T_wc_lst[0].inverse()
    pbe = FrameAlignmentPbe(T_wc_lst, T_bc_lst, T_bw0)

    if use_r3:
        res_size = 3
        res_fun = pbe.func_r3_res
    else:
        res_size = 6
        res_fun = pbe.func_se3_res

    x0 = np.zeros(6)
    result = least_squares(fun=res_fun, x0=x0, jac='2-point', method='trf', verbose=2, xtol=1e-10, loss='linear')
    # result = least_squares(fun=res_fun, x0=x0, jac='2-point', method='trf', verbose=2, xtol=1e-10, loss='soft_l1')
    T_wb_est = pbe.T_bw0 * pin.exp6(result.x)
    np.set_printoptions(precision=4)
    print('===========')
    print('Frame alignement T_WB result')
    print(T_wb_est)
    print('Residuals')
    print(result.fun.reshape((len(T_wc_lst),res_size)))
    print('message')
    print(result.message)
    print('cost')
    print(result.cost)
    print('success')
    print(result.success)
    
    return T_wb_est



if __name__ == '__main__':
    N = 10
    T_bw_gt = pin.SE3.Random()
    T_WC = [pin.SE3.Random() for _ in range(N)]
    T_BC = [T_bw_gt*T_bc for T_bc in T_WC]

    T_bw_est_r3 = align_robot_fk_cosy_multi(T_WC, T_BC, use_r3=True)
    T_bw_est_se3 = align_robot_fk_cosy_multi(T_WC, T_BC, use_r3=False)

    print('T_bw_gt')
    print(T_bw_gt)
    print('T_bw_est_r3')
    print(T_bw_est_r3)
    T_diff_r3 = T_bw_gt * T_bw_est_r3.inverse()
    T_diff_se3 = T_bw_gt * T_bw_est_se3.inverse()
    print('T_diff_r3')
    print(T_diff_r3)
    print('T_diff_se3')
    print(T_diff_se3)

