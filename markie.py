import robolink as rl    # RoboDK API
import robodk as rdk     # Robot toolbox
import numpy as np
import math as math
import time

""" ---------------------------------------------- """
""" ---------------- Setup Robot ----------------- """
""" ---------------------------------------------- """
RDK = rl.Robolink()

robot = RDK.Item('UR5')
world_frame = RDK.Item('UR5 Base')
target = RDK.Item('Home')   # existing target in station
robot.setPoseFrame(world_frame)
robot.setPoseTool(robot.PoseTool())

""" ---------------------------------------------- """
""" ------------- Define Functions --------------- """
""" ---------------------------------------------- """
#region Functions
def testMove(transform):
    robot.MoveJ(target, blocking=True)
    RDK.RunProgram("Grinder Tool Attach (Tool Stand)", True)
    robot.MoveJ(rdk.Mat(transform.tolist()))
    RDK.RunProgram("Grinder Tool Detach (Tool Stand)", True)

def normalise(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])
#endregion

""" ---------------------------------------------- """
""" ------------- Define variables --------------- """
""" ---------------------------------------------- """
xDir = np.array([1, 0, 0])
yDir = np.array([0, 1, 0])
zDir = np.array([0, 0, 1])

# Directly use the RDK Matrix object from to hold pose (its an HT)
T_home = rdk.Mat([[ 0.000000,     0.000000,     1.000000,   523.370000 ],
                  [-1.000000,     0.000000,     0.000000,  -109.000000 ],
                  [-0.000000,    -1.000000,     0.000000,   607.850000 ],
                  [ 0.000000,     0.000000,     0.000000,     1.000000 ]])

# ------------- Tamp Stand --------------- #
#region Tamp Stand
ts_ref_g    = np.array([600.1, 52.8, 254.5])
ts_pointX_g = np.array([582.5, 128.9, 236.0])
ts_pointY_g = np.array([678.4, 70.7, 250.5])

ts_xDir_g = normalise(ts_ref_g - ts_pointX_g)
ts_yDir_g = normalise(ts_pointY_g - ts_ref_g)

# Z Rotation
ts_cAlpha = np.cos(-76 * np.pi/180)
ts_sAlpha = np.sin(-76 * np.pi/180)
ts_zRot = np.array([[ts_cAlpha, -1 * ts_sAlpha, 0],
                    [ts_sAlpha,      ts_cAlpha, 0],
                    [        0,              0, 1]])

# Y Rotation
ts_cBeta = np.cos(-6.3 * np.pi/180)
ts_sBeta = np.sin(-6.3 * np.pi/180)
ts_yRot = np.array([[     ts_cBeta, 0, ts_sBeta],
                    [            0, 1,        0], 
                    [-1 * ts_sBeta, 0, ts_cBeta]])

# X Rotation
ts_cGamma = np.cos(-10 * np.pi/180)
ts_sGamma = np.sin(-10 * np.pi/180)
ts_xRot = np.array([[1,         0,              0],
                    [0, ts_cGamma, -1 * ts_sGamma],
                    [0, ts_sGamma,      ts_cGamma]])

ts_Rot = np.matmul(np.matmul(ts_zRot, ts_yRot), ts_xRot)

T_tamp_stand = np.append(np.c_[ts_Rot.tolist(), ts_ref_g.transpose().tolist()], np.array([[0, 0, 0, 1]]))
#endregion

# ------------- Cup Stand --------------- #
#region Cup Stand
cs_ref_g = np.array([-1.5, -600.8, -20])

T_cup_stand = np.array([[-1.000000,     0.000000,     0.000000,    cs_ref_g[0]],
                        [ 0.000000,    -1.000000,     0.000000,    cs_ref_g[1]],
                        [ 0.000000,     0.000000,     1.000000,    cs_ref_g[2]],
                        [ 0.000000,     0.000000,     0.000000,       1.000000]])
#endregion

# ----------- Coffee Grinder ------------- #
#region Coffee Grinder
cg_point_g = np.array([370.1, -322.3, 317.3])
cg_ref_g   = np.array([482.7, -434.3, 317.3])
cg_diff    = normalise(cg_point_g - cg_ref_g)

cg_sTheta = np.dot(cg_diff, yDir)
cg_cTheta = -1 * np.dot(cg_diff, yDir)

T_coffee_grinder = np.array([[ cg_cTheta, -1 * cg_sTheta,     0.000000,   cg_ref_g[0] ],
                      [ cg_sTheta,      cg_cTheta,     0.000000,   cg_ref_g[1] ],
                      [  0.000000,       0.000000,     1.000000,   cg_ref_g[2] ],
                      [  0.000000,       0.000000,     0.000000,      1.000000 ]])

# Grinder button 1 transform
T_grinder_but_1 = np.array([[ 1.000000,  0.000000,     0.000000,  -64.420000],
                            [ 0.000000,  0.000000,    -1.000000,   84.820000],
                            [ 0.000000,  1.000000,     0.000000, -227.680000],
                            [ 0.000000,  0.000000,     0.000000,    1.000000]])
# Grinder button 1 standoff
T_grinder_but_1_so = np.array([[ 1.000000,  0.000000,     0.000000,  -64.420000],
                               [ 0.000000,  0.000000,    -1.000000,   100.0000],
                               [ 0.000000,  1.000000,     0.000000, -227.680000],
                               [ 0.000000,  0.000000,     0.000000,    1.000000]])

# Grinder button 2 transform
T_grinder_but_2 = np.array([[ 1.000000,  0.000000,     0.000000,  -83.710000],
                            [ 0.000000,  0.000000,    -1.000000,   89.820000],
                            [ 0.000000,  1.000000,     0.000000, -227.680000],
                            [ 0.000000,  0.000000,     0.000000,    1.000000]])
# Grinder button 2 standoff
T_grinder_but_2_so = np.array([[ 1.000000,  0.000000,     0.000000,  -80.710000],
                               [ 0.000000,  0.000000,    -1.000000,   110.0000],
                               [ 0.000000,  1.000000,     0.000000, -227.680000],
                               [ 0.000000,  0.000000,     0.000000,    1.000000]])


def coffee_grinder_routine():

    grinder_but_so_angles = np.array([-60.590000, -154.320000, -38.940000, -166.720000, 167.520000, 50.000000])
    grinder_but_intermediate_angles = np.array([-61.780000, -105.740000, -53.830000, -134.870000, 120.500000, -78.640000])
    
    T_but_1_so = T_coffee_grinder @ T_grinder_but_1_so @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_grinder_tool)
    T_but_2_so = T_coffee_grinder @ T_grinder_but_2_so @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_grinder_tool)
    T_but_1_push = T_coffee_grinder @ T_grinder_but_1 @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_grinder_tool)
    T_but_2_push = T_coffee_grinder @ T_grinder_but_2 @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_grinder_tool)

    # Routine
    # robot.MoveJ(target, blocking=True)
    RDK.RunProgram("Grinder Tool Attach (Tool Stand)", True)
    robot.MoveJ(rdk.Mat(grinder_but_intermediate_angles), blocking=True)
    robot.MoveJ(rdk.Mat(grinder_but_so_angles), blocking=True)
    robot.MoveJ(rdk.Mat(T_but_1_so.tolist()), blocking=True)
    time.sleep(1)
    robot.MoveL(rdk.Mat(T_but_1_push.tolist()), blocking=True)
    time.sleep(3)
    robot.MoveJ(rdk.Mat(grinder_but_so_angles), blocking=True)
    robot.MoveJ(rdk.Mat(T_but_2_so.tolist()), blocking=True)
    time.sleep(1)
    robot.MoveL(rdk.Mat(T_but_2_push.tolist()), blocking=True)
    time.sleep(3)
    robot.MoveJ(rdk.Mat(grinder_but_intermediate_angles), blocking=True)
    RDK.RunProgram("Grinder Tool Detach (Tool Stand)", True)
    # robot.MoveJ(target, blocking=True)
#endregion

# ------------- Coffee Machine --------------- #
#region Coffee Machine
cm_point_g = np.array([-580.4, -444.7, 350.6])
cm_ref_g   = np.array([-368.4, -389.0, 350.6])
cm_diff    = normalise(cm_point_g - cm_ref_g)

cm_cTheta = np.dot(cm_diff, yDir)
cm_sTheta = -1 * np.dot(cm_diff, xDir)

cm_theta = 50/180 * np.pi

T_coffee_machine = np.array([[ cm_cTheta, -1 * cm_sTheta,     0.000000,   cm_ref_g[0]],
                      [ cm_sTheta,      cm_cTheta,     0.000000,   cm_ref_g[1]],
                      [ 0.0000000,        0.000000,    1.000000,   cm_ref_g[2]],
                      [ 0.0000000,        0.000000,    0.000000,      1.000000]])

#region Button Transforms                                 
T__coffee_machine_button_off = np.array([[ 0.0,     np.cos(cm_theta),     -np.sin(cm_theta),   51],
                                 [ 0.0,     np.sin(cm_theta),     np.cos(cm_theta),  35.25],
                                 [ 1.0,     0.0,     0.0,  -30.90],
                                 [ 0.0,     0.0,     0.0,    1.0]])

T__coffee_machine_button_on = np.array([[ 0.0,     np.cos(cm_theta),     -np.sin(cm_theta),   51],
                                 [ 0.0,     np.sin(cm_theta),     np.cos(cm_theta),  35.25],
                                 [ 1.0,     0.0,     0.0,  -44.0],
                                 [ 0.0,     0.0,     0.0,    1.0]])

T_machine_button_so = np.array([[ 0.0,     np.cos(cm_theta),     -np.sin(cm_theta),   60],
                                 [ 0.0,     np.sin(cm_theta),     np.cos(cm_theta),  35.25],
                                 [ 1.0,     0.0,     0.0,  -38.0],
                                 [ 0.0,     0.0,     0.0,    1.0]])


#endregion
def coffee_machine_routine():

    coffee_machine_but_so_angles = np.array([-139.450534, -82.824468, -112.637521, -164.538011, -114.171479, 140.000000])
    coffee_machine_but_intermediate_angles = np.array([-118.810000, -61.780000, -123.560000, -179.410000, -68.910000, 75.120000])  #-136.630000, -54.650000, -142.570000, -153.640000, -71.090000, 75.120000
    
    T_but_off = T_coffee_machine @ T__coffee_machine_button_off @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_grinder_tool)
    T_but_on = T_coffee_machine @ T__coffee_machine_button_on @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_grinder_tool)
    T_but_so = T_coffee_machine @ T_machine_button_so @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_grinder_tool)


    # Routine
    # robot.MoveJ(target, blocking=True)
    RDK.RunProgram("Grinder Tool Attach (Tool Stand)", True)
    robot.MoveJ(rdk.Mat(coffee_machine_but_intermediate_angles), blocking=True)
    robot.MoveJ(rdk.Mat(coffee_machine_but_so_angles), blocking=True)
    robot.MoveJ(rdk.Mat(T_but_so.tolist()), blocking=True)
    time.sleep(4)
    robot.MoveL(rdk.Mat(T_but_off.tolist()), blocking=True)
    time.sleep(4)
    robot.MoveJ(rdk.Mat(T_but_so.tolist()), blocking=True)
    robot.MoveL(rdk.Mat(T_but_on.tolist()), blocking=True)
    time.sleep(4)
    robot.MoveJ(rdk.Mat(coffee_machine_but_intermediate_angles), blocking=True)
    RDK.RunProgram("Grinder Tool Detach (Tool Stand)", True)


#endregion

# ------------- Tool Stand --------------- #
#region Tool Stand
alpha_bt = np.arctan(1)-np.arctan(89.1/155.9)
T_tool_stand = np.array([[    np.cos(alpha_bt),     np.sin(alpha_bt),     0.000000,   -556.500000 ],
                        [-np.sin(alpha_bt),    np.cos(alpha_bt),     0.000000,  -77.400000 ],
                        [0.000000,     0.000000,     1.000000,   19.050000 ],
                        [0.000000,     0.000000,     0.000000,     1.000000 ]])

# ts_point_g = np.array([-645.7, 78.5, 19.05])
# ts_ref_g   = np.array([-556.5, -77.4, 19.05])
# ts_diff    = normalise(ts_point_g - ts_ref_g)

# ts_cTheta = np.dot(cg_diff, yDir)
# ts_sTheta = -1 * np.dot(cg_diff, xDir)

# T_machine = np.array([[    ts_diff[0],     -ts_diff[1],     0.000000,   -556.500000],
#      [ts_diff[1],    ts_diff[0],     0.000000,  -77.400000 ],
#      [0.000000,     0.000000,     1.000000,   19.050000 ],
#      [0.000000,     0.000000,     0.000000,     1.000000 ]])
#endregion

# ------------- Grinder Tool --------------- #
#region Grinder Tool
theta_gt = 50*(np.pi/180)
T_grinder_tool = np.array([[ np.cos(theta_gt),     np.sin(theta_gt),     0.000000,   0.0 ],
                           [-np.sin(theta_gt),     np.cos(theta_gt),     0.000000,  0.0 ],
                           [0.000000,    0.000000,     1.000000,   0.0 ],
                           [0.000000,     0.000000,     0.000000,     1.000000 ]])


T_push_button = np.array([[1.000000,     0.0,     0.000000,   0.0 ],
                          [0.000000,    1.0,     0.000000,  0.0 ],
                          [0.000000,    .000000,     1.000000,   102.82 ],
                          [0.000000,     0.000000,     0.000000,     1.000000 ]])
#endregion

""" ---------------------------------------------- """
""" ----------- -- Define Motion  - -------------- """
""" ---------------------------------------------- """
# ------------- Align with coffee machine buttons --------------- #
#region Align with coffee machine buttons

#endregion

# ------------- Place tool in grinder --------------- #
#region Place tool in grinder
T_grinder_place_tool_l = np.array([[ 1.000000,     0.000000,     0.000000,   157.61 ],
                                    [0.000000,     1.000000,     0.000000,  0 ],
                                    [0.000000,     0.000000,     1.000000,   -250.45 ],
                                    [0.000000,     0.000000,     0.000000,     1.000000 ]])


T_grinder_place_tool_g = np.matmul(T_coffee_grinder, T_grinder_place_tool_l)


#endregion

def main():
    
    # robot.MoveJ(target, blocking=True)
    # coffee_grinder_routine()
    # robot.MoveJ(target, blocking=True)
    coffee_machine_routine()


if __name__ == '__main__':
    
    main()
