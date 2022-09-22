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
ts_zDir_g = normalise(np.cross(ts_xDir_g, ts_yDir_g))

T_tamp_stand = np.append(np.c_[ts_xDir_g, ts_yDir_g, ts_zDir_g, ts_ref_g.transpose()], np.array([[0, 0, 0, 1]])).reshape(4, 4)

T_tamper = np.array([[0.0,     1.0,    0.0,   -80.0],
                     [0.0,     0.0,    1.0,     0.0],
                     [1.0,     0.0,    0.0,   -55.0],
                     [0.0,     0.0,    0.0,     1.0]])
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

# Coffee Grinder Origin
T_grinder = np.array([[ cg_cTheta, -1 * cg_sTheta,     0.0,   cg_ref_g[0] ],
                      [ cg_sTheta,      cg_cTheta,     0.0,   cg_ref_g[1] ],
                      [       0.0,            0.0,     1.0,   cg_ref_g[2] ],
                      [       0.0,            0.0,     0.0,           1.0 ]])
#region Button Transforms
# Grinder button 1 transform
cg_but1_l = np.array([-64.42, 84.82, -227.68])
cg_but1_adj = np.array([0, 0, 0]) 
T_grinder_but_1 = np.array([[ 1.0,  0.0,  0.0,  cg_but1_l[0] + cg_but1_adj[0]],
                            [ 0.0,  0.0, -1.0,  cg_but1_l[1] + cg_but1_adj[1]],
                            [ 0.0,  1.0,  0.0,  cg_but1_l[2] + cg_but1_adj[2]],
                            [ 0.0,  0.0,  0.0,                       1.000000]])
# Grinder button 1 standoff
cg_but1_so_l = np.array([-64.42, 100, -227.68])
cg_but1_so_adj = np.array([0, 0, 0]) 
T_grinder_but_1_so = np.array([[ 1.0,  0.0,  0.0,  cg_but1_so_l[0] + cg_but1_so_adj[0]],
                               [ 0.0,  0.0, -1.0,  cg_but1_so_l[1] + cg_but1_so_adj[1]],
                               [ 0.0,  1.0,  0.0,  cg_but1_so_l[2] + cg_but1_so_adj[2]],
                               [ 0.0,  0.0,  0.0,                             1.000000]])

# Grinder button 2 transform
cg_but2_l = np.array([-83.71, 89.82, -227.68])
cg_but2_adj = np.array([0, 0, 0]) 
T_grinder_but_2 = np.array([[ 1.0,  0.0,   0.0,  cg_but2_l[0] + cg_but2_adj[0]],
                            [ 0.0,  0.0,  -1.0,  cg_but2_l[1] + cg_but2_adj[1]],
                            [ 0.0,  1.0,   0.0,  cg_but2_l[2] + cg_but2_adj[2]],
                            [ 0.0,  0.0,   0.0,                       1.000000]])
# Grinder button 2 standoff
cg_but2_so_l = np.array([-80.71, 110.0, -227.68])
cg_but2_so_adj = np.array([0, 0, 0]) 
T_grinder_but_2_so = np.array([[ 1.0,  0.0,   0.0,  cg_but2_so_l[0] + cg_but2_so_adj[0]],
                               [ 0.0,  0.0,  -1.0,  cg_but2_so_l[1] + cg_but2_so_adj[1]],
                               [ 0.0,  1.0,   0.0,  cg_but2_so_l[2] + cg_but2_so_adj[2]],
                               [ 0.0,  0.0,   0.0,                             1.000000]])

# Grinder latch standoff
cg_latch_so_l = np.array([-45.82, 83.8, -153.68])
cg_latch_so_adj = np.array([0, 0, 0]) 
T_grinder_latch_so = np.array([[ 0.0,   0.0,  -1.0,   cg_latch_so_l[0] + cg_latch_so_adj[0] ],
                               [ 1.0,   0.0,   0.0,   cg_latch_so_l[1] + cg_latch_so_adj[1] ],
                               [ 0.0,  -1.0,   0.0,   cg_latch_so_l[2] + cg_latch_so_adj[2] ],
                               [ 0.0,   0.0,   0.0,                               1.000000] ])

# Grinder latch
cg_latch_l = np.array([-35.82, 83.8, -153.68])
cg_latch_adj = np.array([0, 0, 0]) 
T_grinder_latch = np.array([[ 0.0,   0.0,  -1.0,   cg_latch_l[0] + cg_latch_adj[0] ],
                            [ 1.0,   0.0,   0.0,   cg_latch_l[1] + cg_latch_adj[1] ],
                            [ 0.0,  -1.0,   0.0,   cg_latch_l[2] + cg_latch_adj[2] ],
                            [ 0.0,   0.0,   0.0,                         1.000000] ])
# Grinder latch 1
cg_latch1_l = np.array([15.82, 95.8, -153.68])
cg_latch1_adj = np.array([0, 0, 0])
T_grinder_latch_1 =  np.array([[ 0.0,   0.0,  -1.0,   cg_latch1_l[0] + cg_latch1_adj[0] ],
                               [ 1.0,   0.0,   0.0,   cg_latch1_l[1] + cg_latch1_adj[1] ],
                               [ 0.0,  -1.0,   0.0,   cg_latch1_l[2] + cg_latch1_adj[2] ],
                               [ 0.0,   0.0,   0.0,                           1.000000] ])

# Grinder latch 2
cg_latch2_l = np.array([35.82, 80.8, -153.68])
cg_latch2_adj = np.array([0, 0, 0])
T_grinder_latch_2 = np.array([[ 0.0,   0.0,  -1.0,   cg_latch2_l[0] + cg_latch2_adj[0] ],
                              [ 1.0,   0.0,   0.0,   cg_latch2_l[1] + cg_latch2_adj[1] ],
                              [ 0.0,  -1.0,   0.0,   cg_latch2_l[2] + cg_latch2_adj[2] ],
                              [ 0.0,   0.0,   0.0,                         1.000000] ])
#endregion

#region Portafilter Tranforms
# Grinder base of portafilter tool
cg_pf_base_l = np.array([157.61, 0.0, -250.45])
cg_pf_base_adj = np.array([0, 0, 0]) 
T_grinder_place_pf_base_l = np.array([[ 0.0,  0.0, -1.0,   cg_pf_base_l[0] + cg_pf_base_adj[0] ],
                                      [ 0.0,  1.0,  0.0,   cg_pf_base_l[1] + cg_pf_base_adj[1] ],
                                      [ 1.0,  0.0,  0.0,   cg_pf_base_l[2] + cg_pf_base_adj[2] ],
                                      [ 0.0,  0.0,  0.0,                              1.000000 ]])
                                      
# Grinder head of portafilter tool
cg_pf_head_l = np.array([40.41, 0, -200])
cg_pf_head_adj = np.array([0, 0, 0])
T_grinder_place_pf_head_l = np.array([[ 0.0,     0.0,    -1.0,   cg_pf_head_l[0] + cg_pf_head_adj[0] ],
                                      [ 0.0,     1.0,     0.0,   cg_pf_head_l[1] + cg_pf_head_adj[0] ],
                                      [ 1.0,     0.0,     0.0,   cg_pf_head_l[2] + cg_pf_head_adj[0] ],
                                      [ 0.0,     0.0,     0.0,                              1.000000 ]])

# Grinder portafilter entrance position
cg_pf_entrance_l = np.array([180, 0, -170])
cg_pf_entrance_adj = np.array([0, 0, 0])
T_grinder_pf_entrance_l = np.array([[ 0.0,     0.0,    -1.0,  cg_pf_entrance_l[0] + cg_pf_entrance_adj[0] ],
                                    [ 0.0,     1.0,     0.0,  cg_pf_entrance_l[1] + cg_pf_entrance_adj[0] ],
                                    [ 1.0,     0.0,     0.0,  cg_pf_entrance_l[2] + cg_pf_entrance_adj[0] ],
                                    [ 0.0,     0.0,     0.0,                                          1.0 ]])

# Grinder portafilter tilt point 1
cg_pf_head_theta1 = -15 * np.pi/180
T_grinder_pf_tilt1_l = np.array([[     np.cos(cg_pf_head_theta1),     0.0,  np.sin(cg_pf_head_theta1),   0.0 ],
                                 [                           0.0,     1.0,                        0.0,   0.0 ],
                                 [-1 * np.sin(cg_pf_head_theta1),     0.0,  np.cos(cg_pf_head_theta1),   0.0 ],
                                 [                           0.0,     0.0,                        0.0,   1.0 ]])

# Grinder portafilter tilt point 2
cg_pf_head_theta2 = -10 * np.pi/180
T_grinder_pf_tilt2_l = np.array([[     np.cos(cg_pf_head_theta2),     0.0,  np.sin(cg_pf_head_theta2),   0.0 ],
                                 [                           0.0,     1.0,                        0.0,   0.0 ],
                                 [-1 * np.sin(cg_pf_head_theta2),     0.0,  np.cos(cg_pf_head_theta2),   0.0 ],
                                 [                           0.0,     0.0,                        0.0,   1.0 ]])

# Grinder portafilter intermediate circular point
cg_offset_dist = np.array([0, 0, -10])
T_grinder_pf_int_wo_l   = np.array([[ 1.0,     0.0,     0.0,   cg_offset_dist[0] ],
                                    [ 0.0,     1.0,     0.0,   cg_offset_dist[1] ],
                                    [ 0.0,     0.0,     1.0,   cg_offset_dist[2] ],
                                    [ 0.0,     0.0,     0.0,                 1.0 ]])
T_grinder_pf_tilt2_wo_l = T_grinder_pf_int_wo_l @ T_grinder_pf_tilt2_l
#endregion

#region Routines
def coffee_grinder_routine():

    grinder_but_so_angles = np.array([-60.590000, -154.320000, -38.940000, -166.720000, 167.520000, 50.000000])
    grinder_but_intermediate_angles = np.array([-61.780000, -105.740000, -53.830000, -134.870000, 120.500000, -78.640000])
    
    T_but_1_so = T_grinder @ T_grinder_but_1_so @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_tool_rot)
    T_but_2_so = T_grinder @ T_grinder_but_2_so @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_tool_rot)
    T_but_1_push = T_grinder @ T_grinder_but_1 @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_tool_rot)
    T_but_2_push = T_grinder @ T_grinder_but_2 @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_tool_rot)

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

def coffee_grinder_portafilter_transforms():
    transforms = {}

    transforms["pose_grinder_pf_pickup_transition"] = np.array([-71.520000, -67.360000, -104.480000, -99.970000, 8.080000, -10.480000])
    transforms["pose_grinder_pf_entrance_transition"] = np.array([-2.087809, -76.812134, -154.002466, -118.968245, -47.681912, 133.081316])
    transforms["pose_grinder_pf_drop_off_transition"] = np.array([-16.170000, -100.110000, -148.810000, -101.540000, -60.630000, 135.160000])

    transforms["T_grinder_place_pf_entrance"] = T_grinder @ T_grinder_pf_entrance_l @ np.linalg.inv(T_pf_head) @ np.linalg.inv(T_tool_rot)
    transforms["T_grinder_place_pf_tilt1"] = T_grinder @ T_grinder_place_pf_head_l @ T_grinder_pf_tilt1_l @ np.linalg.inv(T_pf_head) @ np.linalg.inv(T_tool_rot)
    transforms["T_grinder_place_pf_tilt2_wo"] = T_grinder @ T_grinder_place_pf_head_l @ T_grinder_pf_tilt2_wo_l @ np.linalg.inv(T_pf_head) @ np.linalg.inv(T_tool_rot)
    transforms["T_grinder_place_tool_final"] = T_grinder @  T_grinder_place_pf_base_l @ np.linalg.inv(T_pf_base) @ np.linalg.inv(T_tool_rot)

    return transforms

def coffee_grinder_place_portafilter_routine():
    transforms = coffee_grinder_portafilter_transforms()

    robot.MoveJ(T_home, blocking=True)
    RDK.RunProgram("Portafilter Tool Attach (Tool Stand)", True)
    time.sleep(1)
    robot.MoveJ(rdk.Mat(transforms["pose_grinder_pf_pickup_transition"]))
    robot.MoveJ(rdk.Mat(transforms["pose_grinder_pf_entrance_transition"]))
    robot.MoveJ(rdk.Mat(transforms["T_grinder_place_pf_entrance"].tolist()))
    time.sleep(1)
    robot.MoveL(rdk.Mat(transforms["T_grinder_place_pf_tilt1"].tolist()))
    time.sleep(1)
    robot.MoveC(rdk.Mat(transforms["T_grinder_place_pf_tilt2_wo"].tolist()), rdk.Mat(transforms["T_grinder_place_tool_final"].tolist()))
    time.sleep(1)
    RDK.RunProgram("Portafilter Tool Detach (Grinder)", True)
    robot.MoveJ(rdk.Mat(transforms["pose_grinder_pf_drop_off_transition"]))
    time.sleep(1)
    robot.MoveJ(T_home, blocking=True)

def coffee_grinder_pickup_portafilter_routine():
    transforms = coffee_grinder_portafilter_transforms()

    robot.MoveJ(T_home, blocking=True)
    robot.MoveJ(rdk.Mat(transforms["pose_grinder_pf_drop_off_transition"]))
    time.sleep(1)
    RDK.RunProgram("Portafilter Tool Attach (Grinder)", True)
    time.sleep(1)
    robot.MoveC(rdk.Mat(transforms["T_grinder_place_pf_tilt2_wo"].tolist()), rdk.Mat(transforms["T_grinder_place_pf_tilt1"].tolist()))
    time.sleep(1)
    robot.MoveL(rdk.Mat(transforms["T_grinder_place_pf_entrance"].tolist()))

def coffee_grinder_latch_routine():

    latch_align = np.array([-43.291130, -113.536843, -100.838648, -145.624508, -88.295703, -130.000000])
    latch_intermediate_angles = np.array([-74.850000, -95.050000, -84.210000, -129.810000, -3.940000, -147.750000]) 

    T_latch_pos_so = T_grinder @ T_grinder_latch_so @ np.linalg.inv(T_pully_bit) @ np.linalg.inv(T_tool_rot)
    T_latch_pos = T_grinder @ T_grinder_latch @ np.linalg.inv(T_pully_bit) @ np.linalg.inv(T_tool_rot)
    T_latch_pos_1 = T_grinder @ T_grinder_latch_1 @ np.linalg.inv(T_pully_bit) @ np.linalg.inv(T_tool_rot)
    T_latch_pos_2 = T_grinder @ T_grinder_latch_2 @ np.linalg.inv(T_pully_bit) @ np.linalg.inv(T_tool_rot)


    # Routine
    robot.MoveJ(target, blocking=True)
    RDK.RunProgram("Grinder Tool Attach (Tool Stand)", True)
    robot.MoveJ(rdk.Mat(latch_intermediate_angles), blocking=True)
    robot.MoveJ(rdk.Mat(latch_align), blocking=True)
    robot.MoveJ(rdk.Mat((T_latch_pos_so).tolist()), blocking=True)
    robot.MoveJ(rdk.Mat((T_latch_pos).tolist()), blocking=True)
    robot.MoveC(rdk.Mat((T_latch_pos_1).tolist()), rdk.Mat((T_latch_pos_2).tolist()), blocking=True)
    time.sleep(1)
    robot.MoveC(rdk.Mat((T_latch_pos_1).tolist()), rdk.Mat((T_latch_pos).tolist()), blocking=True)
    time.sleep(1)
    robot.MoveC(rdk.Mat((T_latch_pos_1).tolist()), rdk.Mat((T_latch_pos_2).tolist()), blocking=True)
    time.sleep(1)
    robot.MoveC(rdk.Mat((T_latch_pos_1).tolist()), rdk.Mat((T_latch_pos).tolist()), blocking=True)
    time.sleep(1)
    robot.MoveC(rdk.Mat((T_latch_pos_1).tolist()), rdk.Mat((T_latch_pos_2).tolist()), blocking=True)
    time.sleep(1)
    robot.MoveC(rdk.Mat((T_latch_pos_1).tolist()), rdk.Mat((T_latch_pos).tolist()), blocking=True)
    robot.MoveJ(rdk.Mat((T_latch_pos_so).tolist()), blocking=True)
    robot.MoveJ(rdk.Mat(latch_align), blocking=True)
    robot.MoveJ(rdk.Mat(latch_intermediate_angles), blocking=True)
    RDK.RunProgram("Grinder Tool Detach (Tool Stand)", True)

#endregion
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
cm_but_off_l = np.array([51, 35.25, -30.9])
cm_but_off_adj = np.array([0, 0, 0])                                 
T__coffee_machine_button_off = np.array([[ 0.0,  np.cos(cm_theta),    -np.sin(cm_theta),  cm_but_off_l[0] + cm_but_off_adj[0] ],
                                         [ 0.0,  np.sin(cm_theta),     np.cos(cm_theta),  cm_but_off_l[1] + cm_but_off_adj[1] ],
                                         [ 1.0,               0.0,                  0.0,  cm_but_off_l[2] + cm_but_off_adj[2] ],
                                         [ 0.0,               0.0,                  0.0,                                  1.0 ]])

cm_but_on_l = np.array([51, 35.25, -44.0])
cm_but_on_adj = np.array([0, 0, 0])  
T__coffee_machine_button_on = np.array([[ 0.0,     np.cos(cm_theta),    -np.sin(cm_theta),  cm_but_on_l[0] + cm_but_on_adj[0] ],
                                        [ 0.0,     np.sin(cm_theta),     np.cos(cm_theta),  cm_but_on_l[1] + cm_but_on_adj[1] ],
                                        [ 1.0,                  0.0,                  0.0,  cm_but_on_l[2] + cm_but_on_adj[2] ],
                                        [ 0.0,                  0.0,                  0.0,                                1.0 ]])

cm_but_so_l = np.array([60.0, 35.25, -38.0])
cm_but_so_adj = np.array([0, 0, 0])
T_machine_button_so = np.array([[ 0.0,     np.cos(cm_theta),    -np.sin(cm_theta),   cm_but_so_l[0] + cm_but_so_adj[0] ],
                                [ 0.0,     np.sin(cm_theta),     np.cos(cm_theta),   cm_but_so_l[1] + cm_but_so_adj[1] ],
                                [ 1.0,                  0.0,                  0.0,   cm_but_so_l[2] + cm_but_so_adj[2] ],
                                [ 0.0,                  0.0,                  0.0,                                1.0] ])


#endregion

#region Routines
def coffee_machine_button_routine():

    coffee_machine_but_so_angles = np.array([-139.450534, -82.824468, -112.637521, -164.538011, -114.171479, 140.000000])
    coffee_machine_but_intermediate_angles = np.array([-118.810000, -61.780000, -123.560000, -179.410000, -68.910000, 75.120000])  #-136.630000, -54.650000, -142.570000, -153.640000, -71.090000, 75.120000
    
    T_but_off = T_coffee_machine @ T__coffee_machine_button_off @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_tool_rot)
    T_but_on = T_coffee_machine @ T__coffee_machine_button_on @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_tool_rot)
    T_but_so = T_coffee_machine @ T_machine_button_so @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_tool_rot)


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

def coffee_machine_portafilter_routine():

    twist_lock_align = np.array([-181.514191, -85.515077, 227.735903, -133.465121, -120.965899, 144.530984])
    twist_lock_intermediate_angles = np.array([-87.920000, -74.850000, -139.010000, -135.450000, 20.200000, -229.820000]) 

    T_head_pos = T_tool_stand @ T_twist_lock_pos @ np.linalg.inv(T_pf_head) @ np.linalg.inv(T_tool_rot)
    T_head_rot = T_tool_stand @ T_twist_lock_rotate @ np.linalg.inv(T_pf_head) @ np.linalg.inv(T_tool_rot)
    T_head_rot1 = T_tool_stand @ T_twist_lock_rotate1 @ np.linalg.inv(T_pf_head) @ np.linalg.inv(T_tool_rot)
    T_head_rot2 = T_tool_stand @ T_twist_lock_rotate2 @ np.linalg.inv(T_pf_head) @ np.linalg.inv(T_tool_rot)

    # Routine
    # robot.MoveJ(target, blocking=True)
    RDK.RunProgram("Portafilter Tool Attach (Tool Stand)", True)
    robot.MoveJ(rdk.Mat(twist_lock_intermediate_angles), blocking=True)
    # robot.MoveJ(rdk.Mat(twist_lock_align), blocking=True)
    robot.MoveJ(rdk.Mat(T_head_pos.tolist()), blocking=True)
    time.sleep(1)
    robot.MoveJ(rdk.Mat(T_head_rot.tolist()), blocking=True)
    time.sleep(1)
    robot.MoveC(rdk.Mat(T_head_rot1.tolist()),rdk.Mat(T_head_rot2.tolist()), blocking=True)
    time.sleep(1)
    robot.MoveC(rdk.Mat(T_head_rot1.tolist()),rdk.Mat(T_head_rot.tolist()), blocking=True)
    robot.MoveJ(rdk.Mat(T_head_pos.tolist()), blocking=True)
    robot.MoveJ(rdk.Mat(twist_lock_intermediate_angles), blocking=True)
    RDK.RunProgram("Portafilter Tool Detach (Tool Stand)", True)

#endregion
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

# ------------- Twist Lock Test --------------- #
#region Twist Lock Test
tl_pos_l = np.array([14.9, 64.9, 180.0])
tl_pos_adj = np.array([0, 0, 0])
T_twist_lock_pos = np.array([[ 0.0,   0.0,    -1.0,   tl_pos_l[0] + tl_pos_adj[0] ],
                             [ 0.0,   1.0,     0.0,   tl_pos_l[1] + tl_pos_adj[1] ],
                             [ 1.0,   0.0,     0.0,   tl_pos_l[2] + tl_pos_adj[2] ],
                             [ 0.0,   0.0,     0.0,                     1.000000] ])

tl_rot_l = np.array([14.9, 64.9, 201.0])
tl_rot_adj = np.array([0, 0, 0])
T_twist_lock_rotate = np.array([[ 0.0,   0.0,    -1.0,   tl_rot_l[0] + tl_rot_adj[0] ],
                                [ 0.0,   1.0,     0.0,   tl_rot_l[1] + tl_rot_adj[1] ],
                                [ 1.0,   0.0,     0.0,   tl_rot_l[2] + tl_rot_adj[2] ],
                                [ 0.0,   0.0,     0.0,                     1.000000] ])

twist_lock_alpha = -135/180 * np.pi
T_twist_lock_rotate2 = np.array([[ 0.0,     np.cos(twist_lock_alpha),     np.sin(twist_lock_alpha),   tl_rot_l[0] + tl_rot_adj[0] ],
                                 [ 0.0,    -np.sin(twist_lock_alpha),     np.cos(twist_lock_alpha),   tl_rot_l[1] + tl_rot_adj[1] ],
                                 [ 1.0,                          0.0,                          0.0,   tl_rot_l[2] + tl_rot_adj[2] ],
                                 [ 0.0,                          0.0,                          0.0,                      1.000000 ]])

twist_lock_alpha2 = -112.5/180 * np.pi
T_twist_lock_rotate1 = np.array([[ 0.0,    np.cos(twist_lock_alpha2),    np.sin(twist_lock_alpha2),   tl_rot_l[0] + tl_rot_adj[0] ],
                                 [ 0.0,   -np.sin(twist_lock_alpha2),    np.cos(twist_lock_alpha2),   tl_rot_l[1] + tl_rot_adj[1] ],
                                 [ 1.0,                          0.0,                          0.0,   tl_rot_l[2] + tl_rot_adj[2] ],
                                 [ 0.0,                          0.0,                          0.0,                      1.000000 ]])
#endregion

# ------------- General Tool --------------- #
#region General Tool
theta_gt = 50*(np.pi/180)
T_tool_rot = np.array([[  np.cos(theta_gt),     np.sin(theta_gt),     0.0,    0.0 ],
                       [ -np.sin(theta_gt),     np.cos(theta_gt),     0.0,    0.0 ],
                       [               0.0,                  0.0,     1.0,    0.0 ],
                       [               0.0,                  0.0,     0.0,    1.0 ]])
#endregion

# ------------- Grinder Tool --------------- #
#region Grinder Tool
T_push_button = np.array([[1.0,   0.0,   0.0,     0.0 ],
                          [0.0,   1.0,   0.0,     0.0 ],
                          [0.0,   0.0,   1.0,  102.82 ],
                          [0.0,   0.0,   0.0,     1.0 ]])

T_pully_bit = np.array([[1.0,   0.0,   0.0,   -45.0 ],
                        [0.0,   1.0,   0.0,     0.0 ],
                        [0.0,   0.0,   1.0,   67.06 ],
                        [0.0,   0.0,   0.0,     1.0 ]])
#endregion

# ------------- Porta Filter Tool --------------- #
#region Portafilter Tool
pf_theta = -7.5 * np.pi/180
T_pf_head     = np.array([[     np.cos(pf_theta),     0.0,  np.sin(pf_theta),     4.71 ],
                          [                  0.0,     1.0,               0.0,      0.0 ],
                          [-1 * np.sin(pf_theta),     0.0,  np.cos(pf_theta),   144.76 ],
                          [                  0.0,     0.0,               0.0,      1.0 ]])

T_pf_base     = np.array([[     np.cos(pf_theta),     0.0,  np.sin(pf_theta),    -32.0 ],
                          [                  0.0,     1.0,               0.0,      0.0 ],
                          [-1 * np.sin(pf_theta),     0.0,  np.cos(pf_theta),    27.56 ],
                          [                  0.0,     0.0,               0.0,      1.0 ]])
#endregion

""" ---------------------------------------------- """
""" ----------- -- Define Motion  - -------------- """
""" ---------------------------------------------- """
def main():
    
    # coffee_grinder_routine()
    # coffee_machine_button_routine()
    # coffee_machine_portafilter_routine()
    # coffee_grinder_latch_routine()
    # coffee_grinder_place_portafilter_routine()
    # coffee_grinder_pickup_portafilter_routine()

    pass


if __name__ == '__main__':
    main()
