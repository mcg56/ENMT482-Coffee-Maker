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
def normalise(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

""" ---------------------------------------------- """
""" ------------- Define variables --------------- """
""" ---------------------------------------------- """
# Define unit vector directions
xDir = np.array([1, 0, 0])
yDir = np.array([0, 1, 0])
zDir = np.array([0, 0, 1])

# Directly use the RDK Matrix object from to hold pose (its an HT)
T_home = rdk.Mat([[ 0.000000,     0.000000,     1.000000,   523.370000 ],
                  [-1.000000,     0.000000,     0.000000,  -109.000000 ],
                  [-0.000000,    -1.000000,     0.000000,   607.850000 ],
                  [ 0.000000,     0.000000,     0.000000,     1.000000 ]])

""" ---------------------------------------------- """
""" ----------------- Tamp Stand ----------------- """
""" ---------------------------------------------- """
#region Tamp Stand
ts_ref_g    = np.array([600.1, 52.8, 254.5])            # Frame origin
ts_pointX_g = np.array([582.5, 128.9, 236.0])           # Reference point in the x-direction
ts_pointY_g = np.array([678.4, 70.7, 250.5])            # Reference point in the y-direction

ts_xDir_g = normalise(ts_ref_g - ts_pointX_g)           # X-direction unit vector in the UR5 reference frame
ts_yDir_g = normalise(ts_pointY_g - ts_ref_g)           # Y-direction unit vector in the UR5 reference frame
ts_zDir_g = normalise(np.cross(ts_xDir_g, ts_yDir_g))   # Z-direction unit vector in the UR5 reference frame

# Transformation matrix from the UR5 reference frame to the tamp stand reference frame 
T_tamp_stand = (
    np.append(np.c_[ts_xDir_g, ts_yDir_g, ts_zDir_g, ts_ref_g.transpose()], 
    np.array([[0, 0, 0, 1]])).reshape(4, 4)
)

#region Transforms
ts_scraper_l = np.array([70.0, 0, -32.0])               # Scraper position in the tamp stand reference frame
ts_scraper_adj = np.array([0, 0, 0])                    # Minor adjustments for the scraper position

# Transformation matrix from the tamp stand origin to the scraper
T_tamp_stand_scraper_l = np.array([[ 0.0,  1.0,  0.0,  ts_scraper_l[0] + ts_scraper_adj[0]],
                                   [ 0.0,  0.0,  1.0,  ts_scraper_l[1] + ts_scraper_adj[1]],
                                   [ 1.0,  0.0,  0.0,  ts_scraper_l[2] + ts_scraper_adj[2]],
                                   [ 0.0,  0.0,  0.0,                             1.000000]])

ts_scraper_fwd_l = np.array([70.0, 50, -32.0])         # Forward scraper position in the tamp stand reference frame
ts_scraper_fwd_adj = np.array([0, 0, 10])              # Minor adjustments for the forward scraper position 

# Transformation matrix from the forward scraper pose to the tamp stand origin 
T_tamp_stand_scraper_fwd_l = np.array([[ 0.0,  1.0,  0.0,  ts_scraper_fwd_l[0] + ts_scraper_fwd_adj[0] ],
                                       [ 0.0,  0.0,  1.0,  ts_scraper_fwd_l[1] + ts_scraper_fwd_adj[1] ],
                                       [ 1.0,  0.0,  0.0,  ts_scraper_fwd_l[2] + ts_scraper_fwd_adj[2] ],
                                       [ 0.0,  0.0,  0.0,                                     1.000000 ]])

ts_scraper_back_l = np.array([70.0, -50, -32.0])       # Back scraper position in the tamp stand reference frame                           
ts_scraper_back_adj = np.array([0, 0, 10])             # Minor adjustments for the back scraper position 

# Transformation matrix from the backward scraper pose to the tamp stand origin 
T_tamp_stand_scraper_back_l = np.array([[ 0.0,  1.0,  0.0,  ts_scraper_back_l[0] + ts_scraper_back_adj[0] ],
                                        [ 0.0,  0.0,  1.0,  ts_scraper_back_l[1] + ts_scraper_back_adj[1] ],
                                        [ 1.0,  0.0,  0.0,  ts_scraper_back_l[2] + ts_scraper_back_adj[2] ],
                                        [ 0.0,  0.0,  0.0,                                       1.000000 ]])

ts_tamper_l = np.array([-80.0, 0, -55.0])              # Tamper position in the tamp stand reference frame
ts_tamper_adj = np.array([0, -2, 15])                  # Minor adjustments for the tamper position 

# Transformation matrix from the tamper pose to the tamp stand origin 
T_tamper_l = np.array([[ 0.0,  1.0,  0.0,  ts_tamper_l[0] + ts_tamper_adj[0] ],
                       [ 0.0,  0.0,  1.0,  ts_tamper_l[1] + ts_tamper_adj[1] ],
                       [ 1.0,  0.0,  0.0,  ts_tamper_l[2] + ts_tamper_adj[2] ],
                       [ 0.0,  0.0,  0.0,                           1.000000 ]])

ts_tamper_so_l = np.array([-80.0, 0, -75.0])           # Tamper stand-off position in the tamp stand reference frame
ts_tamper_so_adj = np.array([0, -2, 0])                # Minor adjustments for the tamper stand-off position 

# Transformation matrix from the tamper stand-off pose to the tamp stand origin 
T_tamper_so_l = np.array([[ 0.0,  1.0,  0.0,  ts_tamper_so_l[0] + ts_tamper_so_adj[0] ],
                          [ 0.0,  0.0,  1.0,  ts_tamper_so_l[1] + ts_tamper_so_adj[1] ],
                          [ 1.0,  0.0,  0.0,  ts_tamper_so_l[2] + ts_tamper_so_adj[2] ],
                          [ 0.0,  0.0,  0.0,                                 1.000000 ]])
#endregion

#region Routines
def tamp_stand_poses():
    """Define the intermediate tamp stand poses to maintain correct orientations"""
    transforms = {}

    transforms["pose_tamp_stand_so"]            = np.array([3.430000, -64.740000, -151.320000, -140.470000, -101.110000, 140.000000]) # Stand-off pose
    transforms["pose_tamp_stand_scraper_fwd"]   = np.array([3.430316, -98.748343, -116.322943, -140.475335, -101.111765, 127.533647]) # Forward scraper pose

    return transforms

def tamp_stand_scrape_and_tamp_routine():
    """Routine to tamp and scrape the portafilter"""
    poses = tamp_stand_poses()

    T_tamp_stand_scraper_fwd    = T_tamp_stand @ T_tamp_stand_scraper_fwd_l @ np.linalg.inv(T_pf_top_edge) @ np.linalg.inv(T_tool_rot)  # Global transformation matrix for forward scraper pose
    T_tamp_stand_scraper_back   = T_tamp_stand @ T_tamp_stand_scraper_back_l @ np.linalg.inv(T_pf_top_edge) @ np.linalg.inv(T_tool_rot) # Global transformation matrix for back scraper pose 
    T_tamp_stand_tamp_so        = T_tamp_stand @ T_tamper_so_l @ np.linalg.inv(T_pf_top_edge) @ np.linalg.inv(T_tool_rot)               # Global transformation matrix for the stand-off tamper pose
    T_tamp_stand_tamp           = T_tamp_stand @ T_tamper_l @ np.linalg.inv(T_pf_top_edge) @ np.linalg.inv(T_tool_rot)                  # Global transformation matrix for the tamper pose

    robot.MoveL(rdk.Mat(poses["pose_tamp_stand_so"]))        # Move to stand-off pose for the tamp stand
    robot.MoveJ(rdk.Mat(T_tamp_stand_scraper_fwd.tolist()))  # Move to the forward scraper pose
    robot.MoveL(rdk.Mat(T_tamp_stand_scraper_back.tolist())) # Move to the back scraper pose
    robot.MoveL(rdk.Mat(poses["pose_tamp_stand_so"]))        # Return to the stand-off pose 
    robot.MoveL(rdk.Mat(T_tamp_stand_tamp_so.tolist()))      # Move to the stand-off pose for the tamper
    robot.MoveL(rdk.Mat(T_tamp_stand_tamp.tolist()))         # Tamp the coffee
    robot.MoveL(rdk.Mat(T_tamp_stand_tamp_so.tolist()))      # Return to tamper stand-off pose
    robot.MoveL(rdk.Mat(poses["pose_tamp_stand_so"]))        # Return to the tamp stand stand-off pose

#endregion
#endregion

""" ---------------------------------------------- """
""" ------------------ Cup Stand ----------------- """ 
""" ---------------------------------------------- """
#region Cup Stand
cs_ref_g = np.array([-1.5, -600.8, -20]) # Frame origin

# Transformation matrix from the UR5 reference frame to the cup stand reference frame 
T_cup_stand = np.array([[-1.0,     0.0,     0.0,    cs_ref_g[0]],
                        [ 0.0,    -1.0,     0.0,    cs_ref_g[1]],
                        [ 0.0,     0.0,     1.0,    cs_ref_g[2]],
                        [ 0.0,     0.0,     0.0,       1.000000]])


cs_cup_inv = np.array([0.0, 0.0, 187.0])                    # Inverted cup position in the cup stand reference frame
cs_cup_inv_adj = np.array([-4, 0, -10])                     # Minor adjustments for the inverted cup position 
# Inverted cup location transform in cup stand reference frame
T_cup_inv = np.array([[ 0.0,  -1.0,   0.0,  cs_cup_inv[0] + cs_cup_inv_adj[0]],
                      [ 0.0,  0.0,   1.0,  cs_cup_inv[1] + cs_cup_inv_adj[1]],
                      [-1.0,  0.0,   0.0,  cs_cup_inv[2] + cs_cup_inv_adj[2]],
                      [ 0.0,  0.0,   0.0,                           1.000000]])

cs_cup_inv_so = np.array([0.0, -80.0, 187.0])               # Inverted cup approach standoff position in the cup stand reference frame
cs_cup_inv_so_adj = np.array([-4, 0, -8])                   # Minor adjustments for the inverted cup approach standoff position
# Inverted cup approach standoff location transform in cup stand reference frame
T_cup_inv_so = np.array([[ 0.0,  -1.0,   0.0,  cs_cup_inv_so[0] + cs_cup_inv_so_adj[0]],
                      [ 0.0,  0.0,   1.0,  cs_cup_inv_so[1] + cs_cup_inv_so_adj[1]],
                      [-1.0,  0.0,   0.0,  cs_cup_inv_so[2] + cs_cup_inv_so_adj[2]],
                      [ 0.0,  0.0,   0.0,                           1.000000]])

cs_cup_inv_upper_so = np.array([0.0, 0.0, 320])             # Inverted cup leave standoff position in the cup stand reference frame
cs_cup_inv_upper_so_adj = np.array([0, 0, 0])               # Minor adjustments for the inverted cup leave standoff position
# Inverted cup leave standoff location transform in cup stand reference frame
T_cup_inv_upper_so = np.array([[ 0.0,  -1.0,   0.0,  cs_cup_inv_upper_so[0] + cs_cup_inv_upper_so_adj[0]],
                      [ 0.0,  0.0,   1.0,  cs_cup_inv_upper_so[1] + cs_cup_inv_upper_so_adj[1]],
                      [-1.0,  0.0,   0.0,  cs_cup_inv_upper_so[2] + cs_cup_inv_upper_so_adj[2]],
                      [ 0.0,  0.0,   0.0,                           1.000000]])

cs_cup_top = np.array([0.0, 0.0, 297.0])                    # Cup seat position in the cup stand reference frame
cs_cup_top_adj = np.array([0, 0, 20])                       # Minor adjustments for the cup seat position
# Cup seat location transform in cup stand reference frame
T_cup_top = np.array([[ 0.0,  1.0,   0.0,  cs_cup_top[0] + cs_cup_top_adj[0]],
                      [ 0.0,  0.0,   1.0,  cs_cup_top[1] + cs_cup_top_adj[1]],
                      [ 1.0,  0.0,   0.0,  cs_cup_top[2] + cs_cup_top_adj[2]],
                      [ 0.0,  0.0,   0.0,                           1.000000]])

cs_cup_top_so = np.array([0.0, -100.0, 297.0])              # Cup seat standoff position in the cup stand reference frame
cs_cup_top_so_adj = np.array([0, 0, 20])                    # Minor adjustments for the cup seat position standoff
# Cup seat standoff location transform in cup stand reference frame
T_cup_top_so = np.array([[ 0.0,  1.0,   0.0,  cs_cup_top_so[0] + cs_cup_top_so_adj[0]],
                         [ 0.0,  0.0,   1.0,  cs_cup_top_so[1] + cs_cup_top_so_adj[1]],
                         [ 1.0,  0.0,   0.0,  cs_cup_top_so[2] + cs_cup_top_so_adj[2]],
                         [ 0.0,  0.0,   0.0,                                 1.000000]])

def cup_to_coffee_machine():
    pose_cup_stand_so = np.array([-58.080954, -77.549329, -150.216101, -132.234570, -58.080954, -40.000000])    # Stand-off pose to cup stand
    pose_cup_cm_so = np.array([-78.207197, -62.292760, -145.704363, -152.002877, -22.928142, -220.000000])      # Stand-off pose to coffee machine
    pose_cup_inter1 = np.array([-48.710000, -62.290000, -145.700000, -152.000000, -22.920000, -220.000000])     # First intermediate transition orientation
    pose_cup_inter2 = np.array([-79.600000, -70.100000, -120.580000, -127.130000, 45.150000, -220.000000])      # Second intermediate transition orientation
    pose_cup_inter3 = np.array([-79.920000, -60.970000, -100.990000, -181.780000, 55.840000, -54.290000])       # Third intermediate transition orientation
    pose_cup_inter4 = np.array([-84.360000, -60.590000, -92.670000, -165.150000, 39.210000, -111.050000])       # Fourth intermediate transition orientation

    T_cup_stand_inv = T_cup_stand @ T_cup_inv @ np.linalg.inv(T_cup_tool) @ np.linalg.inv(T_tool_rot)                                   # Global transformation matrix for inverted cup position
    T_cup_stand_inv_so = T_cup_stand @ T_cup_inv_so @ np.linalg.inv(T_cup_tool) @ np.linalg.inv(T_tool_rot)                             # Global transformation matrix for inverted cup standoff position
    T_cup_stand_inv_upper_so = T_cup_stand @ T_cup_inv_upper_so @ np.linalg.inv(T_cup_tool) @ np.linalg.inv(T_tool_rot)                 # Global transformation matrix for inverted cup upper standoff position
    T_cup_coffee_machine = T_coffee_machine @ T_coffee_machine_platform @ np.linalg.inv(T_cup_tool) @ np.linalg.inv(T_tool_rot)         # Global transformation matrix for coffee machine platform position
    T_cup_coffee_machine_so = T_coffee_machine @ T_coffee_machine_platform_so @ np.linalg.inv(T_cup_tool) @ np.linalg.inv(T_tool_rot)   # Global transformation matrix for coffee machine standoff position

    # robot.MoveJ(T_home, blocking=True)
    RDK.RunProgram("Cup Tool Attach (Stand)", True)             # Attach the cup tool
    robot.MoveJ(rdk.Mat(pose_cup_inter3), blocking=True)        # Move to the third transition pose
    robot.MoveJ(rdk.Mat(pose_cup_inter4), blocking=True)        # Move to the fourth transition pose
    robot.MoveJ(rdk.Mat(pose_cup_stand_so), blocking=True)      # Move to the cup standoff pose
    robot.MoveJ(rdk.Mat(T_cup_stand_inv_so.tolist()))           # Move to the inverted cup stand off pose
    RDK.RunProgram("Cup Tool Open", True)                       # Open cup tool
    robot.MoveL(rdk.Mat(T_cup_stand_inv.tolist()))              # Move to inverted cup position
    RDK.RunProgram("Cup Tool Close", True)                      # Close cup tool
    robot.MoveL(rdk.Mat(T_cup_stand_inv_upper_so.tolist()))     # Move to upper inverted cup standoff position
    robot.MoveJ(rdk.Mat(pose_cup_cm_so))                        # Move to coffee machine standoff joint angles
    robot.MoveL(rdk.Mat(T_cup_coffee_machine_so.tolist()))      # Move to coffee machine standoff pose
    robot.MoveL(rdk.Mat(T_cup_coffee_machine.tolist()))         # Move to coffee machine platform pose
    RDK.RunProgram("Cup Tool Open", True)                       # Open cup tool
    robot.MoveL(rdk.Mat(T_cup_coffee_machine_so.tolist()))      # Move to coffee machine standoff pose
    RDK.RunProgram("Cup Tool Close", True)                      # Close cup tool
    robot.MoveL(rdk.Mat(pose_cup_inter1))                       # Move to the first transition pose
    robot.MoveJ(rdk.Mat(pose_cup_inter2))                       # Move to the second transition pose
    RDK.RunProgram("Cup Tool Detach (Stand)", True)             # Detach the cup tool


def cup_to_stand():

    pose_cup_cm_so = np.array([-78.207197, -62.292760, -145.704363, -152.002877, -22.928142, -220.000000])          # Stand-off pose to coffee machine
    pose_cup_top_so = np.array([-54.084890, -66.164274, -149.284735, 35.449009, 54.084890, -40.000000])             # Stand-off pose to cup seat standoff
    pose_cup_top_inter = np.array([-70.100000, -72.480000, -98.610000, -162.870000, -25.280000, -216.950000])       # First intermediate transition orientation
    pose_cup_top_inter2 = np.array([-125.260000, -74.680000, -88.530000, -138.520000, 90.300000, -211.250000])      # Second intermediate transition orientation

    T_cup_stand_top = T_cup_stand @ T_cup_top @ np.linalg.inv(T_cup_tool) @ np.linalg.inv(T_tool_rot)                                   # Global transformation matrix for cup stand seat 
    T_cup_stand_top_so = T_cup_stand @ T_cup_top_so @ np.linalg.inv(T_cup_tool) @ np.linalg.inv(T_tool_rot)                             # Global transformation matrix for cup stand seat stand off
    T_cup_coffee_machine = T_coffee_machine @ T_coffee_machine_platform @ np.linalg.inv(T_cup_tool) @ np.linalg.inv(T_tool_rot)         # Global transformation matrix for coffee machine platform position
    T_cup_coffee_machine_so = T_coffee_machine @ T_coffee_machine_platform_so @ np.linalg.inv(T_cup_tool) @ np.linalg.inv(T_tool_rot)   # Global transformation matrix for coffee machine standoff position
    

    RDK.RunProgram("Cup Tool Attach (Stand)", True)             # Attach the cup tool
    robot.MoveJ(rdk.Mat(pose_cup_top_inter))                    # Move to the first transition pose
    robot.MoveL(rdk.Mat(pose_cup_top_so))                       # Move to cup seat standoff pose
    robot.MoveJ(rdk.Mat(pose_cup_cm_so))                        # Move to coffee machine standoff joint angles
    robot.MoveL(rdk.Mat(T_cup_coffee_machine_so.tolist()))      # Move to coffee machine standoff pose
    RDK.RunProgram("Cup Tool Open", True)                       # Open the cup tool
    robot.MoveL(rdk.Mat(T_cup_coffee_machine.tolist()))         # Move to coffee machine platform pose
    RDK.RunProgram("Cup Tool Close", True)                      # Clsoe the cup tool
    robot.MoveL(rdk.Mat(T_cup_coffee_machine_so.tolist()))      # Move to coffee machine standoff pose
    robot.MoveL(rdk.Mat(pose_cup_top_so))                       # Move to cup seat standoff pose
    robot.MoveL(rdk.Mat(T_cup_stand_top_so.tolist()))           # Move to cup stand standoff pose
    robot.MoveL(rdk.Mat(T_cup_stand_top.tolist()))              # Move to cup stand seat pose
    RDK.RunProgram("Cup Tool Open", True)                       # Open the cup tool
    robot.MoveL(rdk.Mat(T_cup_stand_top_so.tolist()))           # Move to cup stand standoff pose
    robot.MoveL(rdk.Mat(pose_cup_top_so))                       # Move to cup seat standoff pose
    RDK.RunProgram("Cup Tool Close", True)                      # Close the cup tool
    robot.MoveJ(rdk.Mat(pose_cup_top_inter))                    # Move to the first transition pose
    robot.MoveJ(rdk.Mat(pose_cup_top_inter2))                   # Move to the second transition pose
    RDK.RunProgram("Cup Tool Detach (Stand)", True)             # Detach the cup tool

#endregion

""" ---------------------------------------------- """
""" --------------- Coffee Grinder --------------- """
""" ---------------------------------------------- """
#region Coffee Grinder
cg_point_g = np.array([370.1, -322.3, 317.3]) # Reference point in global coordinates
cg_ref_g   = np.array([482.7, -434.3, 317.3]) # Frame origin in global coordinates
cg_diff    = normalise(cg_point_g - cg_ref_g) # Direction vector in local coordinates

cg_sTheta = np.dot(cg_diff, yDir)             # Reference frame rotation component (sin(theta))
cg_cTheta = -1 * np.dot(cg_diff, yDir)        # Reference frame rotation component (cos(theta))

# Transformation matrix from the UR5 reference frame to the coffee grinder reference frame
T_grinder = np.array([[ cg_cTheta, -1 * cg_sTheta,     0.0,   cg_ref_g[0] ],
                      [ cg_sTheta,      cg_cTheta,     0.0,   cg_ref_g[1] ],
                      [       0.0,            0.0,     1.0,   cg_ref_g[2] ],
                      [       0.0,            0.0,     0.0,           1.0 ]])


#region Button Transforms
cg_but1_l = np.array([-64.42, 84.82, -227.68]) # Position of button 1 in the coffee grinder reference frame
cg_but1_adj = np.array([0, 0, 0])              # Minor adjustments for the button 1 position 

# Transformation matrix from the coffee grinder reference frame to button 1
T_grinder_but_1 = np.array([[ 1.0,  0.0,  0.0,  cg_but1_l[0] + cg_but1_adj[0]],
                            [ 0.0,  0.0, -1.0,  cg_but1_l[1] + cg_but1_adj[1]],
                            [ 0.0,  1.0,  0.0,  cg_but1_l[2] + cg_but1_adj[2]],
                            [ 0.0,  0.0,  0.0,                       1.000000]])

cg_but1_so_l = np.array([-64.42, 100, -227.68]) # Position of the button 1 stand-off in the coffee grinder reference frame
cg_but1_so_adj = np.array([0, 0, 0])            # Minor adjustments for the button 1 stand-off position 

# Transformation matrix from the coffee grinder reference frame to the button 1 stand-off
T_grinder_but_1_so = np.array([[ 1.0,  0.0,  0.0,  cg_but1_so_l[0] + cg_but1_so_adj[0]],
                               [ 0.0,  0.0, -1.0,  cg_but1_so_l[1] + cg_but1_so_adj[1]],
                               [ 0.0,  1.0,  0.0,  cg_but1_so_l[2] + cg_but1_so_adj[2]],
                               [ 0.0,  0.0,  0.0,                             1.000000]])

# Grinder button 2 transform
cg_but2_l = np.array([-83.71, 89.82, -227.68]) # Position of button 2 in the coffee grinder reference frame
cg_but2_adj = np.array([0, 0, 0])              # Minor adjustments for the button 2 position

# Transformation matrix from the coffee grinder reference frame to button 2
T_grinder_but_2 = np.array([[ 1.0,  0.0,   0.0,  cg_but2_l[0] + cg_but2_adj[0]],
                            [ 0.0,  0.0,  -1.0,  cg_but2_l[1] + cg_but2_adj[1]],
                            [ 0.0,  1.0,   0.0,  cg_but2_l[2] + cg_but2_adj[2]],
                            [ 0.0,  0.0,   0.0,                       1.000000]])

cg_but2_so_l = np.array([-80.71, 110.0, -227.68]) # Position of the button 2 stand-off in the coffee grinder reference frame
cg_but2_so_adj = np.array([0, 0, 0])              # Minor adjustments for the button 2 stand-off position 

# Transformation matrix from the coffee grinder reference frame to the button 2 stand-off
T_grinder_but_2_so = np.array([[ 1.0,  0.0,   0.0,  cg_but2_so_l[0] + cg_but2_so_adj[0]],
                               [ 0.0,  0.0,  -1.0,  cg_but2_so_l[1] + cg_but2_so_adj[1]],
                               [ 0.0,  1.0,   0.0,  cg_but2_so_l[2] + cg_but2_so_adj[2]],
                               [ 0.0,  0.0,   0.0,                             1.000000]])

cg_latch_so_l = np.array([-45.82, 83.8, -153.68]) # Latch stand-off position in the coffee grinder reference frame
cg_latch_so_adj = np.array([0, 0, 0])             # Minor adjustments for the latch stand-off position 

# Transformation matrix from the coffee grinder reference frame to the latch stand-off pose
T_grinder_latch_so = np.array([[ 0.0,   0.0,  -1.0,   cg_latch_so_l[0] + cg_latch_so_adj[0] ],
                               [ 1.0,   0.0,   0.0,   cg_latch_so_l[1] + cg_latch_so_adj[1] ],
                               [ 0.0,  -1.0,   0.0,   cg_latch_so_l[2] + cg_latch_so_adj[2] ],
                               [ 0.0,   0.0,   0.0,                               1.000000] ])

cg_latch_l = np.array([-35.82, 80.8, -153.68])   # Latch position in the coffee grinder reference frame
cg_latch_adj = np.array([0, 0, 0])               # Minor adjustments for the latch position 

# Transformation matrix from the coffee grinder reference frame to the latch pose
T_grinder_latch = np.array([[ 0.0,   0.0,  -1.0,   cg_latch_l[0] + cg_latch_adj[0] ],
                            [ 1.0,   0.0,   0.0,   cg_latch_l[1] + cg_latch_adj[1] ],
                            [ 0.0,  -1.0,   0.0,   cg_latch_l[2] + cg_latch_adj[2] ],
                            [ 0.0,   0.0,   0.0,                         1.000000] ])

cg_latch1_l = np.array([15.82, 95.8, -153.68])  # Latch mid-point position in the coffee grinder reference frame
cg_latch1_adj = np.array([0, 0, 0])             # Minor adjustments for the latch mid-point position

# Tranformation matrix from the coffee grinder reference frame to the mid-point latch pose
T_grinder_latch_1 =  np.array([[ 0.0,   0.0,  -1.0,   cg_latch1_l[0] + cg_latch1_adj[0] ],
                               [ 1.0,   0.0,   0.0,   cg_latch1_l[1] + cg_latch1_adj[1] ],
                               [ 0.0,  -1.0,   0.0,   cg_latch1_l[2] + cg_latch1_adj[2] ],
                               [ 0.0,   0.0,   0.0,                           1.000000] ])

cg_latch2_l = np.array([65.82, 70.8, -153.68])  # Extended latch position in the coffee grinder reference frame
cg_latch2_adj = np.array([0, 0, 0])             # Minor adjustments for the extended latch position

# Tranformation matrix from the coffee grinder reference frame to the extended latch pose
T_grinder_latch_2 = np.array([[ 0.0,   0.0,  -1.0,   cg_latch2_l[0] + cg_latch2_adj[0] ],
                              [ 1.0,   0.0,   0.0,   cg_latch2_l[1] + cg_latch2_adj[1] ],
                              [ 0.0,  -1.0,   0.0,   cg_latch2_l[2] + cg_latch2_adj[2] ],
                              [ 0.0,   0.0,   0.0,                         1.000000] ])
#endregion

#region Portafilter Tranforms
cg_pf_base_l = np.array([157.61, 0.0, -250.45]) # Portafilter base position in the coffee grinder reference frame
cg_pf_base_adj = np.array([2.8, -0.25, -2.8])   # Minor adjustments for the portafilter base position

# Tranformation matrix from the coffee grinder reference frame to the portafilter base pose
T_grinder_place_pf_base_l = np.array([[ 0.0,  0.0, -1.0,   cg_pf_base_l[0] + cg_pf_base_adj[0] ],
                                      [ 0.0,  1.0,  0.0,   cg_pf_base_l[1] + cg_pf_base_adj[1] ],
                                      [ 1.0,  0.0,  0.0,   cg_pf_base_l[2] + cg_pf_base_adj[2] ],
                                      [ 0.0,  0.0,  0.0,                              1.000000 ]])
                                      
cg_pf_head_l = np.array([40.41, 0, -200])       # Portafilter head position in the coffee grinder reference frame
cg_pf_head_adj = np.array([8, 0, 0])            # Minor adjustments for the portafilter head position

# Tranformation matrix from the coffee grinder reference frame to the portafilter head pose
T_grinder_place_pf_head_l = np.array([[ 0.0,     0.0,    -1.0,   cg_pf_head_l[0] + cg_pf_head_adj[0] ],
                                      [ 0.0,     1.0,     0.0,   cg_pf_head_l[1] + cg_pf_head_adj[1] ],
                                      [ 1.0,     0.0,     0.0,   cg_pf_head_l[2] + cg_pf_head_adj[2] ],
                                      [ 0.0,     0.0,     0.0,                              1.000000 ]])

cg_pf_entrance_l = np.array([180, 0, -170])     # Portafilter entrance position in the coffee grinder reference frame
cg_pf_entrance_adj = np.array([0, 0, -25])      # Minor adjustments for the portafilter entrance position

# Tranformation matrix from the coffee grinder reference frame to the portafilter entrance pose
T_grinder_pf_entrance_l = np.array([[ 0.0,     0.0,    -1.0,  cg_pf_entrance_l[0] + cg_pf_entrance_adj[0] ],
                                    [ 0.0,     1.0,     0.0,  cg_pf_entrance_l[1] + cg_pf_entrance_adj[1] ],
                                    [ 1.0,     0.0,     0.0,  cg_pf_entrance_l[2] + cg_pf_entrance_adj[2] ],
                                    [ 0.0,     0.0,     0.0,                                          1.0 ]])

cg_pf_head_theta1 = -5 * np.pi/180              # First tilt angle for the portafilter

# Tranformation matrix to tilt the portafilter head by a specified angle
T_grinder_pf_tilt1_l = np.array([[     np.cos(cg_pf_head_theta1),     0.0,  np.sin(cg_pf_head_theta1),   0.0 ],
                                 [                           0.0,     1.0,                        0.0,   0.0 ],
                                 [-1 * np.sin(cg_pf_head_theta1),     0.0,  np.cos(cg_pf_head_theta1),   0.0 ],
                                 [                           0.0,     0.0,                        0.0,   1.0 ]])

cg_pf_head_theta2 = -2 * np.pi/180              # Second tilt angle for the portafilter

# Tranformation matrix to tilt the portafilter head by a specified angle
T_grinder_pf_tilt2_l = np.array([[     np.cos(cg_pf_head_theta2),     0.0,  np.sin(cg_pf_head_theta2),   0.0 ],
                                 [                           0.0,     1.0,                        0.0,   0.0 ],
                                 [-1 * np.sin(cg_pf_head_theta2),     0.0,  np.cos(cg_pf_head_theta2),   0.0 ],
                                 [                           0.0,     0.0,                        0.0,   1.0 ]])

#endregion

#region Routines
def coffee_grinder_button_routine():
    """Define the routine to press the buttons on the coffee grinder"""

    grinder_but_so_angles = np.array([-60.590000, -154.320000, -38.940000, -166.720000, 167.520000, 50.000000])             # Button stand-off orientation
    grinder_but_intermediate_angles = np.array([-61.780000, -105.740000, -53.830000, -134.870000, 120.500000, -78.640000])  # Intermediate transition orientation
    
    T_but_1_so = T_grinder @ T_grinder_but_1_so @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_tool_rot) # Global transformation matrix for button 1 stand-off pose
    T_but_2_so = T_grinder @ T_grinder_but_2_so @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_tool_rot) # Global transformation matrix for button 2 stand-off pose
    T_but_1_push = T_grinder @ T_grinder_but_1 @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_tool_rot)  # Global transformation matrix for button 1 pressed pose
    T_but_2_push = T_grinder @ T_grinder_but_2 @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_tool_rot)  # Global transformation matrix for button 1 pressed pose

    RDK.RunProgram("Grinder Tool Attach (Tool Stand)", True)                # Attach the grinder tool
    robot.MoveJ(rdk.Mat(grinder_but_intermediate_angles), blocking=True)    # Move to the transition pose
    robot.MoveJ(rdk.Mat(grinder_but_so_angles), blocking=True)              # Move to the stand-off button pose
    robot.MoveJ(rdk.Mat(T_but_1_so.tolist()), blocking=True)                # Move to the stand-off pose for button 1
    time.sleep(1)
    robot.MoveL(rdk.Mat(T_but_1_push.tolist()), blocking=True)              # Press button 1
    time.sleep(3)                                                           # Pause while grinding coffee
    robot.MoveJ(rdk.Mat(grinder_but_so_angles), blocking=True)              # Move to the stand-off button pose
    robot.MoveJ(rdk.Mat(T_but_2_so.tolist()), blocking=True)                # Move to the stand-off pose for button 2
    time.sleep(1)                                       
    robot.MoveL(rdk.Mat(T_but_2_push.tolist()), blocking=True)              # Press button 2
    time.sleep(3)                                                           # Pause to let grinding stop
    robot.MoveJ(rdk.Mat(grinder_but_intermediate_angles), blocking=True)    # Return to transition pose    

def coffee_grinder_portafilter_transforms():
    """Define the poses of the portafilter tool for the coffee machine"""
    transforms = {}

    transforms["pose_grinder_pf_pickup_transition"]    = np.array([-71.520000, -67.360000, -104.480000, -99.970000, 8.080000, -10.480000])     # Transition orientation
    transforms["pose_grinder_pf_entrance_transition"]  = np.array([-2.063290, -83.165226, -155.895434, -110.920679, -47.640078, 133.211859])   # Portafilter entrance orientation
    transforms["pose_grinder_pf_drop_off1_transition"] = np.array([-16.170000, -100.110000, -148.810000, -101.540000, -60.630000, -224.38000]) # First drop-off orientation
    transforms["pose_grinder_pf_drop_off2_transition"] = np.array([-39.80000, -100.110000, -148.800000, -101.540000, -60.630000, -224.380000]) # Second drop-off orienatation

    transforms["T_grinder_place_pf_entrance"] = T_grinder @ T_grinder_pf_entrance_l @ np.linalg.inv(T_pf_head) @ np.linalg.inv(T_tool_rot)     # Global transformation matrix for portafilter entrance pose
    transforms["T_grinder_place_pf_tilt1"]    = T_grinder @ T_grinder_place_pf_head_l @ T_grinder_pf_tilt1_l @ np.linalg.inv(T_pf_head) @ np.linalg.inv(T_tool_rot) # Global transformation matrix for the portafilter head pose
    transforms["T_grinder_place_tool_final"]  = T_grinder @  T_grinder_place_pf_base_l @ np.linalg.inv(T_pf_base) @ np.linalg.inv(T_tool_rot)  # Global transformation matrix for portafilter base pose

    return transforms

def coffee_grinder_place_portafilter_routine():
    """Routine to place the portafilter in the coffee grinder"""
    transforms = coffee_grinder_portafilter_transforms()

    RDK.RunProgram("Portafilter Tool Attach (Tool Stand)", True)                # Pick-up the portafilter tool
    robot.MoveJ(rdk.Mat(transforms["pose_grinder_pf_pickup_transition"]))       # Move to the transition pose
    robot.MoveJ(rdk.Mat(transforms["pose_grinder_pf_entrance_transition"]))     # Align the portafilter with the coffee grinder
    robot.MoveJ(rdk.Mat(transforms["T_grinder_place_pf_entrance"].tolist()))    # Move to the entrance pose
    robot.MoveL(rdk.Mat(transforms["T_grinder_place_pf_tilt1"].tolist()))       # Place the head of the portafilter in the coffee grinder
    robot.MoveL(rdk.Mat(transforms["T_grinder_place_tool_final"].tolist()))     # Place the base on the tool on the coffee grinder
    RDK.RunProgram("Portafilter Tool Detach (Grinder)", True)                   # Release the portafilter tool
    robot.MoveL(rdk.Mat(transforms["pose_grinder_pf_drop_off1_transition"]))    # Move to the first intermediate pose
    robot.MoveL(rdk.Mat(transforms["pose_grinder_pf_drop_off2_transition"]))    # Move to the second intermediate pose
    robot.MoveJ(rdk.Mat(transforms["pose_grinder_pf_pickup_transition"]))       # Return to the transition pose

def coffee_grinder_pickup_portafilter_routine():
    """Routine to pickup the portafilter from the coffee grinder"""
    transforms = coffee_grinder_portafilter_transforms()

    # This reverses the process during drop-off
    robot.MoveJ(rdk.Mat(transforms["pose_grinder_pf_pickup_transition"]))       # Move to the transition pose
    robot.MoveJ(rdk.Mat(transforms["pose_grinder_pf_drop_off2_transition"]))    # Move to the second intermediate pose
    robot.MoveJ(rdk.Mat(transforms["pose_grinder_pf_drop_off1_transition"]))    # Move to the first intermediate pose    
    RDK.RunProgram("Portafilter Tool Attach (Grinder)", True)                   # Attach the portafilter tool
    robot.MoveL(rdk.Mat(transforms["T_grinder_place_pf_tilt1"].tolist()))       # Tilt the base of the portafilter tool up
    robot.MoveL(rdk.Mat(transforms["T_grinder_place_pf_entrance"].tolist()))    # Move the portafilter out to the entrance pose

def coffee_grinder_latch_routine():
    """Routine to pull the coffee grinder latch"""

    latch_align = np.array([-43.291130, -113.536843, -100.838648, -145.624508, -88.295703, -130.000000])            # Latch alignment orientation
    latch_intermediate_angles = np.array([-74.850000, -95.050000, -84.210000, -129.810000, -3.940000, -147.750000]) # Transition orientation

    T_latch_pos_so = T_grinder @ T_grinder_latch_so @ np.linalg.inv(T_pully_bit) @ np.linalg.inv(T_tool_rot) # Global transformation matrix for the latch stand-off pose
    T_latch_pos = T_grinder @ T_grinder_latch @ np.linalg.inv(T_pully_bit) @ np.linalg.inv(T_tool_rot)       # Global transformation matrix for the latch pose
    T_latch_pos_1 = T_grinder @ T_grinder_latch_1 @ np.linalg.inv(T_pully_bit) @ np.linalg.inv(T_tool_rot)   # Global transformation matrix for the latch mid-point pose
    T_latch_pos_2 = T_grinder @ T_grinder_latch_2 @ np.linalg.inv(T_pully_bit) @ np.linalg.inv(T_tool_rot)   # Global transformation matrix for the latch extended pose

    robot.MoveJ(rdk.Mat(latch_intermediate_angles), blocking=True)                                      # Move to the intermediate pose
    robot.MoveJ(rdk.Mat(latch_align), blocking=True)                                                    # Align the tool with the latch
    robot.MoveJ(rdk.Mat((T_latch_pos_so).tolist()), blocking=True)                                      # Move to the latch stand-off pose
    robot.MoveJ(rdk.Mat((T_latch_pos).tolist()), blocking=True)                                         # Move the tool behind the latch
    robot.MoveC(rdk.Mat((T_latch_pos_1).tolist()), rdk.Mat((T_latch_pos_2).tolist()), blocking=True)    # Extend the latch
    time.sleep(1)                                                                                       # Pause as required
    robot.MoveC(rdk.Mat((T_latch_pos_1).tolist()), rdk.Mat((T_latch_pos).tolist()), blocking=True)      # Let the latch retract
    time.sleep(1)                                                                                       # Pause as required
    robot.MoveC(rdk.Mat((T_latch_pos_1).tolist()), rdk.Mat((T_latch_pos_2).tolist()), blocking=True)    # Extend the latch again    
    time.sleep(1)                                                                                       # Pause as required           
    robot.MoveC(rdk.Mat((T_latch_pos_1).tolist()), rdk.Mat((T_latch_pos).tolist()), blocking=True)      # Let the latch retract
    time.sleep(1)                                                                                       # Pause as required            
    robot.MoveC(rdk.Mat((T_latch_pos_1).tolist()), rdk.Mat((T_latch_pos_2).tolist()), blocking=True)    # Extend the latch again
    time.sleep(1)                                                                                       # Pause as required        
    robot.MoveC(rdk.Mat((T_latch_pos_1).tolist()), rdk.Mat((T_latch_pos).tolist()), blocking=True)      # Let the latch retract
    robot.MoveJ(rdk.Mat((T_latch_pos_so).tolist()), blocking=True)                                      # Return to the stand-off pose
    robot.MoveJ(rdk.Mat(latch_align), blocking=True)                                                    # Return to the alignment pose
    robot.MoveJ(rdk.Mat(latch_intermediate_angles), blocking=True)                                      # Return to the transition pose                                         
    RDK.RunProgram("Grinder Tool Detach (Tool Stand)", True)                                            # Detach the grinder tool

#endregion
#endregion

""" ---------------------------------------------- """
""" --------------- Coffee Machine --------------- """
""" ---------------------------------------------- """
#region Coffee Machine
cm_point_g = np.array([-580.4, -444.7, 350.6]) # Reference point in global coordinates
cm_ref_g   = np.array([-368.4, -389.0, 350.6]) # Frame origin in global coordinates
cm_diff    = normalise(cm_point_g - cm_ref_g)  # Direction vector in local coordinates

cm_cTheta = np.dot(cm_diff, yDir)              # Reference frame rotation component (cos(theta))
cm_sTheta = -1 * np.dot(cm_diff, xDir)         # Reference frame rotation component (sin(theta))

cm_theta = 50/180 * np.pi                      # 50 degree rotation to the coffee machine for button press so UR5 does not hit tool stand

# Transformation matrix from the UR5 reference frame to the coffee machine reference frame
T_coffee_machine = np.array([[ cm_cTheta, -1 * cm_sTheta,     0.000000,   cm_ref_g[0]],
                             [ cm_sTheta,      cm_cTheta,     0.000000,   cm_ref_g[1]],
                             [ 0.0000000,        0.000000,    1.000000,   cm_ref_g[2]],
                             [ 0.0000000,        0.000000,    0.000000,      1.000000]])

#region Button Transforms
cm_but_on_l = np.array([51, 35.25, -30.9])    # Local position of the off state of the button
cm_but_on_adj = np.array([-3, 0, 15])         # Minor adjustments for the button off position

# Transformation matrix from the coffee machine reference frame to the button off reference frame
T__coffee_machine_button_on = np.array([[ 0.0,  np.cos(cm_theta),    -np.sin(cm_theta),   cm_but_on_l[0] + cm_but_on_adj[0] ],
                                         [ 0.0,  np.sin(cm_theta),     np.cos(cm_theta),  cm_but_on_l[1] + cm_but_on_adj[1] ],
                                         [ 1.0,               0.0,                  0.0,  cm_but_on_l[2] + cm_but_on_adj[2] ],
                                         [ 0.0,               0.0,                  0.0,                                  1.0 ]])

cm_but_off_l = np.array([51, 35.25, -44.0])     # Local position of the on state of the button
cm_but_off_adj = np.array([-3, 0, 13])          # Minor adjustments for the button on position

# Transformation matrix from the coffee machine reference frame to the button on reference frame
T__coffee_machine_button_off = np.array([[ 0.0,     np.cos(cm_theta),    -np.sin(cm_theta), cm_but_off_l[0] + cm_but_off_adj[0] ],
                                        [ 0.0,     np.sin(cm_theta),     np.cos(cm_theta),  cm_but_off_l[1] + cm_but_off_adj[1] ],
                                        [ 1.0,                  0.0,                  0.0,  cm_but_off_l[2] + cm_but_off_adj[2] ],
                                        [ 0.0,                  0.0,                  0.0,                                1.0 ]])

cm_but_so_l = np.array([60.0, 35.25, -38.0])   # Button stand-off position in the coffee machine reference frame
cm_but_so_adj = np.array([5, 0, 15])           # Minor adjustments for the button stand-off position

# Transformation matrix from the coffee machine reference frame to the button stand-off reference frame
T_machine_button_so = np.array([[ 0.0,     np.cos(cm_theta),    -np.sin(cm_theta),   cm_but_so_l[0] + cm_but_so_adj[0] ],
                                [ 0.0,     np.sin(cm_theta),     np.cos(cm_theta),   cm_but_so_l[1] + cm_but_so_adj[1] ],
                                [ 1.0,                  0.0,                  0.0,   cm_but_so_l[2] + cm_but_so_adj[2] ],
                                [ 0.0,                  0.0,                  0.0,                                1.0] ])


cmp_theta = 20/180 * np.pi                           # 20 degree approach angle to avoid portafilter
cup_height = 80                                      # Paper cup height to the top of the lip               
cm_platform = np.array([-10, 72, -290+cup_height])   # Frame origin in coffee machine coordinates
cm_platform_adj = np.array([0, 0, 9.5])              # Frame adjustments

# Transformation matrix from the coffee machine reference frame to the cup platform pose
T_coffee_machine_platform = np.array([[ 0.0,  np.cos(cmp_theta),   -np.sin(cmp_theta),   cm_platform[0] + cm_platform_adj[0]],
                                      [ 0.0,  np.sin(cmp_theta),    np.cos(cmp_theta),   cm_platform[1] + cm_platform_adj[1]],
                                      [ 1.0,                0.0,                  0.0,   cm_platform[2] + cm_platform_adj[2]],
                                      [ 0.0,                0.0,                  0.0,                                   1.0]])

stand_off = 120     # 120 mm offset for standoff in the direction of the 20 degree angle

# Transformation matrix from the coffee machine reference frame to the cup platform standoff pose
T_coffee_machine_platform_so = np.array([[ 0.0,  np.cos(cmp_theta),   -np.sin(cmp_theta),   cm_platform[0] + stand_off*np.sin(cmp_theta)],
                                         [ 0.0,  np.sin(cmp_theta),    np.cos(cmp_theta),   cm_platform[1] - stand_off*np.cos(cmp_theta)],
                                         [ 1.0,                0.0,                  0.0,   cm_platform[2] + cm_platform_adj[2]         ],
                                         [ 0.0,                0.0,                  0.0,                                            1.0]])


#endregion

#region Routines
def coffee_machine_button_routine():
    """Routine to press the coffee machine buttons"""

    coffee_machine_but_so_angles = np.array([-140.247036, -82.198223, -110.912461, -166.889316, -114.967982, 140.000000])           # Stand-off orientation
    coffee_machine_but_intermediate_angles = np.array([-118.810000, -61.780000, -123.560000, -179.410000, -68.910000, 75.120000])   # Transition orientation
    
    T_but_on = T_coffee_machine @ T__coffee_machine_button_on @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_tool_rot)            # Global transformation matrix for the button off pose
    T_but_off = T_coffee_machine @ T__coffee_machine_button_off @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_tool_rot)          # Global transformation matrix for the button on pose
    T_but_so = T_coffee_machine @ T_machine_button_so @ np.linalg.inv(T_push_button) @ np.linalg.inv(T_tool_rot)                    # Global transformation matrix for the button stand-off pose


    RDK.RunProgram("Grinder Tool Attach (Tool Stand)", True)                        # Attach the grinder tool
    robot.MoveJ(rdk.Mat(coffee_machine_but_intermediate_angles), blocking=True)     # Move to the transition pose
    robot.MoveJ(rdk.Mat(coffee_machine_but_so_angles), blocking=True)               # Move to the button stand-off pose
    robot.MoveJ(rdk.Mat(T_but_so.tolist()), blocking=True)                          # Move to the button stand-off pose
    robot.MoveL(rdk.Mat(T_but_on.tolist()), blocking=True)                          # Turn the switch on 
    robot.MoveJ(rdk.Mat(T_but_so.tolist()), blocking=True)                          # Return to the stand-off pose
    time.sleep(10)                                                                  # Pause as required
    robot.MoveL(rdk.Mat(T_but_off.tolist()), blocking=True)                         # Turn the switch off
    robot.MoveJ(rdk.Mat(T_but_so.tolist()), blocking=True)                          # Return to the stand-off pose
    robot.MoveJ(rdk.Mat(coffee_machine_but_intermediate_angles), blocking=True)     # Return to the transition pose
    RDK.RunProgram("Grinder Tool Detach (Tool Stand)", True)                        # Detach the grinder tool

#endregion

# ------------- MoveL --------------- #
def coffee_machine_portafilter_routine(): 

    twist_lock_intermediate_angles = np.array([-87.920000, -74.850000, -139.010000, -135.450000, 20.200000, -229.820000])       # Transition orientation

    T_head_pos = T_tool_stand @ T_twist_lock_pos @ np.linalg.inv(T_pf_head) @ np.linalg.inv(T_tool_rot)                         # Global transformation matrix for the twist lock unlocked position
    T_head_rot = T_tool_stand @ T_twist_lock_rotate @ np.linalg.inv(T_pf_head) @ np.linalg.inv(T_tool_rot)                      # Global transformation matrix for the twist lock standoff

    # Routine
    RDK.RunProgram("Portafilter Tool Attach (Tool Stand)", True)            # Attach the portafilter tool
    robot.MoveJ(rdk.Mat(twist_lock_intermediate_angles), blocking=True)     # Move to the transition pose
    robot.MoveJ(rdk.Mat(T_head_rot.tolist()), blocking=True)                # Move to the twist lock stand-off pose
    robot.MoveJ(rdk.Mat(T_head_pos.tolist()), blocking=True)                # Move to the twist lock unlocked position

    # Calcuate global tranformation matrix for 5 interpolated poses to lock the portafilter tool
    tl_rot_l = np.array([14.9, 64.9, 214.0])                                # Twist lock position in global coordinates
    tl_rot_adj = np.array([0, 0, -5])                                       # Twist lock position in global coordinates adjustment

    for i in range(90, 135+1, 9):
        twist_lock_alphai = -i/180 * np.pi                                  # Angle in radians for every 9 degree increment

        # Transformation matrix for intermediate pose in tool stand frame
        T_twist_lock_rotatei = np.array([[ 0.0,    np.cos(twist_lock_alphai),    np.sin(twist_lock_alphai),   tl_rot_l[0] + tl_rot_adj[0] ],
                                         [ 0.0,   -np.sin(twist_lock_alphai),    np.cos(twist_lock_alphai),   tl_rot_l[1] + tl_rot_adj[1] ],
                                         [ 1.0,                          0.0,                          0.0,   tl_rot_l[2] + tl_rot_adj[2] ],
                                         [ 0.0,                          0.0,                          0.0,                      1.000000 ]])
        T_head_roti = T_tool_stand @ T_twist_lock_rotatei @ np.linalg.inv(T_pf_head) @ np.linalg.inv(T_tool_rot) # Global transformation matrix for the twist lock intermediate position
        robot.MoveL(rdk.Mat(T_head_roti.tolist()), blocking=True)           # Move to intermediate pose

    # Calcuate global tranformation matrix for 5 interpolated poses to unlock the portafilter tool
    for i in range(135,90-1,-9):
        twist_lock_alphai = -i/180 * np.pi                                  # Angle in radians for every 9 degree decrement

        # Transformation matrix for intermediate pose in tool stand frame
        T_twist_lock_rotatei = np.array([[ 0.0,    np.cos(twist_lock_alphai),    np.sin(twist_lock_alphai),   tl_rot_l[0] + tl_rot_adj[0] ],
                                         [ 0.0,   -np.sin(twist_lock_alphai),    np.cos(twist_lock_alphai),   tl_rot_l[1] + tl_rot_adj[1] ],
                                         [ 1.0,                          0.0,                          0.0,   tl_rot_l[2] + tl_rot_adj[2] ],
                                         [ 0.0,                          0.0,                          0.0,                      1.000000 ]])
        T_head_roti = T_tool_stand @ T_twist_lock_rotatei @ np.linalg.inv(T_pf_head) @ np.linalg.inv(T_tool_rot) # Global transformation matrix for the twist lock intermediate position
        robot.MoveL(rdk.Mat(T_head_roti.tolist()), blocking=True)           # Move to intermediate pose

    robot.MoveJ(rdk.Mat(T_head_pos.tolist()), blocking=True)                # Move to the twist lock unlocked position
    robot.MoveJ(rdk.Mat(T_head_rot.tolist()), blocking=True)                # Move to the twist lock stand-off pose
    robot.MoveJ(rdk.Mat(twist_lock_intermediate_angles), blocking=True)     # Move to the transition pose
    time.sleep(10)                                                          # Delay to allow Rodney to detach the portafilter tool

#endregion

# Tom this was all for other moves on twistlock
# pf_angle = -7.5*np.pi/180 #radians


# T_porta_pose_tool = np.array([[np.cos(pf_angle),  0, np.sin(pf_angle),   4.71],
#                               [0,                 1,                0,      0],
#                               [-np.sin(pf_angle), 0, np.cos(pf_angle), 144.76],
#                               [0,                 0,                0,      1]])

# twist_lock = np.array([14.9, 64.9, 214.0])
# twist_lock_adj = np.array([0, 0, -5]) 
# T_porta_on = np.array([[0, 0, -1, twist_lock[0] + twist_lock_adj[0] ],
#                         [0, 1, 0, twist_lock[1] + twist_lock_adj[1]  ], 
#                         [1, 0, 0, twist_lock[2] + twist_lock_adj[2]],
#                         [0, 0, 0, 1]])


# T_porta_rot_45 = np.array([[0, -np.sqrt(0.5), -np.sqrt(0.5), twist_lock[0] + twist_lock_adj[0]],
#                                 [0,  np.sqrt(0.5), -np.sqrt(0.5), twist_lock[1] + twist_lock_adj[1]],
#                                 [1,             0,             0, twist_lock[2] + twist_lock_adj[2]],
#                                 [0,             0,             0, 1]])


# T_pf_so = np.array([[0, 0, -1, twist_lock[0] + twist_lock_adj[0]],
#                                 [0, 1, 0, twist_lock[1] + twist_lock_adj[1]],
#                                 [1, 0, 0, twist_lock[2] -50 + twist_lock_adj[2]],
#                                 [0, 0, 0, 1]])



# T_porta_rot = np.array([[np.cos(np.pi/180 * -50), -np.sin(np.pi/180 * -50), 0, 0],
#                         [np.sin(np.pi/180 * -50), np.cos(np.pi/180 * -50),  0, 0],
#                         [0,                     0,                      1, 0],
#                         [0,                     0,                      0, 1]])

# #endregion

""" ---------------------------------------------- """
""" ----------------- Tool stand ----------------- """
""" ---------------------------------------------- """
#region Tool Stand
alpha_bt = np.arctan(1)-np.arctan(89.1/155.9) # Angle offset of the tool stand wrt the UR5

# Transformation matrix from the UR5 reference frame to the tool stand reference frame
T_tool_stand = np.array([[ np.cos(alpha_bt),   np.sin(alpha_bt),     0.0,       -556.5 ],
                        [ -np.sin(alpha_bt),   np.cos(alpha_bt),     0.0,        -77.4 ],
                        [               0.0,                0.0,     1.0,         19.05],
                        [               0.0,                0.0,     0.0,          1.0 ]])
#endregion

""" ---------------------------------------------- """
""" ----------------- Twist lock ----------------- """
""" ---------------------------------------------- """
#region Twist Lock
tl_pos_l = np.array([14.9, 64.9, 214.0])     # Twist lock position in global coordinates
tl_pos_adj = np.array([0, 0, -5])            # Minor adjustments to the twist lock position

# Transformation matrix from the UR5 reference frame to the twist lock pose
T_twist_lock_pos = np.array([[ 0.0,   0.0,    -1.0,   tl_pos_l[0] + tl_pos_adj[0] ],
                             [ 0.0,   1.0,     0.0,   tl_pos_l[1] + tl_pos_adj[1] ],
                             [ 1.0,   0.0,     0.0,   tl_pos_l[2] + tl_pos_adj[2] ],
                             [ 0.0,   0.0,     0.0,                     1.000000] ])

# Transformation matrix from the UR5 reference frame to the twist lock stand off pose
tl_rot_l = np.array([14.9, 64.9, 214.0])     # Twist lock position in global coordinates
tl_rot_adj = np.array([0, 0, -55])           # Minor adjustments to the twist lock position including 50 mm standoff
T_twist_lock_rotate = np.array([[ 0.0,   0.0,    -1.0,   tl_rot_l[0] + tl_rot_adj[0] ],
                                [ 0.0,   1.0,     0.0,   tl_rot_l[1] + tl_rot_adj[1] ],
                                [ 1.0,   0.0,     0.0,   tl_rot_l[2] + tl_rot_adj[2] ],
                                [ 0.0,   0.0,     0.0,                     1.000000] ])

def tamp_stand_to_twist_lock():
    """Routine to move from the tamp stand to the twist lock station"""
    transition_pose1 = np.array([-87.430000, -79.730000, -153.320000, -119.460000, -101.110000, -219.990000])   # First transition orientation 
    transition_pose2 = np.array([-162.000000, -79.730000, -153.320000, -119.460000, -101.110000, -219.990000])  # Second transition orientation

    ts_poses = tamp_stand_poses()                                       # Retrieve the tamp stand poses

    robot.MoveL(rdk.Mat(ts_poses["pose_tamp_stand_so"]))                # Move to the tamp stand stand-off pose 
    robot.MoveC(rdk.Mat(transition_pose1), rdk.Mat(transition_pose2))   # Move through the transition poses to the twist lock station


#endregion

""" ---------------------------------------------- """
""" ---------------- General Tool ---------------- """
""" ---------------------------------------------- """
#region General Tool
theta_gt = 50*(np.pi/180) # Angle offset of the tool piece wrt the UR5

# Transformation matrix to rotate the tool by the above angle
T_tool_rot = np.array([[  np.cos(theta_gt),     np.sin(theta_gt),     0.0,    0.0 ],
                       [ -np.sin(theta_gt),     np.cos(theta_gt),     0.0,    0.0 ],
                       [               0.0,                  0.0,     1.0,    0.0 ],
                       [               0.0,                  0.0,     0.0,    1.0 ]])
#endregion

""" ---------------------------------------------- """
""" ---------------- Grinder Tool ---------------- """
""" ---------------------------------------------- """
#region Grinder Tool
gt_push_but_l = np.array([0, 0, 102.82]) # Local position of the push button

# Transformation matrix from the grinder tool reference frame to the push button reference frame
T_push_button = np.array([[1.0,   0.0,   0.0,     gt_push_but_l[0] ],
                          [0.0,   1.0,   0.0,     gt_push_but_l[1] ],
                          [0.0,   0.0,   1.0,     gt_push_but_l[2] ],
                          [0.0,   0.0,   0.0,                  1.0 ]])

gt_pully_bit_l = np.array([-45.0, 0.0, 67.06]) # Local position of the pulling tool

# Transformation matrix from the grinder tool reference frame to the pulling tool reference frame
T_pully_bit = np.array([[1.0,   0.0,   0.0,   gt_pully_bit_l[0] ],
                        [0.0,   1.0,   0.0,   gt_pully_bit_l[1] ],
                        [0.0,   0.0,   1.0,   gt_pully_bit_l[2] ],
                        [0.0,   0.0,   0.0,                 1.0 ]])
#endregion

""" ---------------------------------------------- """
""" -------------- Portafilter Tool -------------- """
""" ---------------------------------------------- """
#region Portafilter Tool
pf_theta = -7.35 * np.pi/180                # Angle tilt of the portafilter head
pf_head_l = np.array([4.71, 0.0, 144.76])   # Local position of the portafilter head

# Tranformation matrix from the portafilter tool reference frame to the head reference frame
T_pf_head     = np.array([[     np.cos(pf_theta),     0.0,  np.sin(pf_theta),     pf_head_l[0] ],
                          [                  0.0,     1.0,               0.0,     pf_head_l[1] ],
                          [-1 * np.sin(pf_theta),     0.0,  np.cos(pf_theta),     pf_head_l[2] ], 
                          [                  0.0,     0.0,               0.0,              1.0 ]])

pf_base_l = np.array([-32.0, 0.0, 27.56])   # Local position of the portafilter base

# Tranformation matrix from the portafilter tool reference frame to the base reference frame
T_pf_base     = np.array([[     np.cos(pf_theta),     0.0,  np.sin(pf_theta),    pf_base_l[0] ],
                          [                  0.0,     1.0,               0.0,    pf_base_l[1] ],
                          [-1 * np.sin(pf_theta),     0.0,  np.cos(pf_theta),    pf_base_l[2] ],
                          [                  0.0,     0.0,               0.0,             1.0 ]])

pf_rim_l = np.array([22.0, 0.0, 0.0]) # Local position of the rim of the portafilter

# Transformation matrix from the portafilter tool reference frame to the rim reference frame
T_pf_top_edge =  T_pf_head @ np.array([[ 1.0,  0.0,  0.0,  pf_rim_l[0] ],
                                       [ 0.0,  1.0,  0.0,  pf_rim_l[1] ],
                                       [ 0.0,  0.0,  1.0,  pf_rim_l[2] ], 
                                       [ 0.0,  0.0,  0.0,          1.0 ]])
#endregion

""" ---------------------------------------------- """
""" ------------------- Cup Tool ----------------- """
""" ---------------------------------------------- """
#region Portafilter Tool
cp_centre_l = np.array([-47.0, 0.0, 186.11]) # Local position of the cup tool centre

# Transformation matrix from the cup tool reference from to the centre reference frame
T_cup_tool = np.array([[1.0,   0.0,   0.0,     cp_centre_l[0] ],
                       [0.0,   1.0,   0.0,     cp_centre_l[1] ],
                       [0.0,   0.0,   1.0,     cp_centre_l[2] ],
                       [0.0,   0.0,   0.0,                1.0 ]])
#endregion

""" ---------------------------------------------- """
""" -------------- Define Motion  ---------------- """
""" ---------------------------------------------- """
def main():
    
    robot.MoveJ(target, blocking=True)              # Start the robot at home position       

    coffee_grinder_place_portafilter_routine()      # Place the portafilter tool in the grinder
    coffee_grinder_button_routine()                 # Press the grinder buttons
    coffee_grinder_latch_routine()                  # Pull the grinder latch
    coffee_grinder_pickup_portafilter_routine()     # Pickup the portafilter from the grinder
    tamp_stand_scrape_and_tamp_routine()            # Tamp and scrape the coffee
    tamp_stand_to_twist_lock()                      # Move the portafilter to the twist lock location
    coffee_machine_portafilter_routine()            # Place the portafilter in the simulated coffee machine
    cup_to_coffee_machine()                         # Place the cup in the coffee machine
    coffee_machine_button_routine()                 # Press the buttons on the coffee machine
    cup_to_stand()                                  # Move the cup back to the stand

    robot.MoveJ(target, blocking=True)              # Return to the home position


if __name__ == '__main__':
    main()
