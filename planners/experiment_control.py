import ast
import time
import numpy as np
import UdpComms as U
from frankapy import FrankaArm
from scipy.spatial.transform import Rotation

def goto_grasp(fa, x, y, z, rx, ry, rz, d):
	"""
	Parameterize a grasp action by the position [x,y,z] Euler angle rotation [rx,ry,rz], and width [d] of the gripper.
	This function was designed to be used for clay moulding, but in practice can be applied to any task.

	:param fa:  franka robot class instantiation
	"""
	pose = fa.get_pose()
	starting_rot = pose.rotation
	orig = Rotation.from_matrix(starting_rot)
	orig_euler = orig.as_euler('xyz', degrees=True)
	rot_vec = np.array([rx, ry, rz])
	new_euler = orig_euler + rot_vec
	r = Rotation.from_euler('xyz', new_euler, degrees=True)
	pose.rotation = r.as_matrix()
	pose.translation = np.array([x, y, z])

	fa.goto_pose(pose)
	fa.goto_gripper(d, force=60.0)
        
    # TODO: update this with the scaled sleep based on distance between fingertips
	time.sleep(3)
        
if __name__=='__main__':

    # initialize the udp communication with the vision computer
    udp = U.UdpComms(udpIP='172.26.5.54', sendIP='172.26.69.200', portTX=5500, portRX=5501, enableRX=True)

    # initialize the robot and reset joints
    fa = FrankaArm()
    fa.reset_joints()
    fa.reset_pose()

    # move to observation pose
    observation_pose = np.array([0.5, 0.0, 0.035]) # TODO: replace with correct values
    pose = fa.get_pose()
    pose.translation = observation_pose
    fa.goto_pose(pose)

    # currently experiment code will run forever - just kill when done (TODO: better stopping)
    while True:
        # wait for signal from vision computer
        while True:
            action_msg = udp.ReadReceivedData()
            if action_msg is not None:
                action_list = ast.literal_eval(action_msg)
                print(action_list)
                break

        # once receive a list of actions, iterate through the list and execute each action
        for action in action_list:
            x = action[0]
            y = action[1]
            z = action[2]
            rx = 0
            ry = 0
            rz = action[3]
            d = action[4]
            goto_grasp(fa, x, y, z, rx, ry, rz, d)

            fa.goto_pose(pose)
            time.sleep(0.5)

        # once done executing actions, send signal back to the vision computer
        udp.SendData('done') 