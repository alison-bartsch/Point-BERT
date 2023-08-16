import numpy as np

def get_rotation_matrix_from_quaternion(quaternion):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0,q1,q2,q3 = quaternion
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
    return rot_matrix

def get_quaternion_from_euler(rotation):
    """
    Convert an Euler angle to a quaternion.
    Input
        :param roll: The roll (rotation around x-axis) angle in radians.
        :param pitch: The pitch (rotation around y-axis) angle in radians.
        :param yaw: The yaw (rotation around z-axis) angle in radians.
    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    roll,pitch,yaw = rotation
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return (qw, qx, qy, qz)

def get_euler_from_quaternion(quaternion):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x, y, z, w = quaternion

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    
    return roll, pitch, yaw # in radians

def get_quaternion_from_rotation_matrix(quaternion):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
    Input
    :quaternion: A 4 element array representing the quaternion (q0,q1,q2,q3) 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0, q1, q2, q3 = quaternion
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

def get_quaternion_from_vectors(vector1,vector2):
    """
    Covert two vectors into a quaternion.
    Input
    :vector1: A 3 element array representing the vector (x1,y1,z1); 
    :vector2: A 3 element array representing the vector (x2,y2,z2); 
    Output
    :return: A 4 element array representing the quaternion (q0,q1,q2,q3) 
    """
    length1 = np.linalg.norm(vector1)
    length2 = np.linalg.norm(vector2)
    x,y,z = np.cross(vector1,vector2)
    w = np.sqrt((length1 ** 2) * (length2 ** 2)) + vector1 @ vector2.T

    quaternion = np.array([x,y,z,w])
    quaternion /= (np.linalg.norm(quaternion) + 1e-6)  # add a very some constant 1e-6 to avoid the case where the dominator is 0.
    return quaternion