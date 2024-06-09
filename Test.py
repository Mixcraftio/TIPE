import numpy as np
import quaternion as quat

# q=quaternion.as_euler_angles(np.array([np.pi/2,0,0]))
q=quat.from_euler_angles(np.array([np.pi/2,0,0]))
print(q)