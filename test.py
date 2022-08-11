import pybullet
import pybullet_data
import os
import time

urdf_path = 'iris_description/iris.urdf'

time_step = 0.01

pybullet.connect(pybullet.GUI)
pybullet.setGravity(0, 0, -9.81)
pybullet.setTimeStep(time_step)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = pybullet.loadURDF("plane.urdf")
drone = pybullet.loadURDF(os.path.abspath(urdf_path), useFixedBase=False)
pybullet.resetBasePositionAndOrientation(drone, [0, 0, 2.0], [0, 0, 0, 1.0])

body_link_id = 0
rotor0_link_id = 1
rotor1_link_id = 2
rotor2_link_id = 3
rotor3_link_id = 4

while True:
    rotor0_thrust = 10.0
    rotor1_thrust = 10.0
    rotor2_thrust = 10.0
    rotor3_thrust = 10.0
    pybullet.applyExternalForce(objectUniqueId=drone, 
                                linkIndex=rotor0_link_id, 
                                forceObj=[0, 0, rotor0_thrust],
                                posObj=[0, 0, 0],
                                flags=pybullet.LINK_FRAME)
    pybullet.applyExternalForce(objectUniqueId=drone, 
                                linkIndex=rotor1_link_id, 
                                forceObj=[0, 0, rotor1_thrust],
                                posObj=[0, 0, 0],
                                flags=pybullet.LINK_FRAME)
    pybullet.applyExternalForce(objectUniqueId=drone, 
                                linkIndex=rotor2_link_id, 
                                forceObj=[0, 0, rotor2_thrust],
                                posObj=[0, 0, 0],
                                flags=pybullet.LINK_FRAME)
    pybullet.applyExternalForce(objectUniqueId=drone, 
                                linkIndex=rotor3_link_id, 
                                forceObj=[0, 0, rotor3_thrust],
                                posObj=[0, 0, 0],
                                flags=pybullet.LINK_FRAME)
    # TODO: add body yaw-torque due to rotor torques
    rotor0_torque = 0.0
    rotor1_torque = 0.0
    rotor2_torque = 0.0
    rotor3_torque = 0.0
    yaw_torque = - rotor0_torque + rotor1_torque - rotor2_torque + rotor3_torque # is this correct?
    pybullet.applyExternalTorque(objectUniqueId=drone, 
                                 linkIndex=body_link_id, 
                                 torqueObj=[0, 0, yaw_torque],
                                 flags=pybullet.LINK_FRAME)

    pybullet.stepSimulation()
    time.sleep(time_step)