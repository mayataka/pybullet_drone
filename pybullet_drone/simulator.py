import pybullet
import pybullet_data
import numpy as np
import time
import os

BODY_LINK_ID = 0
ROTOR0_LINK_ID = 1
ROTOR1_LINK_ID = 2
ROTOR2_LINK_ID = 3
ROTOR3_LINK_ID = 4

class DroneSimulator(object):
    def __init__(self, urdf_path, time_step, gui=True, log=False, record=False):
        # simulation settings
        self.urdf_path = urdf_path
        self.time_step = time_step
        self.gui = gui
        self.log = log
        self.record = record
        # runtime variables
        self.time = None
        self.drone = None
        # camera 
        self.calib_camera = False
        self.camera_distance = 0.0
        self.camera_yaw = 0.0
        self.camera_pitch = 0.0
        self.camera_target_pos = [0., 0., 0.]
        # log 
        self.log_dir = None
        self.x_log = None
        self.u_log = None
        self.t_log = None
        self.sim_name = None

    def set_urdf(self, path_to_urdf):
        self.path_to_urdf = path_to_urdf

    def set_camera(self, camera_distance, camera_yaw, camera_pitch, camera_target_pos):
        self.calib_camera = True
        self.camera_distance = camera_distance
        self.camera_yaw = camera_yaw
        self.camera_pitch = camera_pitch
        self.camera_target_pos = camera_target_pos
    
    def get_time(self):
        return self.time

    def get_body_position(self) -> np.ndarray:
        base_position, _ = pybullet.getBasePositionAndOrientation(self.drone)
        return np.array(base_position)

    def get_body_quaternion(self) -> np.ndarray:
        _, base_orientation = pybullet.getBasePositionAndOrientation(self.drone)
        return np.array(base_orientation) 

    def get_body_rotation_matrix(self) -> np.ndarray:
        _, base_orientation = pybullet.getBasePositionAndOrientation(self.drone)
        base_rotation_matrix = np.reshape(pybullet.getMatrixFromQuaternion(base_orientation), [3, 3]) 
        return base_rotation_matrix

    def get_body_local_linear_velocity(self) -> np.ndarray:
        body_local_linear_velocity, _ = pybullet.getBaseVelocity(self.drone)
        return np.array(body_local_linear_velocity)

    def get_body_local_angular_velocity(self) -> np.ndarray:
        _, body_local_angular_velocity = pybullet.getBaseVelocity(self.drone)
        return np.array(body_local_angular_velocity)

    def get_body_world_linear_velocity(self) -> np.ndarray:
        return self.get_body_rotation_matrix().T @ self.get_body_local_linear_velocity()

    def get_body_world_angular_velocity(self) -> np.ndarray:
        return self.get_body_rotation_matrix().T @ self.get_body_local_angular_velocity()

    def get_state(self, reference_frame='world') -> np.ndarray:
        if reference_frame == 'world':
            return np.concatenate([
                self.get_body_position(),
                self.get_body_quaternion(),
                self.get_body_world_linear_velocity(),
                self.get_body_world_angular_velocity()
            ])
        elif reference_frame == 'local':
            return np.concatenate([
                self.get_body_position(),
                self.get_body_quaternion(),
                self.get_body_local_linear_velocity(),
                self.get_body_local_angular_velocity()
            ])
        else:
            return NotImplementedError()

    def init_simulation(self, simulation_name: str, initial_time, initial_body_position: np.ndarray, 
                        initial_body_quaternion: np.ndarray = np.array([0., 0., 0., 1.])) -> None:
        assert initial_body_position.shape[0] == 3
        assert initial_body_quaternion.shape[0] == 4
        self.time = initial_time
        if self.gui:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)
        pybullet.setGravity(0, 0, -9.81)
        pybullet.setTimeStep(self.time_step)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane = pybullet.loadURDF("plane.urdf")
        self.drone = pybullet.loadURDF(os.path.abspath(self.urdf_path), useFixedBase=False)
        pybullet.resetBasePositionAndOrientation(self.drone, initial_body_position.tolist(), 
                                                 initial_body_quaternion.tolist())
        if self.log:
            log_dir = os.path.join(os.getcwd(), simulation_name+"_log")
            self.log_dir = log_dir
            os.makedirs(log_dir, exist_ok=True)
            self.x_log = open(os.path.join(log_dir, "x.log"), mode='w')
            self.u_log = open(os.path.join(log_dir, "u.log"), mode='w')
            self.t_log = open(os.path.join(log_dir, "t.log"), mode='w')

        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        if self.record:
            pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, simulation_name+'.mp4')

    def step_simulation(self, u: np.ndarray) -> None:
        if self.drone is None:
            return RuntimeError()
        assert u.shape[0] == 4
        rotor0_thrust = u[0]
        rotor1_thrust = u[1]
        rotor2_thrust = u[2]
        rotor3_thrust = u[3]
        pybullet.applyExternalForce(objectUniqueId=self.drone, 
                                    linkIndex=ROTOR0_LINK_ID, 
                                    forceObj=[0, 0, rotor0_thrust],
                                    posObj=[0, 0, 0],
                                    flags=pybullet.LINK_FRAME)
        pybullet.applyExternalForce(objectUniqueId=self.drone, 
                                    linkIndex=ROTOR1_LINK_ID, 
                                    forceObj=[0, 0, rotor1_thrust],
                                    posObj=[0, 0, 0],
                                    flags=pybullet.LINK_FRAME)
        pybullet.applyExternalForce(objectUniqueId=self.drone, 
                                    linkIndex=ROTOR2_LINK_ID, 
                                    forceObj=[0, 0, rotor2_thrust],
                                    posObj=[0, 0, 0],
                                    flags=pybullet.LINK_FRAME)
        pybullet.applyExternalForce(objectUniqueId=self.drone, 
                                    linkIndex=ROTOR3_LINK_ID, 
                                    forceObj=[0, 0, rotor3_thrust],
                                    posObj=[0, 0, 0],
                                    flags=pybullet.LINK_FRAME)
        # TODO: add body yaw-torque due to rotor torques
        rotor0_torque = 0.0
        rotor1_torque = 0.0
        rotor2_torque = 0.0
        rotor3_torque = 0.0
        yaw_torque = - rotor0_torque + rotor1_torque - rotor2_torque + rotor3_torque # is this correct?
        pybullet.applyExternalTorque(objectUniqueId=self.drone, 
                                     linkIndex=BODY_LINK_ID, 
                                     torqueObj=[0, 0, yaw_torque],
                                     flags=pybullet.LINK_FRAME)
        if self.log:
            np.savetxt(self.x_log, [self.get_state()])
            np.savetxt(self.u_log, [u])
            np.savetxt(self.t_log, np.array([self.get_time()]))

        pybullet.stepSimulation()
        time.sleep(self.time_step)
        self.time += self.time_step