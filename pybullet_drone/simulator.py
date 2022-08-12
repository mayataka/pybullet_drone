import pybullet
import pybullet_data
import numpy as np
import time
import os

class ModelInfo(object):
    """ Info of a drone model. 

        Args: 
            urdf_path: PATH to the URDF file.
            body_frame: Name of the body frame. 
            prop_frames: List of name of each propeller frame. 
    """
    def __init__(self, urdf_path: str, body_frame: str, prop_frames: list):
        self.urdf_path = urdf_path
        self.body_frame = body_frame
        self.prop_frames = prop_frames

class DroneSimulator(object):
    """ Drone simulator. 

        Args: 
            model_info: ModelInfo instance.
            time_step: Time step of the simulation. 
            gui: If true, the gui is enabled. Default is true.
            log: If true, the logs are taken. Default is false.
            record: If true, the simulation videos are recorded. Default is false.
    """
    def __init__(self, model_info: ModelInfo, time_step: float, 
                 gui: bool=True, log: bool=False, record: bool=False):
        # simulation settings
        self.model_info = model_info
        self.time_step = time_step
        self.gui = gui
        self.log = log
        self.record = record
        # runtime variables
        self.time = None
        self.drone = None
        self.body_frame_id = None
        self.prop_frame_ids = []
        self.body_world_linear_acceleration = np.zeros(3)
        self.body_local_linear_acceleration = np.zeros(3)
        self.body_world_linear_velocity_prev = np.zeros(3)
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

    def set_camera(self, camera_distance, camera_yaw, camera_pitch, camera_target_pos):
        """ Sets camera calibration settings. 

            Args: 
                camera_distance: Distance between the camera and the target position .
                camera_yaw: Yaw angle of the camera.
                camera_pitch: Pitch angle of the camera.
                camera_target_pos: Target position.
        """
        self.calib_camera = True
        self.camera_distance = camera_distance
        self.camera_yaw = camera_yaw
        self.camera_pitch = camera_pitch
        self.camera_target_pos = camera_target_pos

    def get_time(self) -> float:
        """ Get the current simulation time. """
        return self.time

    def get_body_position(self) -> np.ndarray:
        """ Get the current body position of the drone. """
        base_position, _ = pybullet.getBasePositionAndOrientation(self.drone)
        return np.array(base_position)

    def get_body_quaternion(self) -> np.ndarray:
        """ Get the current body orientation of the drone expressed as a quaternion. """
        _, base_orientation = pybullet.getBasePositionAndOrientation(self.drone)
        return np.array(base_orientation) 

    def get_body_rotation_matrix(self) -> np.ndarray:
        """ Get the current body orientation of the drone expressed as a rotation matrix. """
        _, base_orientation = pybullet.getBasePositionAndOrientation(self.drone)
        base_rotation_matrix = np.reshape(pybullet.getMatrixFromQuaternion(base_orientation), [3, 3]) 
        return base_rotation_matrix

    def get_body_local_linear_velocity(self) -> np.ndarray:
        """ Get the current body linear velocity expressed in the body local frame. """
        body_local_linear_velocity, _ = pybullet.getBaseVelocity(self.drone)
        return np.array(body_local_linear_velocity)

    def get_body_local_angular_velocity(self) -> np.ndarray:
        """ Get the current body angular velocity expressed in the body local frame. """
        _, body_local_angular_velocity = pybullet.getBaseVelocity(self.drone)
        return np.array(body_local_angular_velocity)

    def get_body_world_linear_velocity(self) -> np.ndarray:
        """ Get the current body linear velocity expressed in the world frame. """
        return self.get_body_rotation_matrix().T @ self.get_body_local_linear_velocity()

    def get_body_world_angular_velocity(self) -> np.ndarray:
        """ Get the current body angular velocity expressed in the world frame. """
        return self.get_body_rotation_matrix().T @ self.get_body_local_angular_velocity()

    def get_body_local_linear_acceleration(self) -> np.ndarray:
        """ Get the current body linear acceleration expressed in the body local frame. """
        return self.body_local_linear_acceleration

    def get_body_world_linear_acceleration(self) -> np.ndarray:
        """ Get the current body linear acceleration expressed in the world frame. """
        return self.body_world_linear_acceleration

    def get_state(self, velocity_reference_frame: str='world') -> np.ndarray:
        """ Get the current state of the drone. 

            Args: 
                velocity_reference_frame: Reference frame of the linear and angular velocity. 
                    Should be chosen from 'world' or 'local'. Default is world.
        """
        if velocity_reference_frame == 'world':
            return np.concatenate([
                self.get_body_position(),
                self.get_body_quaternion(),
                self.get_body_world_linear_velocity(),
                self.get_body_world_angular_velocity()
            ])
        elif velocity_reference_frame == 'local':
            return np.concatenate([
                self.get_body_position(),
                self.get_body_quaternion(),
                self.get_body_local_linear_velocity(),
                self.get_body_local_angular_velocity()
            ])
        else:
            return NotImplementedError()

    def _get_link_index_from_joint_info(joint_info):
        return joint_info[0]

    def _get_link_name_from_joint_info(joint_info):
        return joint_info[12].decode('utf-8')

    def init(self, simulation_name: str, initial_time, initial_body_position: np.ndarray, 
             initial_body_quaternion: np.ndarray = np.array([0., 0., 0., 1.])) -> None:
        """ Initializes the simulator. 

            Args: 
                simulation_name: Name of this simulation.
                initial_time: Initial time of this simulation.
                initial_body_position: Initial body position of the drone.
                initial_body_quaternion: Initial body orientation of the drone expressed as a quaternion.
                    Default is the unit quaternion (no rotation).
        """
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

        self.drone = pybullet.loadURDF(os.path.abspath(self.model_info.urdf_path), useFixedBase=False)
        self.body_frame_id = None
        self.prop_frame_ids = []
        nJoints = pybullet.getNumJoints(self.drone)
        for j in range(nJoints):
            joint_info = pybullet.getJointInfo(self.drone, j)
            print(DroneSimulator._get_link_name_from_joint_info(joint_info))
            if DroneSimulator._get_link_name_from_joint_info(joint_info) == self.model_info.body_frame:
                self.body_frame_id = DroneSimulator._get_link_index_from_joint_info(joint_info)
            for prop_frame in self.model_info.prop_frames:
                if DroneSimulator._get_link_name_from_joint_info(joint_info) == prop_frame:
                    self.prop_frame_ids.append(DroneSimulator._get_link_index_from_joint_info(joint_info)) 
        if self.body_frame_id is None:
            raise RuntimeError("Could not find the specified body frame!")
        if len(self.prop_frame_ids) != len(self.model_info.prop_frames):
            raise RuntimeError("Could not find the all of the specified prop frames!")

        if self.calib_camera:
            pybullet.resetDebugVisualizerCamera(self.camera_distance,
                                                self.camera_yaw,
                                                self.camera_pitch,
                                                self.camera_target_pos)

        pybullet.resetBasePositionAndOrientation(self.drone, initial_body_position.tolist(), 
                                                 initial_body_quaternion.tolist())
        self.body_world_linear_acceleration = np.zeros(3)
        self.body_local_linear_acceleration = np.zeros(3)
        self.body_world_linear_velocity_prev = np.zeros(3)
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

    def step(self, u: np.ndarray) -> None:
        """ Proceeds a simulation step. 

            Args: 
                u: Control input, i.e., the thrusts of propellers that are specified as the prop_frames of ModelInfo. 
        """
        if self.drone == None:
            raise RuntimeError("This simulator is not initialized!")
        assert u.shape[0] == len(self.prop_frame_ids)
        for thrust, prop_frame_id in zip(u.tolist(), self.prop_frame_ids):
            pybullet.applyExternalForce(objectUniqueId=self.drone, 
                                        linkIndex=prop_frame_id,
                                        forceObj=[0, 0, thrust],
                                        posObj=[0, 0, 0],
                                        flags=pybullet.LINK_FRAME)
        # TODO: add body yaw-torque due to prop torques
        yaw_torque = 0.0
        pybullet.applyExternalTorque(objectUniqueId=self.drone, 
                                     linkIndex=self.body_frame_id, 
                                     torqueObj=[0, 0, yaw_torque],
                                     flags=pybullet.LINK_FRAME)
        if self.log:
            np.savetxt(self.x_log, [self.get_state()])
            np.savetxt(self.u_log, [u])
            np.savetxt(self.t_log, np.array([self.get_time()]))

        pybullet.stepSimulation()
        time.sleep(self.time_step)
        self.time += self.time_step

        self.body_world_linear_acceleration = (self.get_body_world_linear_velocity() - self.body_world_linear_velocity_prev) / self.time_step
        self.body_local_linear_acceleration = self.get_body_rotation_matrix() @ self.body_world_linear_acceleration
        self.body_world_linear_velocity_prev = self.get_body_world_linear_velocity()