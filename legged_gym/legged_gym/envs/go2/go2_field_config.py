import numpy as np
import os.path as osp
from legged_gym.envs.go2.go2_config import Go2RoughCfg, Go2RoughCfgPPO

class Go2FieldCfg( Go2RoughCfg ):
    class env( Go2RoughCfg.env ):
        num_envs = 4096 # 8192
        obs_components = [
            "proprioception", # 48
            # "height_measurements", # 187
            "base_pose",
            "robot_config",
            "engaging_block",
            "sidewall_distance",
        ]
        goal_command=False
        # privileged_use_lin_vel = True # for the possible of setting "proprioception" in obs and privileged obs different

        ######## configs for training a walk policy ############
        # obs_components = [
        #     "proprioception", # 48
        #     # "height_measurements", # 187
        #     # "forward_depth",
        #     # "base_pose",
        #     # "robot_config",
        #     # "engaging_block",
        #     # "sidewall_distance",
        # ]
        # privileged_obs_components = [
        #     "proprioception",
        #     # "height_measurements",
        #     # "forward_depth",
        #     "robot_config",
        # ]
        ######## End configs for training a walk policy ############

    class sensor:
        class forward_camera:
            resolution = [16, 16]
            position = [0.26, 0., 0.03] # position in base_link
            rotation = [0., 0., 0.] # ZYX Euler angle in base_link
    
        class proprioception:
            delay_action_obs = False
            latency_range = [0.0, 0.0]
            latency_resample_time = 2.0 # [s]
            
    class init_state( Go2RoughCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
    
    class terrain( Go2RoughCfg.terrain ):
        mesh_type = "trimesh" # Don't change
        num_rows = 20
        num_cols = 50
        selected = "BarrierTrack" # "BarrierTrack" or "TerrainPerlin", "TerrainPerlin" can be used for training a walk policy.
        max_init_terrain_level = 0
        border_size = 0
        slope_treshold = 20.
        unify = False

        curriculum = False # for walk
        horizontal_scale = 0.025 # [m]
        # vertical_scale = 1. # [m] does not change the value in hightfield
        pad_unavailable_info = True

        BarrierTrack_kwargs = dict(
            options= [
                # "jump",
                # "crawl",
                # "tilt",
                # "leap",
            ], # each race track will permute all the options
            track_width= 1.6,
            track_block_length= 2., # the x-axis distance from the env origin point
            wall_thickness= (0.04, 0.2), # [m]
            wall_height= -0.05,
            jump= dict(
                height= (0.2, 0.6),
                depth= (0.1, 0.8), # size along the forward axis
                fake_offset= 0.0, # [m] an offset that make the robot easier to get into the obstacle
                jump_down_prob= 0., # probability of jumping down use it in non-virtual terrain
            ),
            crawl= dict(
                height= (0.25, 0.5),
                depth= (0.1, 0.6), # size along the forward axis
                wall_height= 0.6,
                no_perlin_at_obstacle= False,
            ),
            tilt= dict(
                width= (0.24, 0.32),
                depth= (0.4, 1.), # size along the forward axis
                opening_angle= 0.0, # [rad] an opening that make the robot easier to get into the obstacle
                wall_height= 0.5,
            ),
            leap= dict(
                length= (0.2, 1.0),
                depth= (0.4, 0.8),
                height= 0.2,
            ),
            add_perlin_noise= True,
            border_perlin_noise= True,
            border_height= 0.,
            virtual_terrain= False,
            draw_virtual_terrain= True,
            engaging_next_threshold= 1.2,
            engaging_finish_threshold= 0.,
            curriculum_perlin= False,
            no_perlin_threshold= 0.1,
        )

        TerrainPerlin_kwargs = dict(
            zScale= [0.08, 0.15],
            # zScale= 0.15, # Use a constant zScale for training a walk policy
            frequency= 10,
        )
    
    class commands( Go2RoughCfg.commands ):
        heading_command = False
        resampling_time = 10 # [s]
        class ranges( Go2RoughCfg.commands.ranges ):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]
        clear_vel_cmd_threshold = 0.1
        clear_ang_cmd_threshold = 0.1

    class control( Go2RoughCfg.control ):
        stiffness = {'joint': 40.}
        damping = {'joint': 1.}
        action_scale = 0.5
        torque_limits = 25 # override the urdf
        computer_clip_torque = False
        motor_clip_torque = False
        # action_delay = False # set to True to enable action delay in sim
        # action_delay_range = [0.002, 0.022] # [s]
        # action_delay_resample_time = 5.0 # [s]

    class asset( Go2RoughCfg.asset ):
        penalize_contacts_on = ["base", "thigh", "calf"]
        terminate_after_contacts_on = ["base", "imu", "Head"]
        front_hip_names = ["FR_hip_joint", "FL_hip_joint"]
        rear_hip_names = ["RR_hip_joint", "RL_hip_joint"]
        rear_calf_names = ["RR_calf_joint", "RL_calf_joint"]
        collision_body_names = ['thigh', 'calf', 'base', 'foot', 'hip']
        rear_left_dof_names = ["RL_thigh_joint", "RL_calf_joint"]
        rear_right_dof_names = ["RR_thigh_joint", "RR_calf_joint"]
        step_num = 100

    class termination:
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            # "out_of_track",
        ]

        roll_kwargs = dict(
            threshold= 1.0, # [rad]
            tilt_threshold= 1.5,
        )
        pitch_kwargs = dict(
            threshold= 1.6, # [rad] # for leap, jump
            jump_threshold= 1.6,
            leap_threshold= 1.5,
        )
        z_low_kwargs = dict(
            threshold= 0.15, # [m]
        )
        z_high_kwargs = dict(
            threshold= 1.5, # [m]
        )
        out_of_track_kwargs = dict(
            threshold= 1., # [m]
        )

        check_obstacle_conditioned_threshold = True
        timeout_at_border = False

    class domain_rand( Go2RoughCfg.domain_rand ):
        randomize_com = True
        class com_range:
            x = [-0.05, 0.15]
            y = [-0.1, 0.1]
            z = [-0.05, 0.05]

        randomize_motor = True
        leg_motor_strength_range = [0.9, 1.1]

        randomize_base_mass = True
        added_mass_range = [1.0, 3.0]

        randomize_friction = True
        friction_range = [0., 2.]

        # init_base_pos_range = dict(
        #     x= [-0.2, 0.2],
        #     y= [-0.25, 0.25],
        # )
        
        init_base_pos_range = dict(
            x= [-0.2, 0.2],
            y= [-0.2, 0.2],
            z= [-0., 0.],
        )

        push_robots = False 

    class rewards( Go2RoughCfg.rewards ):
        class scales:
            tracking_ang_vel = 0.05
            world_vel_l2norm = -1.
            legs_energy_substeps = -2e-5
            legs_energy = -0.
            alive = 2.
            # penalty for hardware safety
            exceed_dof_pos_limits = -0.1
            exceed_torque_limits_l1norm = -0.1
            exceed_torque_limits_i = -2e-1

    class normalization( Go2RoughCfg.normalization ):
        class obs_scales( Go2RoughCfg.normalization.obs_scales ):
            forward_depth = 1.
            base_pose = [0., 0., 0., 1., 1., 1.]
            engaging_block = 1.
            robot_config = 1.
            robot_friction = 1.
        """ The following action clip is used for tanh policy activation. """
        # clip_actions_method = "hard"
        # dof_pos_redundancy = 0.2
        # clip_actions_low = []
        # clip_actions_high = []
        # for sdk_joint_name, sim_joint_name in zip(
        #     ["Hip", "Thigh", "Calf"] * 4,
        #     [ # in the order as simulation
        #         "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        #         "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        #         "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        #         "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        #     ],
        # ): # This setting is only for position control.
        #     clip_actions_low.append( (A1RoughCfg.asset.sdk_dof_range[sdk_joint_name + "_min"] + dof_pos_redundancy - A1RoughCfg.init_state.default_joint_angles[sim_joint_name]) / a1_action_scale )
        #     clip_actions_high.append( (A1RoughCfg.asset.sdk_dof_range[sdk_joint_name + "_max"] - dof_pos_redundancy - A1RoughCfg.init_state.default_joint_angles[sim_joint_name]) / a1_action_scale )
        # del dof_pos_redundancy, sdk_joint_name, sim_joint_name # This is not intended to be an attribute of normalization
        """ The above action clip is used for tanh policy activation. """
        height_measurement_offset = -0.3

    class noise( Go2RoughCfg.noise ):
        add_noise = True # disable internal uniform +- 1 noise, and no noise in proprioception
        class noise_scales( Go2RoughCfg.noise.noise_scales ):
            forward_depth = 0.1
            base_pose = 1.0

    class viewer( Go2RoughCfg.viewer ):
        pos = [0, 0, 5]  # [m]
        lookat = [5., 5., 2.]  # [m]

        draw_volume_sample_points = False

    class sim( Go2RoughCfg.sim ):
        body_measure_points = { # transform are related to body frame
            "base": dict(
                x= [i for i in np.arange(-0.2, 0.31, 0.03)],
                y= [-0.08, -0.04, 0.0, 0.04, 0.08],
                z= [i for i in np.arange(-0.061, 0.061, 0.03)],
                transform= [0., 0., 0.005, 0., 0., 0.],
            ),
            "thigh": dict(
                x= [
                    -0.16, -0.158, -0.156, -0.154, -0.152,
                    -0.15, -0.145, -0.14, -0.135, -0.13, -0.125, -0.12, -0.115, -0.11, -0.105, -0.1, -0.095, -0.09, -0.085, -0.08, -0.075, -0.07, -0.065, -0.05,
                    0.0, 0.05, 0.1,
                ],
                y= [-0.015, -0.01, 0.0, -0.01, 0.015],
                z= [-0.03, -0.015, 0.0, 0.015],
                transform= [0., 0., -0.1,   0., 1.57079632679, 0.],
            ),
            "calf": dict(
                x= [i for i in np.arange(-0.13, 0.111, 0.03)],
                y= [-0.015, 0.0, 0.015],
                z= [-0.015, 0.0, 0.015],
                transform= [0., 0., -0.11,   0., 1.57079632679, 0.],
            ),
        }

    class curriculum:
        no_moveup_when_fall = False
        # chosen heuristically, please refer to `LeggedRobotField._get_terrain_curriculum_move` with fixed body_measure_points

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go2FieldCfgPPO( Go2RoughCfgPPO ):
    class algorithm( Go2RoughCfgPPO.algorithm ):
        entropy_coef = 0.01
        clip_min_std = 1e-12

    class policy( Go2RoughCfgPPO.policy ):
        rnn_type = 'gru'
        activation = 'elu'
        mu_activation = "tanh" 
        stumble_encoder_dims = None
    
    class runner( Go2RoughCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        experiment_name = "field_go2"
        resume = False
        
        init_from_pretrained = False
        
        run_name = "".join(["WalkForward",
        ("_propDelay{:.2f}-{:.2f}".format(
                Go2FieldCfg.sensor.proprioception.latency_range[0],
                Go2FieldCfg.sensor.proprioception.latency_range[1],
            ) if Go2FieldCfg.sensor.proprioception.delay_action_obs else ""
        ),
        ("_aScale{:d}{:d}{:d}".format(
                int(Go2FieldCfg.control.action_scale[0] * 10),
                int(Go2FieldCfg.control.action_scale[1] * 10),
                int(Go2FieldCfg.control.action_scale[2] * 10),
            ) if isinstance(Go2FieldCfg.control.action_scale, (tuple, list)) \
            else "_aScale{:.1f}".format(Go2FieldCfg.control.action_scale)
        ),
        ])
        resume = False
        max_iterations = 10000
        save_interval = 500
    