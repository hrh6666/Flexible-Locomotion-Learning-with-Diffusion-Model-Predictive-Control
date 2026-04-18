import numpy as np
import os.path as osp
from legged_gym.envs.go2.go2_field_config import Go2FieldCfg, Go2FieldCfgPPO
from legged_gym.utils.helpers import merge_dict

class Go2RemoteCfg( Go2FieldCfg ):
    class env(Go2FieldCfg.env ):
        num_envs = 4096
        obs_components = [ # The first 2 must be proprioception and height_measurements
            "proprioception", # 48
            # "height_measurements", # 187
            # "forward_depth",
            # "base_pose",
            # "robot_config",
            "robot_friction",
            #"stumble_state",
            # "engaging_block",
            # "sidewall_distance",
        ]
        privileged_obs_components = [ # The first 2 must be proprioception and height_measurements
            "proprioception",
            # "height_measurements", 
            # "forward_depth",
            # "robot_config",
            "robot_friction", 
            #"stumble_state",
        ]
        estimated_obs_components = [ # The first 1 must be lin_vel
            "lin_vel",
            # "height_measurements", 
            # "forward_depth",
            # "robot_config",
            "robot_friction", 
            #"stumble_state",
        ]
        transition_states_components = [
            "dof_pos",
            "base_height",
            "base_lin_vel",
            "base_ang_vel", 
            # "commands_transition"
        ]
        use_lin_vel = True
        # use_lin_vel = False
        privileged_use_lin_vel = True
        
    class init_state(Go2FieldCfg.init_state ):
        zero_actions=True

    class terrain(Go2FieldCfg.terrain ):
        num_rows = 10
        num_cols = 10
        selected = "TerrainPerlin"
        TerrainPerlin_kwargs = dict(
            zScale= 0.1,
            frequency= 10,
        )
        unify = False
        max_height = 0.6 # max slope height
        max_init_terrain_level = 1
        curriculum = False
        measure_heights = True

    class commands(Go2FieldCfg.commands ):
        class ranges(Go2FieldCfg.commands.ranges ):
            lin_vel_x = [-1., 1.]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-1., 1.] 
        resampling_time = 5.

    class domain_rand(Go2FieldCfg.domain_rand ):
        class com_range(Go2FieldCfg.domain_rand.com_range ):
            x = [-0.2, 0.2]
        max_push_vel_ang = 0.5
        push_robots = True
        push_interval_s = 9
        # init_dof_pos_ratio = 0.25
        
        # init_base_vel_range = dict(
        #     x=[-1.,1.],
        #     y=[-1.,1.],
        #     z=[0.,0.],
        #     roll=[0.,0.],
        #     pitch=[0.,0.],
        #     yaw=[-1.,-1.]
        # )
        init_base_vel_range = [-1.,1.]
        init_base_rot_range = dict(
            roll= [0, 0],
            pitch= [0, 0],
            yaw= [-3.14,3.14]
        )

    class rewards(Go2FieldCfg.rewards ):
        class scales:
            ###### hacker from Field
            tracking_ang_vel = 2.5
            # lin_vel_l2norm = -3.0
            tracking_lin_vel = 7.0
            legs_energy_substeps = -1e-6
            alive =1
            # penalty for hardware safety
            exceed_dof_pos_limits = -0.01
            exceed_torque_limits_l1norm = -2.0
            # penalty for walking gait, probably no need
            # collision = -0.1
            orientation = -0.5
            feet_contact_forces = -5e-3
            stand_still = -4.0
            
            # stand_rotation = -20.0
            termination = -1
            # feet_air_time = -20
            feet_air_time_l1 = 0.1
            dof_vel=-0.002
            dof_acc=-2e-6
            # thigh_acc=-3e-6
            front_hip_pos = -0.1
            rear_hip_pos = -0.1
            # foot_slip = -0.04
            # foot_gait=-1e0.21
            foot_contact=-2e-5
            # foot_contact_x=-2e-4
            ang_vel_xy = -0.2
            base_height = -1000.0
            
        soft_dof_pos_limit = 0.9
        # soft_dof_vel_limit: 0.5,
        max_contact_force = 60.0
        tracking_sigma = 0.20
        base_height_target = 0.25
        
    class termination(Go2FieldCfg.termination):
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            # "out_of_track",
            # "stuck",
        ]
        stuck_kwargs = dict(
            threshold= 0.1,
            steps=20,
            recover_steps=50
        )
        # additional factors that determines whether to terminates the episode
        timeout_at_finished = True
        
    # class sim(Go2FieldCfg.sim ):
    #     no_camera = False
    
    class noise(Go2FieldCfg.noise ):
        add_noise = True # disable internal uniform +- 1 noise, and no noise in proprioception
        class noise_scales(Go2FieldCfg.noise.noise_scales ):
            lin_vel = 0.05
            robot_config = 0.
            # robot_friction = 0.1
            height_measurements = 0.
        
    class sensor(Go2FieldCfg.sensor ):
        # class forward_camera(Go2FieldCfg.sensor.forward_camera ):
        #     resolution = [int(240/4), int(424/4)]
        #     position = dict(
        #         mean= [0.27, 0.0075, 0.033],
        #         std= [0.01, 0.0025, 0.0005],
        #     ) # position in base_link ##### small randomization
        #     rotation = dict(
        #         lower= [0, 0, 0],
        #         upper= [0, 5 * np.pi / 180, 0],
        #     ) # rotation in base_link ##### small randomization
        #     resized_resolution= [48, 64]
        #     output_resolution = [48, 64]
        #     horizontal_fov = [85, 88]

        #     # adding randomized latency
        #     latency_range = [0.2, 0.26] # for [16, 32, 32] -> 128 -> 128 visual model in (240, 424 option)
        #     latency_resample_time = 5.0 # [s]
        #     refresh_duration = 1/10 # [s] for (240, 424 option with onboard script fixed to no more than 20Hz)

        #     # config to simulate stero RGBD camera
        #     crop_top_bottom = [0, 0]
        #     crop_left_right = [int(60/4), int(46/4)]
        #     depth_range = [0.0, 1.5] # [m]

        class proprioception:
            delay_action_obs = False
            latency_range = [0.005, 0.045] # [min, max] in seconds
            # latency_range = [0.04-0.0025, 0.04+0.0075] # [min, max] in seconds
            latency_resample_time = 2.0 # [s]    
    

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go2RemoteCfgPPO( Go2FieldCfgPPO ):
    class policy(Go2FieldCfgPPO.policy):
        scan_encoder_dims = [128, 64, 32]
        scan_decoder_dims = [64, 128]
        actor_hidden_dims = [128, 128]
        critic_hidden_dims = [128, 128]
        
        # Activation function configuration
        activation = 'elu'
        mu_activation = None  # Use a specific activation for the output layer if needed
        
        num_classes = 7
        
        rnn_type = "lstm"
        has_scan = True if "height_measurements" in Go2RemoteCfg.env.obs_components else False
    
    class algorithm( Go2FieldCfgPPO.algorithm ):
        entropy_coef = 0.01
        clip_min_std = 0.05

    class runner( Go2FieldCfgPPO.runner ):
        # policy_class_name = "ActorCriticEncode"
        # algorithm_class_name = "PPO" # PPO for MLP actor, RPPO for RNN actor
        policy_class_name = "ActorCriticRecurrent"
        algorithm_class_name = "RPPO" # PPO for MLP actor, RPPO for RNN actor
        num_steps_per_env = 20
        amp_reward_coeff = 0.
        resume = True
        load_run = "Apr10_06-27-05_WalkByRemoteRecurrent_rLinTrack7.0_rAng2.5_rAlive1.0_pEnergySubsteps-1e-06_noResume"
        
        run_name = "".join(["WalkByRemoteRecurrent",
        ("_rLinTrack{:.1f}".format(Go2RemoteCfg.rewards.scales.tracking_lin_vel) if getattr(Go2RemoteCfg.rewards.scales, "tracking_lin_vel", 0.) != 0. else ""),
        ("_pLin{:.1f}".format(-Go2RemoteCfg.rewards.scales.lin_vel_l2norm) if getattr(Go2RemoteCfg.rewards.scales, "lin_vel_l2norm", 0.) != 0. else ""),
        ("_rAng{:.1f}".format(Go2RemoteCfg.rewards.scales.tracking_ang_vel) if Go2RemoteCfg.rewards.scales.tracking_ang_vel != 0.2 else ""),
        ("_rAlive{:.1f}".format(Go2RemoteCfg.rewards.scales.alive) if getattr(Go2RemoteCfg.rewards.scales, "alive", 2.) != 2. else ""),
        ("_pEnergySubsteps{:.0e}".format(Go2RemoteCfg.rewards.scales.legs_energy_substeps) if getattr(Go2RemoteCfg.rewards.scales, "legs_energy_substeps", -2e-5) != -2e-5 else "_nopEnergy"),
        ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])
        save_interval= 1000
        max_iterations = 4000
        
    class estimator:
        train_with_estimated_states = True
        train_together = False
        learning_rate = 1.e-4
        input_dim = 45
        rnn_type = 'lstm'
        # rnn_type = 'CNN' 
        rnn_hidden_size = 256
        latent_encoder_hidden_dims=[256, 128]
        estimator_decay = 0.
        decay_start_step = 0
        # scan_vae_loss=0.01
        ppo_train_together = True
        estimator_loss_coeff = 0.
        class_loss_coeff = 0.
        
    class discriminator:
        state_size=19
        hidden_sizes=[1024, 512]
        learning_rate = 1.e-4
        file_path = './mpc_data_no_command_go2.npy'
        batch_size = 1024
        gradient_penalty = 10