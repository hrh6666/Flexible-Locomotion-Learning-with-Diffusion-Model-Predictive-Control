import numpy as np
import os.path as osp
from legged_gym.envs.go2.go2_field_config import Go2FieldCfg, Go2FieldCfgPPO
from legged_gym.utils.helpers import merge_dict
from legged_gym.envs.base.base_config import BaseConfig

class Go2DiffuserCfg( Go2FieldCfg ):
    class env(Go2FieldCfg.env ):
        num_envs = 4096
        obs_components = [
            "proprioception",
            "robot_friction",
        ]
        privileged_obs_components = [ 
            "proprioception",
            "robot_friction", 
        ]
        estimated_obs_components = [
            "lin_vel",
            "robot_friction", 
        ]
        transition_states_components = [
            "dof_pos",
            "base_height",
            "base_lin_vel",
            "base_ang_vel", 
        ]
        use_lin_vel = False
        privileged_use_lin_vel = False
        use_friction = False
        
    class init_state(Go2FieldCfg.init_state ):
        zero_actions=True

    class terrain(Go2FieldCfg.terrain ):
        num_rows = 10
        num_cols = 10
        selected = "TerrainPerlin"
        TerrainPerlin_kwargs = dict(
            zScale= 0.01,
            frequency= 10,
        )
        unify = False
        max_height = 0.6 
        max_init_terrain_level = 1
        curriculum = False
        measure_heights = True

    class commands(Go2FieldCfg.commands ):
        class ranges(Go2FieldCfg.commands.ranges ):
            lin_vel_x = [0.5, 0.9]
            lin_vel_y = [-0.0, 0.0]
            ang_vel_yaw = [-1.0, 1.0] 
        resampling_time = 5.
        discretize_step = 1.0 # for discretizated commands, only on ang command now

    class domain_rand(Go2FieldCfg.domain_rand ):
        class com_range(Go2FieldCfg.domain_rand.com_range ):
            x = [-0.2, 0.2]
        max_push_vel_ang = 0.5
        push_robots = True
        push_interval_s = 9
        init_base_vel_range = [-1.,1.]
        init_base_rot_range = dict(
            roll= [0, 0],
            pitch= [0, 0],
            yaw= [-3.14,3.14]
        )

    class rewards(Go2FieldCfg.rewards ):
        class scales:
            tracking_ang_vel = 5.0
            tracking_lin_vel = 7.5 
            legs_energy_substeps = -8e-7
            alive = 1
            termination = -5
            
            # # penalty for hardware safety
            collision = -1.0
            orientation = -5.0
            dof_vel=-5e-4
            dof_acc=-5e-7
            dof_pos = -1.0
            
        soft_dof_pos_limit = 0.9
        max_contact_force = 60.0
        tracking_sigma = 0.25
        base_height_target = 0.35
        target_feet_height = 0.04
        cycle_time = 0.64
        
        target_rear_calf = 0.3
        
        
    class termination(Go2FieldCfg.termination):
        termination_terms = [
            "roll",
            "pitch",
            # "z_low",
            # "z_high",
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
        
    class noise(Go2FieldCfg.noise ):
        add_noise = True # disable internal uniform +- 1 noise, and no noise in proprioception
        class noise_scales(Go2FieldCfg.noise.noise_scales ):
            lin_vel = 0.
            robot_friction = 0.
            # As we apply zeroed proprioception
            robot_config = 0.
            height_measurements = 0.
        
    class sensor(Go2FieldCfg.sensor ):
        class proprioception:
            delay_action_obs = False
            latency_range = [0.005, 0.045] # [min, max] in seconds
            # latency_range = [0.04-0.0025, 0.04+0.0075] # [min, max] in seconds
            latency_resample_time = 2.0 # [s]    
    

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go2DiffuserCfgRunner(BaseConfig):
    """
    Runner config for diffusion training and play on Go2.

    This config is primarily tuned for the off-policy training path
    (`DiffuserOffPolicyRunner`). A small set of on-policy fields is kept only
    for backward compatibility with legacy scripts.
    """

    # Global runner selection.
    seed = 3
    runner_class_name = 'DiffuserOffPolicyRunner'

    class policy:
        """
        Diffusion policy and sampling-time behavior.

        These fields are consumed by off-policy training and play-time policy construction.
        Some keys are also read by legacy on-policy code paths.
        Keep `horizon` aligned with `runner.horizon`.
        """
        horizon = 24
        transition_dim = 61
        cond_dim = 49
        observation_dim = cond_dim
        dim = 64
        dim_mults = (1, 2, 4, 8)
        attention = True

        # Kept for backward compatibility. If `action_noise` is missing,
        # off-policy runner falls back to this scalar.
        action_noise_scale = 0.

        num_candidate = 1

        # Per-dimension scaling used before diffusion/reward model inputs.
        diff_obs_scale = [
            0.33, 0.33, 0.33,
            0.5,  0.5,  0.5,
            1.0,  1.0,  1.0,
            0.5,  0.5,  0.5,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
            0.5,
        ]
        diff_action_scale = [
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ]

        # Receding-horizon execution controls.
        num_history = 1
        execute_num = 11
        variance_clip = 0.0
        replan_margin = 4

        # Off-policy action noise schedule used during rollout.
        action_noise = {
            "base_scale": 0.3,
            "schedule_type": "constant",  # linear | cosine | exp_cycle | exponential | custom
            "schedule_params": {
                "decay_rate": 0.98,
                "floor": 0.05,
            },
            "custom_schedule_func": "rsl_rl.utils.noise_schedule.exp_cycle_schedule",
        }

        # Optional denoising cache in play-time policy.
        early_cache = {
            "enable": False,
            "m_steps": 7,
            "reset_threshold": 0.5,
            "reset_type": "cmd",
        }

    class runner:
        """
        Training-loop schedule and diffusion objective setup.

        Main consumer: `DiffuserOffPolicyRunner`.
        """
        horizon = 24
        num_steps_per_env = 50

        # Segment / minibatch sampling.
        mini_batch_size = 8192
        min_traj_len = horizon
        segment_length = horizon
        segment_number = 8192  # on-policy-only legacy field
        traj_num = 1500        # off-policy replay sampling size

        log_interval = 1
        n_timesteps = 10
        loss_type = "l2"
        value_loss_type = "value_l2"
        clip_denoised = True
        predict_epsilon = False
        action_weight = 10.0
        loss_discount = 0.99
        loss_weights = None

        diffusion_training_epochs = 1000
        seed_steps = 0
        warmup_noise = None
        update_interval = 5
        
        # Off-policy replay pruning.
        clear_cycle = 1
        clear_ratio = 0.9
        min_traj_after_clear = 100000
        
        # Checkpoint / experiment bookkeeping.
        resume = True
        load_run = ""
        experiment_name = "diffuser_go2"
        
        run_name = "".join(["Go2WalkByDiffuser",
        ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])
        save_interval= 20
        max_iterations = 150
        checkpoint = -1

    class algorithm:
        """Algorithm-level hyperparameters for off-policy diffusion/reward updates."""
        learning_rate = 1e-4
        diffusion_loss_coeff = 1.0
        reward_loss_coeff = 1.0
        max_grad_norm = 1.0
        temperature = 0.02
        filter_ratio = 0.985

    class reward_model:
        """Reward model training and loading options for off-policy pipeline."""
        class full:
            dim = 32
            dim_mults = (1, 2, 4, 8)
            batch_size = 8192
            num_training_epochs = 3
            traj_num = 10000

        # Used by runner.load(...) when loading reward model from external logs.
        load_external = False
        reward_path = "REWARD_PATH"
        rew_ckpt = -1

    class guided_policy:
        """
        Guided sampling behavior for train/play.

        `sample_kwargs` is passed directly to `GuidedPolicy`.
        Reward/candidate/constraint fields are consumed by `play_diffuser.py`.
        During off-policy training rollout, candidate scoring is disabled by
        enforcing `policy.num_candidate = 1`.
        """
        sample_kwargs = {
            "scale": 1.0,
            "t_stopgrad": 0,
            "n_guide_steps": 1,
            "scale_grad_by_std": False,
            "sample_type": "ddpm",
            "skip_type": "quad",
            "timesteps": 10,
            "eta": 0.991,
        }
        resample_guidance_scale = True

        # Reward model for guidance in play.
        reward_type = "analytic"   # nn | analytic
        reward_name = "LegPosExcursionRew"
        reward_args = {
            "leg": ["RL", "RR"],
            "q_ref": [0.0, 0.5, 1.0]
        }
        reward_load_external = False
        reward_external_path = "logs/diffusion_pretrain/diffusion_pretrain_go2_Aug09_09-42-15/model.pt"

        # Candidate scorer for multi-candidate selection in play. Use with num_candidate > 1 in deployment.
        candidate_same_as_reward = False
        candidate_type = "analytic"  # nn | analytic
        candidate_path = "logs/diffusion_pretrain/diffusion_pretrain_go2_Aug09_09-42-15/model.pt"
        candidate_name = "LegPosExcursionRew"
        candidate_args = {
            "leg": ["RL", "RR"],
            "q_ref": [0.0, 0.5, 1.0]
        }

        # Optional constraints appended to `guided_policy.sample_kwargs` in play.
        apply_constraint = True
        constraint_name = ["BoxConstraint"]
        constraint_args = [
            {
                "indices": "obs_qd",
                "low":  -0.05,
                "high": 0.05,
            },
        ]
        