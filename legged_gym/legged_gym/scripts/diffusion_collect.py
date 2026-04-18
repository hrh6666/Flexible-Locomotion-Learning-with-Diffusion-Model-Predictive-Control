import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import time
import numpy as np

# --- Isaac Gym / Torch imports ------------------------------------------------
import isaacgym
from isaacgym import gymtorch, gymapi
import torch
import json

# --- Legged Gym imports -------------------------------------------------------
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import task_registry
from legged_gym.utils import get_args, export_policy_as_jit
from datetime import datetime

# -----------------------------------------------------------------------------


def create_recording_camera(
        gym, env_handle,
        resolution=(1920, 1080),
        h_fov=86,
        actor_to_attach=None,
        transform=None,
):
    """Create a camera sensor (optionally attached to a body)."""
    camera_props = gymapi.CameraProperties()
    camera_props.enable_tensors = True
    camera_props.width = resolution[0]
    camera_props.height = resolution[1]
    camera_props.horizontal_fov = h_fov
    camera_handle = gym.create_camera_sensor(env_handle, camera_props)
    if actor_to_attach is not None:
        gym.attach_camera_to_body(camera_handle, env_handle, actor_to_attach,
                                  transform, gymapi.FOLLOW_POSITION)
    elif transform is not None:
        gym.set_camera_transform(camera_handle, env_handle, transform)
    return camera_handle


def collect(args):
    # -------------------------------------------------------------------------
    # 1) Build env & policy
    # -------------------------------------------------------------------------
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    train_cfg.runner.resume = True

    # -- parallel envs --------------------------------------------------------
    env_cfg.env.num_envs = 128                # << parallel collection >>
    # ------------------------------------------------------------------------
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.terrain_length = 30
    env_cfg.terrain.terrain_width = 30
    env_cfg.init_state.pos = [0, 0, 0.32]
    env_cfg.domain_rand.push_robots = True
    env_cfg.terrain.unify = False
    env_cfg.terrain.max_Zscale = 0.0
    env_cfg.domain_rand.init_base_vel_range = [0, 0]
    env_cfg.domain_rand.init_base_rot_range = dict(roll=[-0.2, 0.2],
                                                  pitch=[-0.2, 0.2],
                                                  yaw=[0, 0])
    env_cfg.viewer.debug_viz = True
    env_cfg.viewer.draw_volume_sample_points = False
    env_cfg.commands.ranges.lin_vel_x = [-1.0, 1.0]
    env_cfg.commands.ranges.lin_vel_y = [-0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]
    env_cfg.commands.resampling_time = int(1e16)

    # viewer pose for multi-env or single-env
    if env_cfg.env.num_envs > 1:
        env_cfg.viewer.pos, env_cfg.viewer.lookat = [4., 6., 1], [5., -7., 0]
        # env_cfg.viewer.pos, env_cfg.viewer.lookat = [4., 5.4, 0.2], [4., 3.8, 0.2]
    else:
        env_cfg.viewer.pos, env_cfg.viewer.lookat = [4., -6., 1], [5., -7., 0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, save_cfg=False)

    policy = ppo_runner.alg.act_play
    if args.sample:                         # stochastic sampling flag
        policy = ppo_runner.alg.actor_critic.act

    # -------------------------------------------------------------------------
    # 2) Pre-allocate tensor buffers (no Python loop per env)
    # -------------------------------------------------------------------------
    obs_dim = 49
    act_dim = 12
    num_envs = env_cfg.env.num_envs
    max_steps = int(3e4)

    # CPU buffers (move to GPU if you truly need the speed)
    obs_buffer   = torch.zeros((max_steps, num_envs, obs_dim), dtype=torch.float32)
    rew_buffer   = torch.zeros((max_steps, num_envs),           dtype=torch.float32)
    act_buffer   = torch.zeros((max_steps, num_envs, act_dim),  dtype=torch.float32)

    # index of first frame for current episode of each env
    episode_start = torch.zeros(num_envs, dtype=torch.int64)
    trajectories = []  # list of finished episodes

    # -------------------------------------------------------------------------
    # 3) Random command initialisation
    # -------------------------------------------------------------------------
    cmd_x = np.random.uniform(*env_cfg.commands.ranges.lin_vel_x)
    cmd_y = np.random.uniform(*env_cfg.commands.ranges.lin_vel_y)
    cmd_yaw = np.random.uniform(*env_cfg.commands.ranges.ang_vel_yaw)
    env.commands[:, 0] = cmd_x
    env.commands[:, 1] = cmd_y
    env.commands[:, 2] = cmd_yaw

    # -------------------------------------------------------------------------
    # 4) Optional viewer recording
    # -------------------------------------------------------------------------
    img_idx = 0
    if RECORD_FRAMES:
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(*env_cfg.viewer.pos)
        transform.r = gymapi.Quat.from_euler_zyx(0., 0., -np.pi/2)
        create_recording_camera(env.gym, env.envs[0], transform=transform)
        images_dir = os.path.join("logs", "images")
        os.makedirs(images_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 5) Main simulation loop (tensor-wise logging, no per-env loop)
    # -------------------------------------------------------------------------
    step = 0
    while step < max_steps:
        # sample a new command every 250 steps
        if step % 250 == 0:
            # sample independent commands for each env (vectorized)
            cmd_x  = torch.rand(num_envs, device=env.sim_device) \
                    * (env_cfg.commands.ranges.lin_vel_x[1] - env_cfg.commands.ranges.lin_vel_x[0]) \
                    +  env_cfg.commands.ranges.lin_vel_x[0]

            cmd_y  = torch.rand(num_envs, device=env.sim_device) \
                    * (env_cfg.commands.ranges.lin_vel_y[1] - env_cfg.commands.ranges.lin_vel_y[0]) \
                    +  env_cfg.commands.ranges.lin_vel_y[0]

            cmd_yaw = torch.rand(num_envs, device=env.sim_device) \
                    * (env_cfg.commands.ranges.ang_vel_yaw[1] - env_cfg.commands.ranges.ang_vel_yaw[0]) \
                    +  env_cfg.commands.ranges.ang_vel_yaw[0]

            # write back in a single tensor op: shape (num_envs, 3)
            env.commands[:, 0] = cmd_x
            env.commands[:, 1] = cmd_y
            env.commands[:, 2] = cmd_yaw

            # optional debug print (show first 3 envs)
            print(f"New cmds @ {step}:",
                f"x={cmd_x[:3].cpu().numpy()}",
                f"y={cmd_y[:3].cpu().numpy()}",
                f"yaw={cmd_yaw[:3].cpu().numpy()}")

        # fetch obs & camera follow (optional)
        obs = env.get_observations()
        if MOVE_CAMERA:
            cam_dir = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
            cam_pos = env.root_states[:, :3].mean(0).cpu().numpy() - cam_dir
            env.set_camera(cam_pos, cam_pos + cam_dir)

        # rollout one step
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, rewards, dones, infos = env.step(actions.detach())

        # >>> tensorised storage (single assignment) <<<
        obs_buffer[step]   = obs.cpu()
        rew_buffer[step]   = rewards.cpu()
        act_buffer[step]   = actions.cpu()

        # optional frame dump
        if RECORD_FRAMES:
            file_path = os.path.join(images_dir, f"{img_idx:06d}.png")
            env.gym.write_viewer_image_to_file(env.viewer, file_path)
            img_idx += 1

        # detect episode termination
        done_cpu = dones.cpu()
        if done_cpu.any():
            done_idxs = torch.nonzero(done_cpu, as_tuple=False).view(-1)
            for idx in done_idxs.tolist():
                st, ed = int(episode_start[idx]), step + 1
                traj = {
                    "obs":          obs_buffer[st:ed, idx].numpy(),
                    "rew":          rew_buffer[st:ed, idx].numpy(),
                    "action":       act_buffer[st:ed, idx].numpy(),
                }
                trajectories.append(traj)
                episode_start[idx] = ed  # next episode starts after current step
        step += 1

    # flush unfinished episodes at the end (if any)
    for idx in range(num_envs):
        st, ed = int(episode_start[idx]), step
        if ed > st:
            traj = {
                "obs":          obs_buffer[st:ed, idx].numpy(),
                "rew":          rew_buffer[st:ed, idx].numpy(),
                "action":       act_buffer[st:ed, idx].numpy(),
            }
            trajectories.append(traj)

    # -------------------------------------------------------------------------
    # 6) Persist trajectories + config
    # -------------------------------------------------------------------------
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    subdir = os.path.join("data", f"trajectory_{run_tag}")
    os.makedirs(subdir, exist_ok=True)

    pt_path = os.path.join(subdir, f"trajectory_{run_tag}.pt")
    torch.save(trajectories, pt_path)
    print(f"Trajectory data saved to {pt_path}")

    def cfg_to_plain_dict(cfg):
        """Convert OmegaConf or namespace cfg to plain Python dict."""
        try:
            from omegaconf import OmegaConf
            return OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}

    cfg_dump = {
        "commands": {
            "lin_vel_x": env_cfg.commands.ranges.lin_vel_x,
            "lin_vel_y": env_cfg.commands.ranges.lin_vel_y,
            "ang_vel_yaw": env_cfg.commands.ranges.ang_vel_yaw
        },
        "action_scale": (
            env.cfg.control.action_scale.detach().cpu().tolist()
            if torch.is_tensor(env.cfg.control.action_scale)
            else float(env.cfg.control.action_scale)
        ),
        "max_steps": max_steps,
        "num_envs": num_envs,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "load_run": train_cfg.runner.load_run,
    }
    with open(os.path.join(subdir, "config.json"), "w") as f:
        json.dump(cfg_dump, f, indent=4)
    print(f"Config saved to {os.path.join(subdir, 'config.json')}")


# ------------------------------------------------------------------------------
# entry
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args([
        dict(name="--slow", type=float, default=0., help="Slow down the simulation by sleep secs (float) every frame"),
        dict(name="--zero_act_until", type=int, default=0, help="Zero action until this step"),
        dict(name="--sample", action="store_true", default=False, help="Sample actions from policy"),
        dict(name="--plot_time", type=int, default=100, help="Plot states after this time"),
        dict(name="--no_throw", action="store_true", default=False),
        dict(name="--record_frames", action="store_true", default=False, help="Record camera frames")
    ])
    collect(args)