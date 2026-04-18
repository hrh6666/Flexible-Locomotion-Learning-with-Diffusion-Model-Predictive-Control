import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import time
import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import modules from your project.
from rsl_rl.modules.diffusion import GaussianDiffusion, ValueDiffusion
from rsl_rl.modules.temporal import TemporalUnet, ValueFunction

##########################################
# Configuration Helper Function
##########################################
def cfg_to_dict(cfg_obj):
    """
    Recursively converts a configuration class (with nested classes) into a dictionary.
    This is useful for saving and passing configurations around.
    """
    cfg_dict = {}
    for key in dir(cfg_obj):
        # Skip built-in attributes and methods.
        if key.startswith("__"):
            continue
        value = getattr(cfg_obj, key)
        # If the attribute is a nested class, recursively convert it.
        if isinstance(value, type):
            cfg_dict[key] = cfg_to_dict(value)
        # If it's not callable, add it to the dictionary.
        elif not callable(value):
            cfg_dict[key] = value
    return cfg_dict

##########################################
# Example Configuration Class
##########################################
class DiffuserCfgRunner:
    # Global random seed and runner class name.
    seed = 1
    runner_class_name = 'DiffuserRunner'
    
    # Policy configuration (model architecture settings).
    class policy:
        # Model architecture settings.
        horizon = 24 # should be the same as runner.horizon
        transition_dim = 61 # 37
        cond_dim = 49 # 25
        observation_dim = cond_dim
        dim = 64
        dim_mults = (1, 2, 4, 8)
        attention = True
        action_noise_scale = None
        num_candidate = 1
        # Scale factors for diffusion on observations and actions.
        # There is one scale value for each observation and action dimension.
        diff_obs_scale = [
            0.33, 0.33, 0.33,         # Base linear velocity
            0.5,  0.5,  0.5,          # Base angular velocity
            1.0,  1.0,  1.0,          # Projected gravity
            0.5,  0.5,  0.5,          # Command
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # DOF positions
            #### currently not needed, future may use. preserve for now ####
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # DOF velocities
            # Last actions for each limb (4 limbs × 3 joints each)
            1.0, 1.0, 1.0,           # FL_HipX, FL_HipY, FL_Knee
            1.0, 1.0, 1.0,           # FR_HipX, FR_HipY, FR_Knee
            1.0, 1.0, 1.0,           # HL_HipX, HL_HipY, HL_Knee
            1.0, 1.0, 1.0,           # HR_HipX, HR_HipY, HR_Knee
            0.5,                     # Friction term
        ]
        diff_action_scale = [
            1.0, 1.0, 1.0,         
            1.0, 1.0, 1.0,         
            1.0, 1.0, 1.0,         
            1.0, 1.0, 1.0,         
        ]
        num_history = 1
        execute_num = 6
        variance_clip = 0.0

    class runner:
        # Training runner details.
        horizon = 24 # should be the same as policy.horizon
        num_steps_per_env = 30
        mini_batch_size = 8192
        min_traj_len = horizon
        segment_length = horizon
        segment_number = 8192 # Only for on-policy, determines the number of segments to sample
        traj_num = 20 # Only for off-policy, determines the number of trajectories to sample from replay buffer
        log_interval = 1
        n_timesteps = 10 # Number of diffusion steps
        loss_type = "l2"
        value_loss_type = "value_l2"
        clip_denoised = True # All set to False now
        predict_epsilon = False
        action_weight = 10.0
        loss_discount = 0.99
        loss_weights = None
        diffusion_training_epochs = 500
        seed_steps = 50 # off policy okay, when on policy, only works with warmup noise.
        warmup_noise = None # After Scaling
        
        # For Replay Buffer Clearance, only for off-policy
        clear_cycle = 10 
        clear_ratio = 0.9 
        min_traj_after_clear = 1000 
        
        resume = False
        load_run = "May08_07-41-10_WalkByDiffuser_fromMay07_04-56-40"
        experiment_name = "diffuser_go2"
        
        run_name = "".join(["Go2WalkByDiffuser",
        ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])
        save_interval= 20000
        max_iterations = 5000
        checkpoint = -1

    class algorithm:
        # Optimization hyperparameters.
        learning_rate = 1e-4
        diffusion_loss_coeff = 1.0
        reward_loss_coeff = 1.0
        # skill_reward_loss_coeff = 1.0
        max_grad_norm = 1.0
        temperature = None
        filter_ratio = 0.9

    class reward_model:
        class full:
            dim = 32
            dim_mults = (1, 2, 4, 8)
            batch_size = 8192
            num_training_epochs = 100
            traj_num = 100
        class skill:
            dim = 32
            dim_mults = (1, 2, 4, 8)
        class transition:
            hidden_dims = [128, 64, 64]
            num_training_epochs = 100
            traj_num = 0 # Number of trajectories to sample from replay buffer in candidate reward model training
        
    class guided_policy:
        # Guided sampling parameters.
        sample_kwargs = {
            "scale": 100.0,
            "t_stopgrad": 0,
            "n_guide_steps": 0,
            "scale_grad_by_std": True,
            "sample_type": "ddpm",
            "skip_type": "uniform",
            "timesteps": 20,
            "eta": 0.991,
        }

##########################################
# TrajectorySegmentDataset Definition
##########################################
class TrajectorySegmentDataset(Dataset):
    def __init__(self, data_dir, horizon, num_history, diff_obs_scale, diff_action_scale, dataset_name=None):
        """
        Loads trajectory files (saved via torch.save) from the given data directory,
        and samples fixed-length segments using a sliding window approach.
        For each trajectory, the first frame (observations, actions, rewards) is duplicated
        (num_history - 1) times and prepended.

        Additionally, builds an 'all_traj_dict' that aggregates all trajectory data for the keys
        "obs", "action", and "rew". 

        Parameters:
            data_dir (str): The directory containing the trajectory files.
            horizon (int): The length of each trajectory segment.
            num_history (int): The number of history frames to include.
            dataset_name (str, optional): Specific dataset filename (without extension) to load.
        """
        self.horizon = horizon
        self.num_history = num_history
        self.diff_obs_scale = np.array(diff_obs_scale, dtype=np.float32)
        self.diff_action_scale = np.array(diff_action_scale, dtype=np.float32)
        self.segments = []

        # Determine which files to load.
        if dataset_name is not None:
            if isinstance(dataset_name, (list, tuple)):
                files = [os.path.join(data_dir, n + ".pt") if n.endswith(".pt")
                        else os.path.join(data_dir, n, n.split('/')[-1] + ".pt")
                        for n in dataset_name]
            else:
                name = dataset_name
                files = [os.path.join(data_dir, name + ".pt") if name.endswith(".pt")
                        else os.path.join(data_dir, name, name.split('/')[-1] + ".pt")]
        else:
            files = glob.glob(os.path.join(data_dir, "**/*.pt"), recursive=True)

        print("Using dataset files:", files)

        # First pass: Load all trajectories from files.
        all_trajs = []
        for file in files:
            traj_list = torch.load(file)
            all_trajs.extend(traj_list)

        # Build all_traj_dict: keys "obs", "action", and "rew" each map to a list of trajectory data.
        self.all_traj_dict = {"obs": [], "action": [], "rew": []}
        for traj in all_trajs:
            self.all_traj_dict["obs"].append(traj["obs"])
            self.all_traj_dict["action"].append(traj["action"])
            self.all_traj_dict["rew"].append(traj["rew"])

        # Second pass: Process each trajectory into fixed-length segments.
        for traj in all_trajs:
            # Ensure entries are lists.
            obs_traj = list(traj["obs"])
            rew_traj = list(traj["rew"])
            action_traj = list(traj["action"])

            # Prepend the first frame if history is required.
            if self.num_history > 1:
                obs_prefix = [obs_traj[0]] * (self.num_history - 1)
                rew_prefix = [rew_traj[0]] * (self.num_history - 1)
                action_prefix = [action_traj[0]] * (self.num_history - 1)
                obs_traj = obs_prefix + obs_traj
                rew_traj = rew_prefix + rew_traj
                action_traj = action_prefix + action_traj

            traj_length = len(obs_traj)
            if traj_length >= horizon:
                for i in range(traj_length - horizon + 1):
                    seg = {
                        "obs": np.array(obs_traj[i:i + horizon]),
                        "rew": np.array(rew_traj[i:i + horizon]),
                        "action": np.array(action_traj[i:i + horizon]),
                    }
                    # NOTE: The scaling multiplication has been removed.
                    self.segments.append(seg)
        print(f"Loaded {len(files)} files and sampled {len(self.segments)} segments.")
    
    def pre_normalize_segments(self):
        """
        Preprocesses all segments by normalizing the "obs" and "action" data and then 
        concatenating specific channels of "obs".

        Steps:
            For "obs", concatenate the first 24 columns with the columns starting from index 48 
            along the last axis.
        
        After calling this function, the segments stored in self.segments are preprocessed.
        The __getitem__ method can then simply convert these processed numpy arrays into torch.Tensors.
        """
        for seg in self.segments:
            seg["obs"] = seg["obs"] * self.diff_obs_scale
            seg["action"] = seg["action"] * self.diff_action_scale
            # seg["obs"] = np.concatenate([seg["obs"][..., :24], seg["obs"][..., 48:]], axis=-1)
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        """
        Retrieves a segment at the given index and converts its data from numpy arrays 
        to torch.Tensor.
        
        Note:
            It is assumed that pre_normalize_segments has been called.
        """
        seg = self.segments[idx]
        seg["obs"] = torch.tensor(seg["obs"], dtype=torch.float32)
        seg["rew"] = torch.tensor(seg["rew"], dtype=torch.float32)
        seg["action"] = torch.tensor(seg["action"], dtype=torch.float32)
        return seg

##########################################
# DiffusionPretrainTrainer Definition
##########################################
class DiffusionPretrainTrainer:
    def __init__(self, cfg, device='cpu', data_dir="data", dataset_name=None):
        """
        Initializes the trainer with the given configuration, device, and dataset.
        
        Builds the diffusion model, reward model, optimizer, and
        constructs the dataset and dataloader.
        
        After loading trajectories and constructing segments, the trainer uses the
        all_traj_dict from the dataset to stack all observation and action data into large arrays.
        """
        self.cfg = cfg
        self.device = device
        self.horizon = self.cfg["runner"]["horizon"]
        self.observation_dim = self.cfg["env"]["num_obs"]
        self.action_dim = self.cfg["env"]["num_actions"]

        # ----------------- Build Diffusion Model ---------------------
        self.diffuser_model = TemporalUnet(
            horizon=self.cfg["policy"]["horizon"],
            transition_dim=self.cfg["policy"]["transition_dim"],
            cond_dim=self.cfg["policy"]["cond_dim"],
            dim=self.cfg["policy"].get("dim", 32),
            dim_mults=self.cfg["policy"].get("dim_mults", (1, 2, 4, 8)),
            attention=self.cfg["policy"].get("attention", True),
        ).to(self.device)

        self.diffuser = GaussianDiffusion(
            self.diffuser_model,
            horizon=self.horizon,
            observation_dim=self.cfg["env"]["num_obs"],
            action_dim=self.cfg["env"]["num_actions"],
            n_timesteps=self.cfg["runner"]["n_timesteps"],
            loss_type=self.cfg["runner"]["loss_type"],
            clip_denoised=self.cfg["runner"]["clip_denoised"],
            predict_epsilon=self.cfg["runner"]["predict_epsilon"],
            action_weight=self.cfg["runner"]["action_weight"],
            loss_discount=self.cfg["runner"]["loss_discount"],
            loss_weights=self.cfg["runner"]["loss_weights"],
            num_history=self.cfg["policy"]["num_history"],
            variance_clip=self.cfg["policy"]["variance_clip"],
        )

        # ----------------- Build Reward Model ------------------------
        self.reward_model = ValueDiffusion(
            model=ValueFunction(
                horizon=self.horizon,
                transition_dim=self.observation_dim + self.action_dim,
                cond_dim=self.observation_dim,
                dim=self.cfg["reward_model"]["full"]["dim"],
                dim_mults=self.cfg["reward_model"]["full"]["dim_mults"],
                out_dim=1,
            ),
            horizon=self.horizon,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            n_timesteps=self.cfg["runner"]["n_timesteps"],
            loss_type=self.cfg["runner"]["value_loss_type"],
            clip_denoised=self.cfg["runner"]["clip_denoised"],
            predict_epsilon=self.cfg["runner"]["predict_epsilon"],
            action_weight=self.cfg["runner"]["action_weight"],
            loss_discount=self.cfg["runner"]["loss_discount"],
            loss_weights=self.cfg["runner"]["loss_weights"],
        ).to(self.device)

        # ------------- Optimizer for Diffuser and Reward Model -------------
        self.optimizer = torch.optim.Adam(
            list(self.diffuser.parameters()) + list(self.reward_model.parameters()),
            lr=self.cfg["algorithm"]["learning_rate"]
        )
        self.max_gradnorm = self.cfg["algorithm"].get("max_grad_norm", 1.0)

        # -------------- Build Dataset and DataLoader ------------------
        self.dataset = TrajectorySegmentDataset(
            data_dir,
            horizon=self.horizon,
            num_history=self.cfg["policy"]["num_history"],
            diff_obs_scale=self.cfg["policy"]["diff_obs_scale"],
            diff_action_scale=self.cfg["policy"]["diff_action_scale"],
            dataset_name=dataset_name
        )

        self.dataset.pre_normalize_segments()
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.cfg["runner"]["mini_batch_size"],
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        self.current_learning_iteration = 0
        self.tot_time = 0.0

        self.writer = SummaryWriter(log_dir=self.cfg["log_dir"]) if "log_dir" in self.cfg else None

    def collate_fn(self, batch):
        """
        Returns:
            obs:     [T,  B, 49]
            action:  [T,  B, 12]
            rew:     [T,  B, 1]
        """
        obs    = torch.stack([b["obs"]    for b in batch], dim=1)   # no squeeze
        action = torch.stack([b["action"] for b in batch], dim=1)   # no squeeze
        rew    = torch.stack([b["rew"]    for b in batch], dim=1).unsqueeze(-1)
        return {"obs": obs, "action": action, "rew": rew}

    def log(self, log_dict, width=80, pad=35):
        """
        Logs training statistics to TensorBoard (if available) and prints a formatted log string.
        """
        if self.writer is not None:
            self.writer.add_scalar("Train/TotalLoss", log_dict["total_loss"], self.current_learning_iteration)
            self.writer.add_scalar("Train/DiffusionLoss", log_dict["diffusion_loss"], self.current_learning_iteration)
            self.writer.add_scalar("Train/RewardLoss", log_dict["reward_loss"], self.current_learning_iteration)
            self.writer.add_scalar("Train/IterationTime", log_dict["iteration_time"], self.current_learning_iteration)
        
        log_string = f"\n{'#' * width}\n"
        log_string += f"{('Iteration ' + str(self.current_learning_iteration)).center(width, ' ')}\n\n"
        log_string += f"{'Total Loss:':>{pad}} {log_dict['total_loss']:.4f}\n"
        log_string += f"{'Diffusion Loss:':>{pad}} {log_dict['diffusion_loss']:.4f}\n"
        log_string += f"{'Reward Loss:':>{pad}} {log_dict['reward_loss']:.4f}\n"
        log_string += f"{'Iteration Time:':>{pad}} {log_dict['iteration_time']:.4f}s\n"
        log_string += f"{'#' * width}\n"
        print(log_string)

    def save(self, path):
        """
        Saves the current model state including diffuser and reward model.
        """
        run_state_dict = {
            'diffuser_state_dict': self.diffuser.state_dict(),
            'reward_model_state_dict': self.reward_model.state_dict(),
            'iter': self.current_learning_iteration,
        }
        torch.save(run_state_dict, path)
        print(f"Saved model at iteration {self.current_learning_iteration} to {path}")

    def load(self, path, load_optimizer=True):
        """
        Loads the model state from the specified file path. Restores the diffuser and reward model.
        """
        loaded_dict = torch.load(path, map_location=self.device)
        self.diffuser.load_state_dict(loaded_dict['diffuser_state_dict'], strict=False)
        self.reward_model.load_state_dict(loaded_dict['reward_model_state_dict'], strict=False)
        if load_optimizer and 'optimizer_state_dict' in loaded_dict:
            self.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        
        
        print(f"Loaded model from {path} at iteration {self.current_learning_iteration}")
        return loaded_dict.get('infos', None)

    def train(self, num_epochs):
        """
        Trains the models for the specified number of epochs.
        For each batch, computes diffusion loss and reward loss,
        performs backpropagation and optimizer updates, logs statistics, and saves checkpoints periodically.
        """
        for epoch in range(num_epochs):
            for batch in self.dataloader:
                iter_start_time = time.time()
                # Prepare training data by concatenating actions and observations.
                batch_transitions = torch.cat([batch["action"], batch["obs"]], dim=-1).to(self.device).transpose(0, 1)
                num_history = self.cfg["policy"].get("num_history", 1)
                batch_cond = {i: batch["obs"][i].to(self.device) for i in range(num_history)}
                target_reward = batch["rew"].mean(dim=0).to(self.device)
                temperature = self.cfg["algorithm"].get("temperature", None)
                if temperature is not None:
                    r_max = target_reward.max()
                    reward_weights = torch.exp((target_reward - r_max) / temperature)
                    reward_weights = reward_weights / reward_weights.sum() * len(target_reward)
                else:
                    reward_weights = None
                loss_diff, _ = self.diffuser.loss(batch_transitions, batch_cond, reward_weights=reward_weights)
                t = torch.randint(0, self.reward_model.n_timesteps, (batch["obs"].shape[1],), device=self.device).long()
                loss_rew, _ = self.reward_model.p_losses(batch_transitions, batch_cond, target_reward, t)

                total_loss = self.cfg["algorithm"]["diffusion_loss_coeff"] * loss_diff + \
                             self.cfg["algorithm"]["reward_loss_coeff"] * loss_rew

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.diffuser.parameters()) + list(self.reward_model.parameters()),
                                                 self.max_gradnorm)
                self.optimizer.step()

                iter_time = time.time() - iter_start_time
                self.tot_time += iter_time
                self.current_learning_iteration += 1

                log_dict = {
                    "total_loss": total_loss.item(),
                    "diffusion_loss": loss_diff.item(),
                    "reward_loss": loss_rew.item(),
                    "iteration_time": iter_time,
                }
                self.log(log_dict)

                if self.current_learning_iteration % self.cfg["runner"]["save_interval"] == 0:
                    save_path = os.path.join(self.cfg["log_dir"], f"diffusion_pretrain_model_{self.current_learning_iteration}.pt")
                    self.save(save_path)

            print(f"Epoch {epoch+1}/{num_epochs} completed.")
        final_path = os.path.join(self.cfg["log_dir"], "diffusion_pretrain_model_final.pt")
        self.save(final_path)
        print(f"Training finished, model saved to {final_path}")

##########################################
# Main entry point
##########################################
if __name__ == '__main__':
    import argparse
    import datetime
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", nargs='+', type=str, required=True,
        help="One or more dataset names under the data folder (omit the .pt extension), e.g. --dataset traj1 traj2"
    )
    args = parser.parse_args()
    
    cfg_obj = DiffuserCfgRunner()
    cfg_dict = cfg_to_dict(cfg_obj)
    
    cfg_dict["dataset_names"] = args.dataset # List of dataset names to load.
    
    # Add environment parameters.
    cfg_dict["env"] = {
        "num_obs": 49,       # Set according to your environment's observation dimension.
        "num_actions": 12,   # Set according to your environment's action dimension.
    }
    
    base_log_dir = "logs/diffusion_pretrain"
    os.makedirs(base_log_dir, exist_ok=True)
    
    run_date = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    run_name = f"diffusion_pretrain_go2_{run_date}"
    run_log_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    
    cfg_dict["log_dir"] = run_log_dir
    
    config_save_path = os.path.join(run_log_dir, "config.json")
    with open(config_save_path, "w") as f:
        json.dump(cfg_dict, f, indent=4)
    print(f"Configuration saved to {config_save_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = DiffusionPretrainTrainer(cfg_dict, device=device, data_dir="data", dataset_name=args.dataset)
    
    num_epochs = cfg_dict["runner"].get("max_iterations", 200)
    
    trainer.train(num_epochs)