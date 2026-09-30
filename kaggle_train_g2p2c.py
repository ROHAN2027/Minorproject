"""
G2P2C Kaggle Training Script
=============================
Paste this script into a Kaggle notebook to train the G2P2C model for adult patients.

Steps:
  1. Upload your G2P2C project folder to Kaggle (or clone from GitHub).
  2. Set PATIENT_ID below (20=adult#001, 21=adult#002, ..., 29=adult#010).
  3. Run the notebook with GPU enabled.
  4. Download the saved weights from /kaggle/working/trained_weights/.

For local use: python kaggle_train_g2p2c.py --patient_id 20 --device cpu --debug 1
"""

import os
import sys
import torch
import json
import shutil
import random
import argparse
import numpy as np
import warnings
from copy import deepcopy

# Mute pandas FutureWarnings caused by older simglucose library
warnings.simplefilter(action='ignore', category=FutureWarning)

# ============================================================
# CONFIGURATION - CHANGE THESE FOR DIFFERENT PATIENTS
# ============================================================
# Patient ID mapping for adults:
#   20 = adult#001, 21 = adult#002, ..., 29 = adult#010
# ============================================================

def detect_environment():
    """Detect if running on Kaggle or locally."""
    if os.path.exists('/kaggle/working'):
        return 'kaggle'
    return 'local'


def setup_paths():
    """Setup project paths based on environment."""
    # Local or Kaggle: use the current project directory where the script is located
    main_path = os.path.dirname(os.path.abspath(__file__))

    # Create .env file
    env_file = os.path.join(main_path, '.env')
    with open(env_file, 'w') as f:
        f.write(f'MAIN_PATH={main_path}\n')

    os.environ['MAIN_PATH'] = main_path
    sys.path.insert(0, main_path)
    return main_path


def get_patient_name(patient_id):
    """Convert patient_id to patient name."""
    patients = (
        [f'adolescent#0{str(i).zfill(2)}' for i in range(1, 11)] +
        [f'child#0{str(i).zfill(2)}' for i in range(1, 11)] +
        [f'adult#0{str(i).zfill(2)}' for i in range(1, 11)]
    )
    return patients[patient_id]


def setup_experiment_folders(main_path, folder_id):
    """Create experiment output directories."""
    log_dir = os.path.join(main_path, 'results', folder_id)
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(os.path.join(log_dir, 'checkpoints'))
    os.makedirs(os.path.join(log_dir, 'training', 'data'))
    os.makedirs(os.path.join(log_dir, 'training', 'plots'))
    os.makedirs(os.path.join(log_dir, 'testing', 'data'))
    os.makedirs(os.path.join(log_dir, 'testing', 'plots'))
    os.makedirs(os.path.join(log_dir, 'code'))
    return log_dir


def save_best_weights(main_path, folder_id, patient_id):
    """
    After training, find the last checkpoint and save it with a clean name
    to the trained_weights/ directory.
    """
    checkpoint_dir = os.path.join(main_path, 'results', folder_id, 'checkpoints')
    output_dir = os.path.join(main_path, 'trained_weights')
    os.makedirs(output_dir, exist_ok=True)

    # Find the latest episode checkpoint
    actor_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('_Actor.pth')]
    if not actor_files:
        print("ERROR: No checkpoints found!")
        return

    # Extract episode numbers and find the latest
    episodes = []
    for f in actor_files:
        try:
            ep = int(f.split('_')[1])
            episodes.append(ep)
        except (ValueError, IndexError):
            continue

    if not episodes:
        print("ERROR: Could not parse checkpoint episode numbers!")
        return

    latest_ep = max(episodes)
    patient_name = get_patient_name(patient_id)
    patient_num = patient_name.split('#')[1]  # e.g., "001"

    # Copy with clean naming
    src_actor = os.path.join(checkpoint_dir, f'episode_{latest_ep}_Actor.pth')
    src_critic = os.path.join(checkpoint_dir, f'episode_{latest_ep}_Critic.pth')
    dst_actor = os.path.join(output_dir, f'g2p2c_adult_{patient_num}_Actor.pth')
    dst_critic = os.path.join(output_dir, f'g2p2c_adult_{patient_num}_Critic.pth')

    shutil.copy2(src_actor, dst_actor)
    shutil.copy2(src_critic, dst_critic)

    print(f"\n{'='*60}")
    print(f"WEIGHTS SAVED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Patient: {patient_name} (ID: {patient_id})")
    print(f"Episode: {latest_ep}")
    print(f"Actor:  {dst_actor}")
    print(f"Critic: {dst_critic}")
    print(f"{'='*60}")

    # On Kaggle, also copy to /kaggle/working for easy download
    if detect_environment() == 'kaggle':
        kaggle_out = '/kaggle/working/trained_weights'
        os.makedirs(kaggle_out, exist_ok=True)
        shutil.copy2(dst_actor, os.path.join(kaggle_out, os.path.basename(dst_actor)))
        shutil.copy2(dst_critic, os.path.join(kaggle_out, os.path.basename(dst_critic)))
        print(f"\nKaggle download: /kaggle/working/trained_weights/")


def train(patient_id=20, seed=3, device='cuda', debug=0):
    """
    Train G2P2C for a specific adult patient.

    Args:
        patient_id: 20-29 for adult#001 through adult#010
        seed: random seed for reproducibility
        device: 'cuda' or 'cpu'
        debug: 1 for quick test (4000 interactions), 0 for full training (800K)
    """
    main_path = setup_paths()

    # Import G2P2C modules (after path setup)
    from utils.core import set_logger, get_patient_env
    from agents.g2p2c.g2p2c import G2P2C
    from agents.g2p2c.parameters import set_args

    patient_name = get_patient_name(patient_id)
    patient_num = patient_name.split('#')[1]
    folder_id = f'g2p2c_adult_{patient_num}_seed{seed}'

    print(f"\n{'='*60}")
    print(f"G2P2C TRAINING")
    print(f"{'='*60}")
    print(f"Patient:    {patient_name} (ID: {patient_id})")
    print(f"Seed:       {seed}")
    print(f"Device:     {device}")
    print(f"Debug:      {debug} ({'Quick test' if debug else 'Full training'})")
    print(f"Folder:     {folder_id}")
    print(f"{'='*60}\n")

    # Create args namespace manually (avoiding argparse conflicts in notebooks)
    import argparse
    args = argparse.Namespace()
    args.agent = 'g2p2c'
    args.folder_id = folder_id
    args.patient_id = patient_id
    args.return_type = 'average'
    args.action_type = 'exponential'
    args.device = device
    args.seed = seed
    args.debug = debug
    args.verbose = True
    args.main_dir = main_path
    args.experiment_dir = ''
    args.restart = '1'
    args.m = ''
    args.kl = 1

    # Simulation parameters
    args.sensor = 'GuardianRT'
    args.pump = 'Insulet'
    args.meal_prob = [0.95, -1, 0.95, -1, 0.95, -1]
    args.meal_amount = [45, 30, 85, 30, 80, 30]
    args.meal_variance = [5, 3, 5, 3, 10, 3]
    args.time_variance = [60, 30, 60, 30, 60, 30]
    args.action_scale = 1
    args.insulin_max = 5
    args.insulin_min = 0
    args.glucose_max = 600
    args.glucose_min = 39
    args.target_glucose = 140
    args.use_bolus = True
    args.use_cf = False
    args.glucose_cf_target = 150
    args.expert_bolus = False
    args.expert_cf = False
    args.use_meal_announcement = False
    args.use_carb_announcement = False
    args.use_tod_announcement = False
    args.carb_estimation_method = 'real'
    args.t_meal = 20
    args.n_features = 3
    args.n_handcrafted_features = 0
    args.use_handcraft = 0
    args.feature_history = 48
    args.calibration = 48
    args.max_epi_length = 2000
    args.n_action = 1
    args.n_hidden = 12
    args.n_rnn_layers = 2
    args.rnn_directions = 1
    args.rnn_only = False
    args.bidirectional = False
    args.n_step = 6
    args.gamma = 0.99
    args.lambda_ = 0.95
    args.max_test_epi_len = 1
    args.eps_clip = 0.2
    args.n_vf_epochs = 80
    args.n_pi_epochs = 80
    args.target_kl = 0.05
    args.pi_lr = 1e-3
    args.vf_lr = 1e-3
    args.batch_size = 64
    args.n_training_workers = 20
    args.n_testing_workers = 5
    args.entropy_coef = 0.01
    args.grad_clip = 20
    args.normalize_reward = False
    args.shuffle_rollout = False
    args.aux_mode = 'dual'
    args.aux_lr = 1e-4
    args.aux_buffer_max = 10
    args.aux_frequency = 3
    args.n_aux_epochs = 3
    args.aux_batch_size = 100
    args.aux_vf_coef = 1
    args.aux_pi_coef = 1
    args.use_planning = 'yes'
    args.planning_n_step = 3
    args.n_planning_simulations = 5
    args.n_plan_epochs = 1
    args.plan_batch_size = 1
    args.planning_lr = 1e-4
    args.bgp_pred_mode = False
    args.n_bgp_steps = 0
    args.pretrain_period = 5760
    args.sample_size = 1000
    args.sac_v2 = False
    args.discrete_actions = False
    args.noise_model = 'normal_dist'
    args.noise_application = 1
    args.noise_std = 0.2
    args.soft_tau = 0.005
    args.mu_penalty = 1
    args.action_penalty_limit = 0
    args.action_penalty_coef = 0.1
    args.replay_buffer_type = 'random'
    args.replay_buffer_alpha = 0.6
    args.replay_buffer_beta = 0.4
    args.replay_buffer_temporal_decay = 1
    args.target_action_std = 0.2
    args.target_action_lim = 0.5

    # Apply G2P2C-specific parameter overrides
    args = set_args(args)

    # Setup folders
    log_dir = setup_experiment_folders(main_path, folder_id)
    args.experiment_dir = log_dir
    set_logger(log_dir)

    # Save args
    with open(os.path.join(log_dir, 'args.json'), 'w') as fp:
        json.dump({k: v for k, v in vars(args).items() if not callable(v)}, fp, indent=4, default=str)

    # Set seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Create agent
    agent = G2P2C(args, device, False, '', '')

    # Get patients and environments
    patients, env_ids = get_patient_env()

    # Train
    print("Starting training...")
    try:
        agent.run(args, patients, env_ids, seed)
    except SystemExit:
        pass  # The original G2P2C code calls exit() when done, we must catch it!

    # Save best weights with clean names
    save_best_weights(main_path, folder_id, patient_id)

    print("\nTraining complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train G2P2C on Kaggle or locally')
    parser.add_argument('--patient_id', type=int, default=20,
                        help='Patient ID (20-29 for adult#001 to adult#010)')
    parser.add_argument('--seed', type=int, default=3, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--debug', type=int, default=0,
                        help='1 for quick test, 0 for full training')
    cli_args = parser.parse_args()

    if cli_args.patient_id < 20 or cli_args.patient_id > 29:
        print("WARNING: Patient ID should be 20-29 for adults!")
        print("  20=adult#001, 21=adult#002, ..., 29=adult#010")

    train(
        patient_id=cli_args.patient_id,
        seed=cli_args.seed,
        device=cli_args.device,
        debug=cli_args.debug
    )
