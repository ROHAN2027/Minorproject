"""
G2P2C Simulation Console
=========================
A Streamlit web app for running glucose control simulations using trained G2P2C models.
Matches the CAPSML-style dark-themed simulation UI.

Usage:
    streamlit run simulation_app.py

Requirements:
    pip install streamlit plotly
"""

import os
import sys
import math
import glob
import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from copy import deepcopy
from collections import deque

# ============================================================
# Path Setup
# ============================================================
MAIN_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MAIN_PATH)

# Create .env if not exists
env_file = os.path.join(MAIN_PATH, '.env')
if not os.path.exists(env_file):
    with open(env_file, 'w') as f:
        f.write(f'MAIN_PATH={MAIN_PATH}\n')
os.environ['MAIN_PATH'] = MAIN_PATH

from utils.core import linear_scaling, inverse_linear_scaling, risk_index, time_in_range

# ============================================================
# Page Config & Custom CSS
# ============================================================
st.set_page_config(
    page_title="G2P2C Simulation Console",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark theme CSS matching the reference screenshots
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #1a2f2a;
        color: #e0e0e0;
    }

    /* Header */
    .sim-header {
        background: linear-gradient(135deg, #2d1b1b, #3d2020);
        padding: 12px 24px;
        border-radius: 8px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .sim-header h2 {
        color: #ff6b6b;
        margin: 0;
        font-size: 18px;
        letter-spacing: 2px;
        font-weight: 600;
    }

    /* Card containers */
    .sim-card {
        background-color: #1e3a34;
        border: 1px solid #2d524a;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }

    /* Metric cards */
    .metric-card {
        background-color: #1e3a34;
        border: 1px solid #2d524a;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: left;
    }
    .metric-card h4 {
        color: #8fbc8f;
        font-size: 13px;
        font-weight: 400;
        margin: 0 0 8px 0;
    }
    .metric-card .metric-value {
        color: #ffffff;
        font-size: 28px;
        font-weight: 600;
        margin: 0;
    }

    /* Run button */
    .run-btn-container .stButton > button {
        background: linear-gradient(135deg, #8b3030, #6b2020);
        color: white;
        width: 100%;
        padding: 14px;
        font-size: 18px;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        letter-spacing: 1px;
    }
    .run-btn-container .stButton > button:hover {
        background: linear-gradient(135deg, #a04040, #8b3030);
    }

    /* Labels */
    .meal-label {
        color: #c4a882;
        font-weight: 500;
        font-size: 14px;
    }
    .input-label {
        color: #8fbc8f;
        font-size: 12px;
        margin-bottom: 4px;
    }

    /* Section headers */
    .section-header {
        color: #c4a882;
        font-weight: 600;
        font-size: 16px;
        margin-bottom: 12px;
    }

    /* Select boxes and inputs */
    .stSelectbox > div > div {
        background-color: #1e3a34;
        border-color: #2d524a;
        color: #e0e0e0;
    }
    .stNumberInput > div > div > input {
        background-color: #1e3a34;
        border-color: #2d524a;
        color: #e0e0e0;
    }
    .stTimeInput > div > div > input {
        background-color: #1e3a34;
        border-color: #2d524a;
        color: #e0e0e0;
    }

    /* Legend badges */
    .legend-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 12px;
        margin-right: 8px;
        margin-bottom: 4px;
    }
    .legend-glucose { background-color: #1a3a4a; color: #00d4ff; }
    .legend-insulin { background-color: #2a2a3a; color: #8888cc; }
    .legend-carb { background-color: #3a1a1a; color: #ff6b6b; }
    .legend-tir { background-color: #1a3a1a; color: #4caf50; }
    .legend-tbr { background-color: #3a3a1a; color: #ffeb3b; }
    .legend-tar { background-color: #3a2a1a; color: #ff9800; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Fix dark text in widgets */
    .stSelectbox label, .stNumberInput label, .stTimeInput label {
        color: #8fbc8f !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Helper Functions
# ============================================================

def get_available_patients():
    """Scan trained_weights/ folder and return list of available adult patients."""
    weights_dir = os.path.join(MAIN_PATH, 'trained_weights')
    if not os.path.exists(weights_dir):
        return []

    available = []
    for i in range(1, 11):
        num = str(i).zfill(3)
        actor_path = os.path.join(weights_dir, f'g2p2c_adult_{num}_Actor.pth')
        critic_path = os.path.join(weights_dir, f'g2p2c_adult_{num}_Critic.pth')
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            available.append({
                'id': 19 + i,  # patient_id: 20-29
                'num': num,
                'name': f'adult#{num}',
                'label': f'{i}',
                'actor_path': actor_path,
                'critic_path': critic_path
            })
    return available


def create_sim_args(patient_id, meal_amounts, meal_times_minutes):
    """Create args namespace for simulation."""
    class Args:
        pass

    args = Args()
    args.agent = 'g2p2c'
    args.patient_id = patient_id
    args.sensor = 'GuardianRT'
    args.pump = 'Insulet'
    args.action_type = 'exponential'
    args.action_scale = 5
    args.insulin_max = 5
    args.insulin_min = 0
    args.glucose_max = 600
    args.glucose_min = 39
    args.t_meal = 20
    args.n_features = 2
    args.n_handcrafted_features = 1
    args.use_handcraft = 0
    args.feature_history = 12
    args.calibration = 12
    args.n_action = 1
    args.n_hidden = 16
    args.n_rnn_layers = 1
    args.rnn_directions = 1
    args.bidirectional = False
    args.max_epi_length = 288 * 10
    args.max_test_epi_len = 288
    args.n_step = 256
    args.return_type = 'average'
    args.gamma = 1
    args.lambda_ = 1
    args.entropy_coef = 0.001
    args.grad_clip = 20
    args.eps_clip = 0.1
    args.target_kl = 0.01
    args.normalize_reward = True
    args.shuffle_rollout = True
    args.n_training_workers = 2
    args.n_testing_workers = 2
    args.n_pi_epochs = 5
    args.n_vf_epochs = 5
    args.pi_lr = 3e-4
    args.vf_lr = 3e-4
    args.batch_size = 1024
    args.aux_mode = 'dual'
    args.aux_lr = 3e-4
    args.aux_buffer_max = 1000
    args.aux_frequency = 1
    args.aux_vf_coef = 0.01
    args.aux_pi_coef = 0.01
    args.aux_batch_size = 1024
    args.n_aux_epochs = 5
    args.use_planning = 'yes'
    args.planning_n_step = 6
    args.n_planning_simulations = 50
    args.plan_batch_size = 1024
    args.n_plan_epochs = 1
    args.planning_lr = 3e-4
    args.verbose = False
    args.debug = 0
    args.seed = 0
    args.kl = 1
    args.expert_bolus = False
    args.expert_cf = False
    args.use_meal_announcement = False
    args.use_carb_announcement = False
    args.use_tod_announcement = False
    args.target_glucose = 140
    args.use_bolus = True
    args.use_cf = False
    args.glucose_cf_target = 150
    args.carb_estimation_method = 'real'
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

    # Meal configuration - deterministic for simulation
    # meal_amounts = [breakfast_carbs, snack1, lunch_carbs, snack2, dinner_carbs, snack3]
    args.meal_amount = meal_amounts
    args.meal_time = meal_times_minutes
    args.meal_variance = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
    args.time_variance = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
    args.meal_prob = [1 if meal_amounts[0] > 0 else -1, -1,
                      1 if meal_amounts[2] > 0 else -1, -1,
                      1 if meal_amounts[4] > 0 else -1, -1]

    # Experiment dir (temporary)
    import tempfile
    args.experiment_dir = tempfile.mkdtemp()
    args.folder_id = 'simulation'
    args.main_dir = MAIN_PATH

    return args


def load_model(actor_path, critic_path, args, device='cpu'):
    """Load trained G2P2C model."""
    from agents.g2p2c.models import ActorCritic
    policy = ActorCritic(args, load=True, actor_path=actor_path,
                         critic_path=critic_path, device=device)
    policy.to(device)
    policy.eval()
    return policy


def run_simulation(policy, args, device='cpu', seed=100):
    """Run a 24-hour simulation and collect data."""
    from utils.core import get_env, custom_reward
    from utils.pumpAction import Pump
    from utils.statespace import StateSpace
    from utils.reward_func import composite_reward
    import time as _time

    patients = (
        [f'adolescent#0{str(i).zfill(2)}' for i in range(1, 11)] +
        [f'child#0{str(i).zfill(2)}' for i in range(1, 11)] +
        [f'adult#0{str(i).zfill(2)}' for i in range(1, 11)]
    )
    patient_name = patients[args.patient_id]
    # Use unique env_id each run to avoid "Cannot re-register" error
    unique_id = int(_time.time() * 1000) % 1000000
    env_id = f'simglucose-sim-{seed}-{unique_id}-v0'

    env = get_env(args, patient_name=patient_name, env_id=env_id,
                  custom_reward=custom_reward, seed=seed)

    state_space = StateSpace(args)
    pump = Pump(args, patient_name=patient_name)
    std_basal = pump.get_basal()

    # Reset environment
    init_state = env.reset()
    cur_state, feat = state_space.update(cgm=init_state.CGM, ins=0, meal=0)
    pump.calibrate(init_state)

    # Calibration period
    for t in range(args.calibration):
        state, reward, is_done, info = env.step(std_basal)
        cur_state, feat = state_space.update(
            cgm=state.CGM, ins=std_basal,
            meal=info['remaining_time'], hour=t,
            meal_type=info['meal_type']
        )

    # Run 24-hour simulation (288 steps at 5-min intervals)
    sim_steps = 288
    glucose_data = []
    insulin_data = []
    meal_data = []
    time_data = []

    for step in range(sim_steps):
        with torch.no_grad():
            policy_step = policy.get_action(cur_state, feat)

        selected_action = policy_step['action'][0]
        rl_action, pump_action = pump.action(
            agent_action=selected_action,
            prev_state=init_state, prev_info=None
        )

        state, _reward, is_done, info = env.step(pump_action)

        # Time of day: simulation starts at 23:00 (11 PM) + calibration steps
        total_minutes_from_23 = (args.calibration + step) * 5
        # Convert to clock time (23:00 + offset, wrapping at 24h)
        clock_minutes = (23 * 60 + total_minutes_from_23) % (24 * 60)
        hours = clock_minutes // 60
        mins = clock_minutes % 60
        time_str = f"{hours:02d}:{mins:02d}"

        # Collect data
        glucose_data.append(state.CGM)
        insulin_data.append(pump_action)
        meal_carbs = info['meal'] * info['sample_time']
        meal_data.append(meal_carbs)
        time_data.append(time_str)

        # Update state
        cur_state, feat = state_space.update(
            cgm=state.CGM, ins=pump_action,
            meal=info['remaining_time'], hour=(step + 1),
            meal_type=info['meal_type'], carbs=info.get('future_carb', 0)
        )

    return {
        'glucose': glucose_data,
        'insulin': insulin_data,
        'meals': meal_data,
        'time': time_data,
    }


def compute_clinical_metrics(glucose_data):
    """Compute clinical metrics from glucose trace."""
    bg = np.array(glucose_data)
    total = len(bg)

    if total == 0:
        return {}

    # Time in ranges
    tir = np.sum((bg >= 70) & (bg <= 180)) / total * 100
    tar_l1 = np.sum((bg > 180) & (bg <= 250)) / total * 100
    tar_l2 = np.sum(bg > 250) / total * 100
    tbr_l1 = np.sum((bg >= 54) & (bg < 70)) / total * 100
    tbr_l2 = np.sum(bg < 54) / total * 100

    # Risk indices
    LBGI, HBGI, RI = risk_index(glucose_data, len(glucose_data))

    return {
        'TAR Level 2': f'{tar_l2:.2f}%',
        'TAR Level 1': f'{tar_l1:.2f}%',
        'Time In Range': f'{tir:.2f}%',
        'TBR Level 1': f'{tbr_l1:.2f}%',
        'TBR Level 2': f'{tbr_l2:.2f}%',
        'Risk Index': f'{RI:.2f}',
        'LBGI': f'{LBGI:.2f}',
        'HBGI': f'{HBGI:.2f}',
    }


def compute_meal_recovery_times(sim_data, meal_config):
    """
    For each meal, find:
    - The time glucose first exceeded 180 mg/dL after the meal
    - The time glucose returned to the 70-180 mg/dL normal range
    - The recovery duration in minutes
    Returns a list of dicts with meal name, peak glucose, and recovery info.
    """
    glucose = np.array(sim_data['glucose'])
    time_labels = sim_data['time']
    meals_carbs = np.array(sim_data['meals'])
    n = len(glucose)

    recovery_results = []

    for meal_name, config in meal_config.items():
        meal_time_minutes = config['time_minutes']

        # The simulation starts at 23:00 with calibration steps.
        # time_labels[i] is the clock time for step i.
        # Find the step index closest to the meal time.
        meal_step = None
        for i, t_str in enumerate(time_labels):
            parts = t_str.split(':')
            t_min = int(parts[0]) * 60 + int(parts[1])
            if t_min >= meal_time_minutes:
                meal_step = i
                break

        if meal_step is None:
            recovery_results.append({
                'meal': meal_name,
                'carbs': config['carbs'],
                'meal_time': f"{meal_time_minutes // 60:02d}:{meal_time_minutes % 60:02d}",
                'peak_glucose': 'N/A',
                'recovery_time': 'Never left range',
                'recovery_minutes': 0,
            })
            continue

        # Search window: from meal time to meal + 5 hours (60 steps at 5 min each)
        search_end = min(meal_step + 60, n)
        window = glucose[meal_step:search_end]

        peak_glucose = float(np.max(window))
        peak_step = meal_step + int(np.argmax(window))

        # Check if glucose ever went above 180 after this meal
        if peak_glucose <= 180:
            recovery_results.append({
                'meal': meal_name,
                'carbs': config['carbs'],
                'meal_time': f"{meal_time_minutes // 60:02d}:{meal_time_minutes % 60:02d}",
                'peak_glucose': f'{peak_glucose:.0f} mg/dL',
                'recovery_time': 'Stayed in range ✅',
                'recovery_minutes': 0,
            })
            continue

        # Find when glucose first exceeds 180 after the meal
        exit_step = None
        for i in range(meal_step, search_end):
            if glucose[i] > 180:
                exit_step = i
                break

        # Find when glucose returns to <= 180 after the peak
        recovery_step = None
        for i in range(peak_step, min(peak_step + 72, n)):  # search up to 6 hours after peak
            if glucose[i] <= 180:
                recovery_step = i
                break

        if recovery_step is not None and exit_step is not None:
            recovery_mins = (recovery_step - exit_step) * 5
            recovery_time_str = time_labels[recovery_step] if recovery_step < len(time_labels) else 'N/A'
            recovery_results.append({
                'meal': meal_name,
                'carbs': config['carbs'],
                'meal_time': f"{meal_time_minutes // 60:02d}:{meal_time_minutes % 60:02d}",
                'peak_glucose': f'{peak_glucose:.0f} mg/dL',
                'recovery_time': f'{recovery_mins} min (at {recovery_time_str})',
                'recovery_minutes': recovery_mins,
            })
        else:
            recovery_results.append({
                'meal': meal_name,
                'carbs': config['carbs'],
                'meal_time': f"{meal_time_minutes // 60:02d}:{meal_time_minutes % 60:02d}",
                'peak_glucose': f'{peak_glucose:.0f} mg/dL',
                'recovery_time': 'Did not recover ⚠️',
                'recovery_minutes': -1,
            })

    return recovery_results


def create_simulation_chart(sim_data):
    """Create the glucose + insulin chart matching the reference design."""
    glucose = sim_data['glucose']
    insulin = sim_data['insulin']
    meals = sim_data['meals']
    time_labels = sim_data['time']

    # Create time axis (minutes from midnight)
    n = len(glucose)
    time_minutes = [i * 5 for i in range(n)]
    time_datetime = [datetime(2024, 1, 1) + timedelta(minutes=m) for m in time_minutes]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=None
    )

    # --- Glucose Trace (Top) ---
    # Target range band (70-180)
    fig.add_trace(go.Scatter(
        x=time_datetime, y=[180] * n,
        fill=None, mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False, hoverinfo='skip'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=time_datetime, y=[70] * n,
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        fillcolor='rgba(76, 175, 80, 0.08)',
        name='TIR 70-180 mg/dL',
        showlegend=True
    ), row=1, col=1)

    # Boundary lines at 70 and 180
    fig.add_trace(go.Scatter(
        x=time_datetime, y=[70] * n,
        mode='lines',
        line=dict(color='#ff6b6b', width=1, dash='dot'),
        showlegend=False, hoverinfo='skip'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=time_datetime, y=[180] * n,
        mode='lines',
        line=dict(color='#ff6b6b', width=1, dash='dot'),
        showlegend=False, hoverinfo='skip'
    ), row=1, col=1)

    # Glucose line
    fig.add_trace(go.Scatter(
        x=time_datetime, y=glucose,
        mode='lines',
        name='Glucose (mg/dL) / Sensor: Guardian RT',
        line=dict(color='#00d4ff', width=2.5),
        hovertemplate='%{y:.1f} mg/dL<extra></extra>'
    ), row=1, col=1)

    # Meal markers (red dashed vertical lines + labels)
    for i, (carb, t) in enumerate(zip(meals, time_datetime)):
        if carb > 0:
            # Vertical dashed line
            fig.add_vline(
                x=t, row=1, col=1,
                line=dict(color='#ff6b6b', width=1.5, dash='dash'),
            )
            # Carb label annotation
            fig.add_annotation(
                x=t, y=520,
                text=f"<b>{carb:.0f} g</b>",
                showarrow=False,
                font=dict(color='#ff6b6b', size=13),
                bgcolor='rgba(255, 107, 107, 0.15)',
                bordercolor='#ff6b6b',
                borderwidth=1,
                borderpad=4,
                row=1, col=1
            )

    # --- Insulin Bars (Bottom) ---
    fig.add_trace(go.Bar(
        x=time_datetime, y=insulin,
        name='Insulin (U/min) / Pump: Insulet',
        marker_color='rgba(100, 120, 180, 0.7)',
        hovertemplate='%{y:.4f} U/min<extra></extra>'
    ), row=2, col=1)

    # Grid lines at key glucose levels
    for level in [70, 180, 250, 300, 400, 600]:
        fig.add_hline(
            y=level, row=1, col=1,
            line=dict(color='rgba(255,255,255,0.1)', width=0.5, dash='dot')
        )

    # Layout
    fig.update_layout(
        height=550,
        plot_bgcolor='#1a2f2a',
        paper_bgcolor='#1a2f2a',
        font=dict(color='#e0e0e0', size=12),
        margin=dict(l=60, r=20, t=30, b=40),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.15,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=11, color='#e0e0e0')
        ),
        hovermode='x unified'
    )

    # Y-axis for glucose
    fig.update_yaxes(
        title_text='<b>Glucose (mg/dL)</b>',
        title_font=dict(color='#ff6b6b', size=13),
        range=[0, 620],
        tickvals=[70, 180, 250, 300, 400, 600],
        gridcolor='rgba(255,255,255,0.05)',
        row=1, col=1
    )

    # Y-axis for insulin
    fig.update_yaxes(
        title_text='<b>Insulin (U/min)</b>',
        title_font=dict(color='#00d4ff', size=13),
        gridcolor='rgba(255,255,255,0.05)',
        row=2, col=1
    )

    # X-axis
    fig.update_xaxes(
        tickformat='%H:%M',
        dtick=3 * 3600000,  # every 3 hours
        gridcolor='rgba(255,255,255,0.05)',
        row=2, col=1
    )

    return fig


# ============================================================
# MAIN APP
# ============================================================

def main():
    # --- Header ---
    st.markdown("""
    <div class="sim-header">
        <h2>🩺 SIMULATION CONSOLE</h2>
    </div>
    """, unsafe_allow_html=True)

    # --- Check for available patients ---
    available_patients = get_available_patients()

    if not available_patients:
        st.error("""
        **No trained weights found!**

        Please place trained weights in the `trained_weights/` folder:
        ```
        trained_weights/
          g2p2c_adult_001_Actor.pth
          g2p2c_adult_001_Critic.pth
        ```

        Train models using `kaggle_train_g2p2c.py` on Kaggle first.
        """)
        st.stop()

    # --- Control Panel ---
    col1, col2, col3 = st.columns([1, 1, 1.5])

    with col1:
        st.markdown('<p class="input-label">Cohort</p>', unsafe_allow_html=True)
        cohort = st.selectbox('Cohort', ['Adult'], label_visibility='collapsed')

    with col2:
        st.markdown('<p class="input-label">Subject</p>', unsafe_allow_html=True)
        patient_labels = [p['label'] for p in available_patients]
        selected_label = st.selectbox('Subject', patient_labels, label_visibility='collapsed')
        selected_patient = next(p for p in available_patients if p['label'] == selected_label)

    with col3:
        st.markdown('<p class="input-label">Controller</p>', unsafe_allow_html=True)
        controller = st.selectbox('Controller', ['🤖 G2P2C (AI)'], label_visibility='collapsed')

    # --- Meal Protocol with Food Selection ---
    st.markdown('<p class="section-header">Meal protocol</p>', unsafe_allow_html=True)

    # Import food database
    try:
        from food_database import FOOD_DATABASE, get_food_info, calculate_carbs, get_food_list
        food_list = get_food_list()
        has_food_db = True
    except ImportError:
        food_list = []
        has_food_db = False

    meal_config = {}
    meals = [
        ('Breakfast', '08:00', 'Apple'),
        ('Lunch', '13:00', 'Roti/Chapati'),
        ('Dinner', '20:00', 'Plain Cooked Rice'),
    ]

    food_options = ['-- Custom (enter carbs manually) --'] + food_list if has_food_db else ['-- Custom (enter carbs manually) --']

    for meal_name, default_time, initial_food in meals:
        st.markdown('<div class="sim-card">', unsafe_allow_html=True)
        st.markdown(f'<p class="meal-label"><b>{meal_name}</b></p>', unsafe_allow_html=True)

        items_key = f'meal_items_{meal_name}'
        next_id_key = f'meal_next_id_{meal_name}'

        # Initialize item list for this meal if not already present
        if items_key not in st.session_state:
            st.session_state[items_key] = [0]
            st.session_state[next_id_key] = 1
            default_item_food = initial_food if initial_food in food_list else food_options[0]
            st.session_state[f'{meal_name}_0_food'] = default_item_food
            f_init = get_food_info(default_item_food) if has_food_db else None
            if f_init:
                st.session_state[f'{meal_name}_0_unit'] = f_init['default_unit']
                st.session_state[f'{meal_name}_0_qty'] = float(f_init['default_qty'])
                st.session_state[f'{meal_name}_0_rate'] = float(f_init['carbs_per_unit'])
                st.session_state[f'{meal_name}_0_prev_food'] = default_item_food
            else:
                st.session_state[f'{meal_name}_0_unit'] = 'grams'
                st.session_state[f'{meal_name}_0_qty'] = 1.0
                st.session_state[f'{meal_name}_0_rate'] = 0.0
                st.session_state[f'{meal_name}_0_prev_food'] = food_options[0]

        item_ids = list(st.session_state[items_key])
        meal_item_details = []
        item_to_remove = None
        total_meal_carbs = 0.0

        if has_food_db:
            # Header column titles
            h1, h2, h3, h4, h5, h6 = st.columns([2.8, 1.2, 0.9, 1.2, 0.9, 0.5])
            with h1:
                st.markdown('<p class="input-label">Food Item</p>', unsafe_allow_html=True)
            with h2:
                st.markdown('<p class="input-label">Unit</p>', unsafe_allow_html=True)
            with h3:
                st.markdown('<p class="input-label">Qty</p>', unsafe_allow_html=True)
            with h4:
                st.markdown('<p class="input-label">Carbs/Unit (g) ✏️</p>', unsafe_allow_html=True)
            with h5:
                st.markdown('<p class="input-label">Carbs</p>', unsafe_allow_html=True)
            with h6:
                st.write("")

            for item_id in item_ids:
                food_key = f'{meal_name}_{item_id}_food'
                unit_key = f'{meal_name}_{item_id}_unit'
                qty_key = f'{meal_name}_{item_id}_qty'
                rate_key = f'{meal_name}_{item_id}_rate'
                prev_food_key = f'{meal_name}_{item_id}_prev_food'

                # Sync defaults when food selection changes
                current_food = st.session_state.get(food_key, food_options[0])
                prev_food = st.session_state.get(prev_food_key)

                if current_food != prev_food:
                    st.session_state[prev_food_key] = current_food
                    if current_food != '-- Custom (enter carbs manually) --':
                        f_info = get_food_info(current_food)
                        if f_info:
                            st.session_state[unit_key] = f_info['default_unit']
                            st.session_state[qty_key] = float(f_info['default_qty'])
                            st.session_state[rate_key] = float(f_info['carbs_per_unit'])
                    else:
                        st.session_state[unit_key] = 'grams'
                        st.session_state[qty_key] = 1.0
                        st.session_state[rate_key] = 0.0

                c1, c2, c3, c4, c5, c6 = st.columns([2.8, 1.2, 0.9, 1.2, 0.9, 0.5])

                with c1:
                    selected_food = st.selectbox(
                        f'{meal_name} Item {item_id}',
                        food_options,
                        key=food_key,
                        label_visibility='collapsed'
                    )

                is_custom = (selected_food == '-- Custom (enter carbs manually) --')
                food_info = get_food_info(selected_food) if not is_custom else None

                with c2:
                    if food_info:
                        avail_units = list(food_info['available_units'].keys())
                        if st.session_state.get(unit_key) not in avail_units:
                            st.session_state[unit_key] = food_info['default_unit']
                        selected_unit = st.selectbox(
                            f'{meal_name} Unit {item_id}',
                            avail_units,
                            key=unit_key,
                            label_visibility='collapsed'
                        )
                    else:
                        selected_unit = 'grams'
                        st.markdown('<p style="color:#8fbc8f;font-size:12px;margin-top:8px;">grams</p>', unsafe_allow_html=True)

                with c3:
                    default_qty = food_info['default_qty'] if food_info else 1.0
                    max_qty = 1000.0 if default_qty > 50 else 500.0
                    step_val = 5.0 if default_qty >= 50 else 0.5
                    quantity = st.number_input(
                        f'{meal_name} Qty {item_id}',
                        min_value=0.0, max_value=max_qty,
                        step=step_val,
                        key=qty_key,
                        label_visibility='collapsed'
                    )

                with c4:
                    carb_rate = st.number_input(
                        f'{meal_name} Rate {item_id}',
                        min_value=0.0, max_value=500.0,
                        step=1.0,
                        key=rate_key,
                        help=f'Carbs per {food_info["default_unit"] if food_info else "unit"}. Edit if product nutrition differs!',
                        label_visibility='collapsed'
                    )

                # Calculate item carbohydrates
                if food_info:
                    conv = food_info['available_units'].get(selected_unit, 1.0)
                else:
                    conv = 1.0
                item_carbs = round(carb_rate * quantity * conv, 1)
                total_meal_carbs += item_carbs

                with c5:
                    st.markdown(f'<div style="padding-top:8px;font-weight:600;color:#68d391;font-size:13px;">{item_carbs:.0f}g</div>', unsafe_allow_html=True)

                with c6:
                    if len(item_ids) > 1:
                        if st.button('🗑️', key=f'del_{meal_name}_{item_id}', help='Remove this food item'):
                            item_to_remove = item_id

                if not is_custom:
                    meal_item_details.append(f"{quantity:g}x {selected_food} ({item_carbs:.0f}g)")
                elif item_carbs > 0:
                    meal_item_details.append(f"Custom ({item_carbs:.0f}g)")

            if item_to_remove is not None:
                st.session_state[items_key].remove(item_to_remove)
                st.rerun()

            # Add Item Button & Combination Summary
            add_col1, add_col2 = st.columns([1.5, 4])
            with add_col1:
                if st.button(f'➕ Add Food to {meal_name}', key=f'add_btn_{meal_name}'):
                    new_id = st.session_state[next_id_key]
                    st.session_state[items_key].append(new_id)
                    st.session_state[next_id_key] += 1
                    # Pick a different default next food
                    new_default_food = food_list[0] if food_list else food_options[0]
                    st.session_state[f'{meal_name}_{new_id}_food'] = new_default_food
                    f_new = get_food_info(new_default_food) if has_food_db else None
                    if f_new:
                        st.session_state[f'{meal_name}_{new_id}_unit'] = f_new['default_unit']
                        st.session_state[f'{meal_name}_{new_id}_qty'] = float(f_new['default_qty'])
                        st.session_state[f'{meal_name}_{new_id}_rate'] = float(f_new['carbs_per_unit'])
                        st.session_state[f'{meal_name}_{new_id}_prev_food'] = new_default_food
                    st.rerun()

            with add_col2:
                if meal_item_details:
                    summary_str = " + ".join(meal_item_details)
                    st.markdown(f'<p style="color:#80cbc4;font-size:12px;margin-top:6px;margin-bottom:0;">🥗 <b>Combination:</b> {summary_str}</p>', unsafe_allow_html=True)

            auto_carbs = round(total_meal_carbs)

        else:
            auto_carbs = 40 if meal_name == 'Breakfast' else (80 if meal_name == 'Lunch' else 60)

        # Automatically synchronize Total Carbohydrates with the calculated sum of items
        carbs_key = f'{meal_name}_carbs'
        last_calc_key = f'{meal_name}_last_calc_carbs'

        if carbs_key not in st.session_state:
            st.session_state[carbs_key] = int(auto_carbs)
            st.session_state[last_calc_key] = auto_carbs
        elif st.session_state.get(last_calc_key) != auto_carbs:
            st.session_state[carbs_key] = int(auto_carbs)
            st.session_state[last_calc_key] = auto_carbs

        # Total Carbs and Meal Time row
        ct1, ct2 = st.columns([1, 1])
        with ct1:
            carbs = st.number_input(
                f'{meal_name} Total Carbohydrates (g)',
                min_value=0, max_value=500,
                step=5,
                key=carbs_key,
                help='Auto-calculated sum of all food items in this meal. You can edit directly to override.'
            )
            if st.session_state.get(carbs_key) != int(auto_carbs):
                st.caption(f'⚠️ Manually overridden (Calculated sum: {int(auto_carbs)}g)')
        with ct2:
            time_val = st.time_input(
                f'{meal_name} Time',
                value=datetime.strptime(default_time, '%H:%M').time(),
                key=f'{meal_name}_time'
            )

        st.markdown('</div>', unsafe_allow_html=True)

        meal_config[meal_name] = {
            'carbs': carbs,
            'time': time_val,
            'time_minutes': time_val.hour * 60 + time_val.minute
        }

    # --- Run Button ---
    st.markdown('<div class="run-btn-container">', unsafe_allow_html=True)
    run_clicked = st.button('▶ Run', use_container_width=True, type='primary')
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Run Simulation ---
    if run_clicked:
        with st.spinner('🔄 Loading model and running simulation...'):
            try:
                # Build meal arrays for the scenario
                # Format: [breakfast, snack1, lunch, snack2, dinner, snack3]
                meal_amounts = [
                    meal_config['Breakfast']['carbs'], 0,
                    meal_config['Lunch']['carbs'], 0,
                    meal_config['Dinner']['carbs'], 0
                ]

                # Create simulation arguments
                args = create_sim_args(
                    patient_id=selected_patient['id'],
                    meal_amounts=meal_amounts,
                    meal_times_minutes=[
                        meal_config['Breakfast']['time_minutes'],
                        0,
                        meal_config['Lunch']['time_minutes'],
                        0,
                        meal_config['Dinner']['time_minutes'],
                        0
                    ]
                )

                device = 'cpu'

                # Load model
                policy = load_model(
                    selected_patient['actor_path'],
                    selected_patient['critic_path'],
                    args, device
                )

                # Run simulation
                sim_data = run_simulation(policy, args, device, seed=42)

                # Store results in session state
                st.session_state['sim_data'] = sim_data
                st.session_state['sim_done'] = True
                st.session_state['meal_config'] = meal_config

            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # --- Display Results ---
    if st.session_state.get('sim_done', False):
        sim_data = st.session_state['sim_data']
        saved_meal_config = st.session_state.get('meal_config', meal_config)

        st.markdown("---")

        # Glucose & Insulin Chart
        fig = create_simulation_chart(sim_data)
        st.plotly_chart(fig, use_container_width=True)

        # Legend badges
        st.markdown("""
        <div style="text-align: center; margin: 8px 0 16px 0;">
            <span class="legend-badge legend-glucose">● Glucose (mg/dL) / Sensor: Guardian RT</span>
            <span class="legend-badge legend-insulin">● Insulin (U/min) / Pump: Insulet</span>
            <span class="legend-badge legend-carb">● Carbohydrate amount (g)</span>
        </div>
        <div style="text-align: center; margin-bottom: 20px;">
            <span class="legend-badge legend-tir">● TIR 70-180 mg/dL</span>
            <span class="legend-badge legend-tbr">● TBR L1 54-70 / L2 <54 mg/dL</span>
            <span class="legend-badge legend-tar">● TAR L1 180-250 / L2 >250 mg/dL</span>
        </div>
        """, unsafe_allow_html=True)

        # Clinical Metrics
        st.markdown('<p class="section-header">Clinical Metrics</p>', unsafe_allow_html=True)

        metrics = compute_clinical_metrics(sim_data['glucose'])

        # Row 1: TAR L2, TAR L1, TIR, TBR L1
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>TAR Level 2</h4>
                <p class="metric-value">{metrics.get('TAR Level 2', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>TAR Level 1</h4>
                <p class="metric-value">{metrics.get('TAR Level 1', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Time In Range</h4>
                <p class="metric-value" style="color: #4caf50;">{metrics.get('Time In Range', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>TBR Level 1</h4>
                <p class="metric-value">{metrics.get('TBR Level 1', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 2: TBR L2, Risk Index, LBGI, HBGI
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <h4>TBR Level 2</h4>
                <p class="metric-value">{metrics.get('TBR Level 2', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        with c6:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Risk Index</h4>
                <p class="metric-value">{metrics.get('Risk Index', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        with c7:
            st.markdown(f"""
            <div class="metric-card">
                <h4>LBGI</h4>
                <p class="metric-value">{metrics.get('LBGI', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        with c8:
            st.markdown(f"""
            <div class="metric-card">
                <h4>HBGI</h4>
                <p class="metric-value">{metrics.get('HBGI', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)

        # Summary stats
        st.markdown("<br>", unsafe_allow_html=True)
        glucose_arr = np.array(sim_data['glucose'])
        st.markdown(f"""
        <div class="sim-card" style="text-align: center;">
            <span style="margin-right: 30px;">📊 Mean Glucose: <b>{glucose_arr.mean():.1f} mg/dL</b></span>
            <span style="margin-right: 30px;">📈 Max: <b>{glucose_arr.max():.1f} mg/dL</b></span>
            <span style="margin-right: 30px;">📉 Min: <b>{glucose_arr.min():.1f} mg/dL</b></span>
            <span>⏱ Duration: <b>24 hours</b></span>
        </div>
        """, unsafe_allow_html=True)

        # --- Meal Recovery Times ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">⏱ Meal Recovery (Time to Normal Range: 70-180 mg/dL)</p>',
                    unsafe_allow_html=True)

        recovery_data = compute_meal_recovery_times(sim_data, saved_meal_config)
        rc_cols = st.columns(len(recovery_data))
        for idx, rec in enumerate(recovery_data):
            with rc_cols[idx]:
                # Color coding based on recovery
                if rec['recovery_minutes'] == 0:
                    color = '#4caf50'  # green - stayed in range
                    icon = '✅'
                elif rec['recovery_minutes'] < 0:
                    color = '#ff5252'  # red - did not recover
                    icon = '⚠️'
                elif rec['recovery_minutes'] <= 60:
                    color = '#4caf50'  # green - fast recovery
                    icon = '🟢'
                elif rec['recovery_minutes'] <= 120:
                    color = '#ffeb3b'  # yellow - moderate
                    icon = '🟡'
                else:
                    color = '#ff9800'  # orange - slow
                    icon = '🟠'

                st.markdown(f"""
                <div class="metric-card">
                    <h4>{icon} {rec['meal']} Recovery</h4>
                    <p class="metric-value" style="color: {color}; font-size: 20px;">{rec['recovery_time']}</p>
                    <p style="color:#8fbc8f; font-size:11px; margin:4px 0 0 0;">
                        Carbs: {rec['carbs']}g at {rec['meal_time']} | Peak: {rec['peak_glucose']}
                    </p>
                </div>
                """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
