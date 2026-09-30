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
    # meal_amounts = [breakfast_carbs, 0, lunch_carbs, 0, dinner_carbs, 0]
    args.meal_amount = meal_amounts
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

    patients = (
        [f'adolescent#0{str(i).zfill(2)}' for i in range(1, 11)] +
        [f'child#0{str(i).zfill(2)}' for i in range(1, 11)] +
        [f'adult#0{str(i).zfill(2)}' for i in range(1, 11)]
    )
    patient_name = patients[args.patient_id]
    env_id = f'simglucose-sim-{seed}-v0'

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

        # Time of day (starts at midnight)
        minutes = step * 5
        hours = minutes // 60
        mins = minutes % 60
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

        # Safety check
        if state.CGM <= 40 or state.CGM >= 600:
            # Fill remaining with last value
            for _ in range(step + 1, sim_steps):
                glucose_data.append(state.CGM)
                insulin_data.append(0)
                meal_data.append(0)
                minutes = len(glucose_data) * 5
                time_data.append(f"{minutes // 60:02d}:{minutes % 60:02d}")
            break

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

    # --- Meal Protocol ---
    st.markdown('<p class="section-header">Meal protocol</p>', unsafe_allow_html=True)

    meal_config = {}
    meals = [
        ('Breakfast', 40, '08:00'),
        ('Lunch', 80, '13:00'),
        ('Dinner', 60, '20:00'),
    ]

    for meal_name, default_carbs, default_time in meals:
        st.markdown(f'<div class="sim-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 2])
        with c1:
            st.markdown(f'<p class="meal-label"><b>{meal_name}</b></p>', unsafe_allow_html=True)
        with c2:
            carbs = st.number_input(
                f'{meal_name} Carbohydrates (g)',
                min_value=0, max_value=200, value=default_carbs,
                step=5, key=f'{meal_name}_carbs'
            )
        with c3:
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

            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # --- Display Results ---
    if st.session_state.get('sim_done', False):
        sim_data = st.session_state['sim_data']

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


if __name__ == '__main__':
    main()
