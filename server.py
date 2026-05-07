import gradio as gr
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal as signal
import json
import os
import random
from pedalboard import Pedalboard, load_plugin
from pedalboard.io import AudioFile
import time
from core.SiameseParamEstimator import SiameseParamEstimator
from core.Distortion2DNet import Distortion2DNet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAMPLE_RATE = 44100

VST_PATH = os.getenv("VST_PATH", "./vst/bx_distorange.vst3")

BASELINE_WEIGHTS = r"./models/Distortion2DNet.pth"
ADVANCED_WEIGHTS = r"./models/SiameseParamEstimator.pth"

try:
    plugin = load_plugin(VST_PATH)
    if hasattr(plugin, 'bypass'): plugin.bypass = False
    board = Pedalboard([plugin])
    
    param_map = {}
    for p in plugin.parameters.keys():
        pl = p.lower()
        if 'dist' in pl or 'drive' in pl or 'gain' in pl: param_map['dist'] = p
        if 'tone' in pl or 'filter' in pl: param_map['tone'] = p
        if 'level' in pl or 'vol' in pl or 'out' in pl: param_map['level'] = p
    print(f"VST {plugin.name} успешно загружен.")
except Exception as e:
    print(f"Внимание: Ошибка загрузки VST: {e}")
    board = None
    param_map = {}


print("Инициализация нейронных сетей...")
baseline_model = Distortion2DNet().to(DEVICE)
if os.path.exists(BASELINE_WEIGHTS):
    baseline_model.load_state_dict(torch.load(BASELINE_WEIGHTS, map_location=DEVICE))
    print("Модель Distortion2DNet успешно загружена.")
baseline_model.eval()

advanced_model = SiameseParamEstimator(num_params=3).to(DEVICE)
if os.path.exists(ADVANCED_WEIGHTS):
    advanced_model.load_state_dict(torch.load(ADVANCED_WEIGHTS, map_location=DEVICE))
    print("Модель Advanced (Siamese) успешно загружена.")
advanced_model.eval()

def process_vst(audio_numpy, p_dist, p_tone, p_level):
    if board is None: return audio_numpy
    if 'dist' in param_map: setattr(plugin, param_map['dist'], float(p_dist))
    if 'tone' in param_map: setattr(plugin, param_map['tone'], float(p_tone))
    if 'level' in param_map: setattr(plugin, param_map['level'], float(p_level))
    if hasattr(plugin, 'reset'): plugin.reset()
    
    audio_stereo = np.concatenate([audio_numpy, audio_numpy], axis=0)
    processed = board.process(audio_stereo, SAMPLE_RATE)
    return np.mean(processed, axis=0, keepdims=True)

def numpy_to_gr_audio(audio_np):
    """Конвертация numpy массива обратно в формат для аудиоплеера Gradio"""
    audio_clipped = np.clip(audio_np.squeeze(), -1.0, 1.0)
    return (SAMPLE_RATE, (audio_clipped * 32767).astype(np.int16))

def calc_rms_over_time(audio_arr, frame_size=2048, hop_size=512):
    y = audio_arr.squeeze()
    num_frames = 1 + (len(y) - frame_size) // hop_size
    if num_frames <= 0: return np.array([0]), np.array([-100])
    
    rms_vals = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        rms_vals[i] = np.sqrt(np.mean(y[start:end]**2) + 1e-10)
        
    time_axis = np.arange(num_frames) * hop_size / SAMPLE_RATE
    rms_db = 20 * np.log10(rms_vals)
    return time_axis, rms_db

def calc_smooth_psd(audio_arr):
    n_per_seg = min(8192, audio_arr.shape[1])
    f, psd = signal.welch(audio_arr.squeeze(), fs=SAMPLE_RATE, nperseg=n_per_seg)
    psd_db = 10 * np.log10(psd + 1e-10)
    window_len = min(151, len(psd_db))
    if window_len % 2 == 0: window_len -= 1 
    if window_len > 3:
        psd_db = signal.savgol_filter(psd_db, window_length=window_len, polyorder=3)
    return f, psd_db

def analyze_and_compare(raw_audio_path, target_audio_path, generate_random_target):
    if raw_audio_path is None:
        raise gr.Error("Отсутствует необработанный аудиосигнал (Dry).")

    max_samples = int(30.0 * SAMPLE_RATE)

    with AudioFile(raw_audio_path).resampled_to(SAMPLE_RATE) as f:
        audio_data = f.read(f.frames) 
    
    if audio_data.shape[0] > 1:
        audio_data = np.mean(audio_data, axis=0, keepdims=True)
        
    dry_numpy = audio_data[:, :max_samples] 
    true_params = {}

    if generate_random_target:
        if board is None:
            raise gr.Error("VST плагин недоступен. Проверьте путь или отключите программную генерацию.")
        
        true_dist = random.uniform(0.0, 10.0)
        true_tone = random.uniform(0.0, 10.0)
        true_level = random.uniform(2.0, 10.0)
        
        target_numpy = process_vst(dry_numpy, true_dist, true_tone, true_level)
        true_params = {"Distortion": round(true_dist, 2), "Tone": round(true_tone, 2), "Level": round(true_level, 2)}
    else:
        if target_audio_path is None:
            raise gr.Error("Загрузите целевой сигнал (Wet) или активируйте генерацию через VST.")
            
        with AudioFile(target_audio_path).resampled_to(SAMPLE_RATE) as f:
            target_data = f.read(f.frames)
            
        if target_data.shape[0] > 1:
            target_data = np.mean(target_data, axis=0, keepdims=True)
            
        target_numpy = target_data[:, :max_samples]
        
        total_samples = min(dry_numpy.shape[1], target_numpy.shape[1])
        dry_numpy = dry_numpy[:, :total_samples]
        target_numpy = target_numpy[:, :total_samples]

    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats(device=DEVICE) 

    start_time = time.perf_counter()

    dry_tensor = torch.from_numpy(dry_numpy).float().to(DEVICE)
    target_tensor = torch.from_numpy(target_numpy).float().to(DEVICE)

    with torch.no_grad():
        preds_base = baseline_model(dry_tensor, target_tensor)
        params_base = torch.clamp(preds_base.squeeze(), 0.0, 1.0).cpu().numpy() * 10.0
        base_dist, base_tone, base_level = params_base if params_base.size == 3 else (0,0,0)

    with torch.no_grad():
        preds_adv = advanced_model(dry_tensor.unsqueeze(0), target_tensor.unsqueeze(0))
        params_adv = torch.clamp(preds_adv.squeeze(), 0.0, 1.0).cpu().numpy() * 10.0
        adv_dist, adv_tone, adv_level = params_adv if params_adv.size == 3 else (0,0,0)

    end_time = time.perf_counter()
    inference_time = end_time - start_time

    print(f"Время инференса обеих моделей: {inference_time:.4f} секунд")

    if DEVICE == 'cuda':
        peak_vram_mb = torch.cuda.max_memory_allocated(device=DEVICE) / (1024**2)
        print(f"Пиковое потребление VRAM (без учета самих весов): {peak_vram_mb:.2f} МБ")
    else:
        print("Тест запущен на CPU. Метрика VRAM не применяется.")
    
    base_render = process_vst(dry_numpy, base_dist, base_tone, base_level)
    adv_render = process_vst(dry_numpy, adv_dist, adv_tone, adv_level)

    audio_target_out = numpy_to_gr_audio(target_numpy)
    audio_base_out = numpy_to_gr_audio(base_render)
    audio_adv_out = numpy_to_gr_audio(adv_render)

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'Рисунок 1 — Сравнение предсказанных значений параметров эффекта', 
            'Рисунок 2 — Амплитудно-частотная характеристика (АЧХ) со сглаживанием', 
            'Рисунок 3 — Оценка микро-динамики (Осциллограмма 20 мс)',
            'Рисунок 4 — Динамика RMS (Огибающая громкости на протяжении всего файла)'
        ),
        vertical_spacing=0.08,
        specs=[[{"type": "bar"}], [{"type": "scatter"}], [{"type": "scatter"}], [{"type": "scatter"}]]
    )

    COLOR_TARGET, COLOR_BASE, COLOR_ADV = '#2ca02c', '#d62728', '#1f77b4'
    labels = ['Distortion', 'Tone ', 'Level']

    if generate_random_target:
        fig.add_trace(go.Bar(name='Целевой сигнал', legendgroup='target', x=labels, y=[true_params["Distortion"], true_params["Tone"], true_params["Level"]], marker_color=COLOR_TARGET), row=1, col=1)
    
    fig.add_trace(go.Bar(name='Distortion2DNet', legendgroup='base', x=labels, y=[base_dist, base_tone, base_level], marker_color=COLOR_BASE), row=1, col=1)
    fig.add_trace(go.Bar(name='SiameseParamEstimator', legendgroup='adv', x=labels, y=[adv_dist, adv_tone, adv_level], marker_color=COLOR_ADV), row=1, col=1)

    f_t, psd_t_db = calc_smooth_psd(target_numpy)
    f_b, psd_b_db = calc_smooth_psd(base_render)
    f_a, psd_a_db = calc_smooth_psd(adv_render)

    fig.add_trace(go.Scatter(x=f_t, y=psd_t_db, legendgroup='target', mode='lines', name='Целевой сигнал', line=dict(color=COLOR_TARGET, width=2.5), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=f_b, y=psd_b_db, legendgroup='base', mode='lines', name='Distortion2DNet', line=dict(color=COLOR_BASE, width=2, dash='dot'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=f_a, y=psd_a_db, legendgroup='adv', mode='lines', name='SiameseParamEstimator', line=dict(color=COLOR_ADV, width=2.5, dash='dash'), showlegend=False), row=2, col=1)

    start_s = min(int(2.0 * SAMPLE_RATE), target_numpy.shape[1] - 100)
    end_s = min(start_s + int(0.02 * SAMPLE_RATE), target_numpy.shape[1]) 
    time_axis_osc = np.linspace(0, 20, end_s - start_s)
    
    fig.add_trace(go.Scatter(x=time_axis_osc, y=target_numpy.squeeze()[start_s:end_s], legendgroup='target', mode='lines', name='Целевой сигнал', line=dict(color=COLOR_TARGET, width=2), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=time_axis_osc, y=base_render.squeeze()[start_s:end_s], legendgroup='base', mode='lines', name='Distortion2DNet', line=dict(color=COLOR_BASE, width=1.5, dash='dot'), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=time_axis_osc, y=adv_render.squeeze()[start_s:end_s], legendgroup='adv', mode='lines', name='SiameseParamEstimator', line=dict(color=COLOR_ADV, width=2, dash='dash'), showlegend=False), row=3, col=1)

    t_rms, target_rms_db = calc_rms_over_time(target_numpy)
    _, base_rms_db = calc_rms_over_time(base_render)
    _, adv_rms_db = calc_rms_over_time(adv_render)

    fig.add_trace(go.Scatter(x=t_rms, y=target_rms_db, mode='lines', legendgroup='target', name='Целевой сигнал', line=dict(color=COLOR_TARGET, width=2), showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=t_rms, y=base_rms_db, mode='lines', legendgroup='base', name='Distortion2DNet', line=dict(color=COLOR_BASE, width=1.5, dash='dot'), showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=t_rms, y=adv_rms_db, mode='lines', legendgroup='adv', name='SiameseParamEstimator', line=dict(color=COLOR_ADV, width=2, dash='dash'), showlegend=False), row=4, col=1)

    fig.update_layout(
        title_text="Сравнение моделей: Distortion2DNet и SiameseParamEstimator",
        title_font=dict(size=22), title_x=0.5, 
        height=1400, autosize=True, 
        barmode='group', template='plotly_white', hovermode="x unified",
        margin=dict(t=150, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5, font=dict(size=14)) 
    )

    fig.update_yaxes(range=[0, 11], title_text="Значение (0-10)", row=1, col=1)
    fig.update_yaxes(title_text="Мощность (дБ)", row=2, col=1)
    fig.update_yaxes(title_text="Амплитуда", row=3, col=1)
    fig.update_yaxes(title_text="Уровень RMS (дБ)", row=4, col=1)

    fig.update_xaxes(type="log", range=[np.log10(20), np.log10(20000)], title_text="Частота (Гц) [Лог. шкала]", row=2, col=1)
    fig.update_xaxes(title_text="Время (мс)", row=3, col=1)
    fig.update_xaxes(title_text="Время (с)", row=4, col=1)

    params_out = {}
    if generate_random_target:
        params_out["Целевой сигнал (Случайная генерация)"] = true_params
    params_out["Distortion2DNet"] = {"Distortion": round(float(base_dist),2), "Tone": round(float(base_tone),2), "Level": round(float(base_level),2)}
    params_out["SiameseParamEstimator"] = {"Distortion": round(float(adv_dist),2), "Tone": round(float(adv_tone),2), "Level": round(float(adv_level),2)}

    return fig, json.dumps(params_out, indent=2, ensure_ascii=False), audio_target_out, audio_base_out, audio_adv_out

with gr.Blocks() as demo:
    gr.Markdown("# Система идентификации параметров гитарного эффекта")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Загрузка аудиоданных")
            raw_input = gr.Audio(label="Необработанный сигнал (Dry Input)", type="filepath")
            
            generate_vst_checkbox = gr.Checkbox(
                label="Программная генерация целевого сигнала (VST bx_distorange)", 
                value=True, 
                info="При активации загруженный ниже целевой сигнал будет проигнорирован."
            )
            
            target_input = gr.Audio(label="Целевой сигнал (Wet Target) — опционально", type="filepath")
            
            analyze_btn = gr.Button("Произвести анализ", variant="primary")
            
            gr.Markdown("### 2. Численные результаты оценки")
            params_output = gr.Code(language="json", label="Значения параметров")
            
        with gr.Column(scale=2):
            gr.Markdown("### Аналитика и графики")
            plot_output = gr.Plot(label="Анализ характеристик")

    gr.Markdown("### 3. Аудиоконтроль синтезированных сигналов")
    audio_target = gr.Audio(label="Целевой сигнал", interactive=False)
    audio_base = gr.Audio(label="Синтез Distortion2DNet", interactive=False)
    audio_adv = gr.Audio(label="Синтез SiameseParamEstimator", interactive=False)

    analyze_btn.click(
        fn=analyze_and_compare,
        inputs=[raw_input, target_input, generate_vst_checkbox],
        outputs=[plot_output, params_output, audio_target, audio_base, audio_adv]
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.Theme.from_hub("Taithrah/Minimal"))