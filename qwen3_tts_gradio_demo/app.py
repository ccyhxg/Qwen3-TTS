"""
Qwen3-TTS Gradio WebUI (纯 pip 包驱动，零本地依赖)
支持三个 Tab：
1. Voice Design - 自定义音色设计
2. Voice Clone - 基于语音克隆的 TTS
3. TTS - 标准文本转语音
"""

import os
import sys
import tempfile
import gradio as gr

# 尝试从 pip 包导入，如果失败则从本地导入
try:
    from qwen_tts import TTS
except ImportError as e:
    sys.stderr.write(f"Warning: Failed to import from qwen_tts pip package: {e}\n")
    # 如果 pip 包不可用，则尝试从本地 qwen_tts 子目录导入
    local_qwen_tts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_tts")
    if os.path.isdir(local_qwen_tts_path):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from qwen_tts import TTS
    else:
        raise RuntimeError(f"无法导入 qwen_tts 模块。请确保已安装 qwen-tts pip 包: pip install qwen-tts")

# 初始化 TTS 模型（仅需一次）
tts_model = None

def init_model():
    global tts_model
    if tts_model is None:
        try:
            tts_model = TTS()
        except Exception as e:
            raise RuntimeError(f"模型初始化失败: {e}")
    return tts_model

# ============= 1. Voice Design Tab =============
def voice_design(tts_text, voice_name, speed=1.0, temperature=0.75):
    """
    语音设计：选择预定义音色进行 TTS
    """
    try:
        model = init_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            output_path = fp.name
        # 用 TTS 生成（voice_name 作为 style 或 speaker）
        model.tts(
            text=tts_text,
            voice=voice_name,
            speed=speed,
            temperature=temperature,
            output=output_path,
        )
        return output_path
    except Exception as e:
        raise gr.Error(f"Voice Design 失败: {e}")

# ============= 2. Voice Clone (Base) Tab =============
def voice_clone_base(tts_text, reference_audio_path, speed=1.0, temperature=0.75):
    """
    语音克隆（Base）：基于参考音频克隆音色
    """
    if not reference_audio_path:
        raise gr.Error("请上传参考音频文件（.wav/.mp3）")
    try:
        model = init_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            output_path = fp.name
        # 使用参考音频进行克隆
        model.clone_voice(
            text=tts_text,
            reference_audio=reference_audio_path,
            reference_text="",  # 可选：参考音频的文本
            speed=speed,
            temperature=temperature,
            output=output_path,
        )
        return output_path
    except Exception as e:
        raise gr.Error(f"Voice Clone 失败: {e}")

# ============= 3. Standard TTS Tab =============
def standard_tts(tts_text, speed=1.0, temperature=0.75):
    """
    标准 TTS：默认音色 + 文本
    """
    try:
        model = init_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            output_path = fp.name
        model.tts(
            text=tts_text,
            speed=speed,
            temperature=temperature,
            output=output_path,
        )
        return output_path
    except Exception as e:
        raise gr.Error(f"标准 TTS 失败: {e}")

# UI 构建
with gr.Blocks(title="Qwen3-TTS Gradio UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Qwen3-TTS Gradio WebUI")
    gr.Markdown("基于 pip 包 `qwen-tts` 的 WebUI。请确保已安装：`pip install qwen-tts`")

    with gr.Tab("🎨 Voice Design（音色设计）"):
        gr.Markdown("选择预定义音色进行合成。支持调节速度与随机性。")
        with gr.Row():
            with gr.Column():
                vd_text = gr.Textbox(
                    label="输入文本", 
                    placeholder="请输入要合成的文本...", 
                    value="欢迎使用 Qwen3-TTS，这是语音设计模式。"
                )
                vd_voice = gr.Dropdown(
                    label="选择音色",
                    choices=["default", "chinese", "english", "emotion", "news"],
                    value="default"
                )
                vd_speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="语速 Speed")
                vd_temp = gr.Slider(minimum=0.1, maximum=1.0, value=0.75, step=0.05, label="随机性 Temperature")
                vd_btn = gr.Button("🚀 合成", variant="primary")
            with gr.Column():
                vd_output = gr.Audio(label="合成音频")
        vd_btn.click(
            fn=voice_design,
            inputs=[vd_text, vd_voice, vd_speed, vd_temp],
            outputs=vd_output,
        )

    with gr.Tab("🎙️ Voice Clone（语音克隆）"):
        gr.Markdown("上传参考音频，克隆其音色进行合成。")
        with gr.Row():
            with gr.Column():
                vc_text = gr.Textbox(
                    label="输入文本", 
                    placeholder="请输入要合成的文本...", 
                    value="这是基于参考音频克隆的语音效果。"
                )
                vc_ref = gr.Audio(
                    label="参考音频（.wav/.mp3）",
                    type="filepath"
                )
                vc_speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="语速 Speed")
                vc_temp = gr.Slider(minimum=0.1, maximum=1.0, value=0.75, step=0.05, label="随机性 Temperature")
                vc_btn = gr.Button("✨ 克隆合成", variant="primary")
            with gr.Column():
                vc_output = gr.Audio(label="克隆音频")
        vc_btn.click(
            fn=voice_clone_base,
            inputs=[vc_text, vc_ref, vc_speed, vc_temp],
            outputs=vc_output,
        )

    with gr.Tab("📝 Standard TTS（标准 TTS）"):
        gr.Markdown("默认音色 + 文本，快速合成。")
        with gr.Row():
            with gr.Column():
                st_text = gr.Textbox(
                    label="输入文本", 
                    placeholder="请输入要合成的文本...", 
                    value="这是标准 TTS 模式，使用默认音色。"
                )
                st_speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="语速 Speed")
                st_temp = gr.Slider(minimum=0.1, maximum=1.0, value=0.75, step=0.05, label="随机性 Temperature")
                st_btn = gr.Button("🗣️ 合成", variant="primary")
            with gr.Column():
                st_output = gr.Audio(label="合成音频")
        st_btn.click(
            fn=standard_tts,
            inputs=[st_text, st_speed, st_temp],
            outputs=st_output,
        )

# 启动
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
