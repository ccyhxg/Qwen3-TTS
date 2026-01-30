# Qwen3-TTS Gradio WebUI Demo

This folder contains a Gradio WebUI demo for Qwen3-TTS models with 3 tabs:

1. **VoiceDesign** - Generate speech with voice design instructions (自然语言描述音色)
2. **Base (Voice Clone)** - Clone voice from reference audio and generate speech (声音克隆)
3. **TTS** - Standard text-to-speech using CustomVoice model (标准TTS)

## Quick Start

1. Install dependencies (if not already installed):
   ```bash
   pip install gradio soundfile librosa torch transformers accelerate
   ```

2. Run the demo:
   ```bash
   cd /path/to/qwen3_tts_gradio_demo
   python app.py
   ```

3. Open your browser at `http://localhost:8000`

## Usage with Different Models

The default model is `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`. You can specify a different model:

```bash
# VoiceDesign model
python app.py --checkpoint Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign

# Base (Voice Clone) model
python app.py --checkpoint Qwen/Qwen3-TTS-12Hz-1.7B-Base

# CustomVoice model (default)
python app.py --checkpoint Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

## Arguments

```
--checkpoint, -c        Model checkpoint path or HuggingFace repo id
--device                Device for device_map (default: cuda:0)
--dtype                 Torch dtype (default: bfloat16)
--ip                    Server bind IP (default: 0.0.0.0)
--port                  Server port (default: 8000)
--share                 Create a public Gradio link
--concurrency           Gradio queue concurrency (default: 16)
```

## Model Types

### VoiceDesign
- Model: `Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- Use natural language to describe the desired voice/style
- Supports: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

### Base (Voice Clone)
- Model: `Qwen3-TTS-12Hz-1.7B-Base`
- Clone voice from reference audio (3 seconds)
- Supports voice reuse via save/load feature

### TTS (CustomVoice)
- Model: `Qwen3-TTS-12Hz-1.7B-CustomVoice`
- Standard TTS with predefined speakers (Vivian, Serena, etc.)
- Optional instruction control
