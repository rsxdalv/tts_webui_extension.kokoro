import functools
import gradio as gr
import torch
import random
import os

from .CHOICES import CHOICES
from tts_webui.decorators import *
from tts_webui.extensions_loader.decorator_extensions import (
    decorator_extension_inner,
    decorator_extension_outer,
)
from tts_webui.utils.list_dir_models import unload_model_button
from tts_webui.utils.manage_model_state import manage_model_state
from tts_webui.utils.randomize_seed import randomize_seed_ui


def extension__tts_generation_webui():
    ui()
    return {
        "package_name": "extension_kokoro",
        "name": "Kokoro",
        "requirements": "git+https://github.com/rsxdalv/extension_kokoro@main",
        "description": "Kokoro: A small, fast, and high-quality TTS model",
        "extension_type": "interface",
        "extension_class": "text-to-speech",
        "author": "hexgrad",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://huggingface.co/hexgrad/Kokoro-82M",
        "extension_website": "https://github.com/rsxdalv/extension_kokoro",
        "extension_platform_version": "0.0.1",
    }


# Dictionary to cache loaded models
_models = {}


@manage_model_state("kokoro")
def get_model(model_name="hexgrad/Kokoro-82M", use_gpu=False):
    """Lazily load the model only when needed"""
    from kokoro import KModel

    gpu_key = bool(use_gpu and torch.cuda.is_available())
    if gpu_key not in _models:
        _models[gpu_key] = (
            KModel(
                repo_id=model_name,
            )
            .to("cuda" if gpu_key else "cpu")
            .eval()
        )
    return _models[gpu_key]


# Dictionary to cache loaded pipelines
_pipelines = {}

# Custom lexicon entries for each language code
_lexicon_entries = {"a": {"kokoro": "kˈOkəɹO"}, "b": {"kokoro": "kˈQkəɹQ"}}


ALIASES = {
    'en-us': 'a',
    'en-gb': 'b',
    'es': 'e',
    'fr-fr': 'f',
    'hi': 'h',
    'it': 'i',
    'pt-br': 'p',
    'ja': 'j',
    'zh': 'z',
}

LANG_CODES = dict(
    # pip install misaki[en]
    a='American English',
    b='British English',

    # espeak-ng
    e='es',
    f='fr-fr',
    h='hi',
    i='it',
    p='pt-br',

    # pip install misaki[ja]
    j='Japanese',

    # pip install misaki[zh]
    z='Mandarin Chinese',
)

def get_pipeline(lang_code):
    """Lazily create a pipeline only when needed"""
    from kokoro import KPipeline

    if lang_code not in _pipelines:
        _pipelines[lang_code] = KPipeline(lang_code=lang_code, model=False)
        # Apply custom lexicon entries if any
        if lang_code in _lexicon_entries:
            for word, pronunciation in _lexicon_entries[lang_code].items():
                _pipelines[lang_code].g2p.lexicon.golds[word] = pronunciation
    return _pipelines[lang_code]


# Dictionary to cache loaded voices
_loaded_voices = {}


def get_voice(voice_name):
    """Lazily load a voice only when needed"""
    if voice_name not in _loaded_voices:
        pipeline = get_pipeline(voice_name[0])
        _loaded_voices[voice_name] = pipeline.load_voice(voice_name)
    return _loaded_voices[voice_name]


def tts(
    text,
    voice="af_heart",
    speed=1,
    use_gpu=True,
    model_name="hexgrad/Kokoro-82M",
    **kwargs,
):
    CUDA_AVAILABLE = torch.cuda.is_available()
    use_gpu = use_gpu and CUDA_AVAILABLE

    pipeline = get_pipeline(voice[0])
    pack = get_voice(voice)

    model = get_model(model_name=model_name, use_gpu=use_gpu)
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps) - 1]
        try:
            audio = model(ps, ref_s, speed)
        except Exception as e:
            if use_gpu:
                print(f"Warning: {str(e)}")
                print("Retrying with CPU. To avoid this error, change Hardware to CPU.")
                audio = model(ps, ref_s, speed)
            else:
                raise e

        return {
            "audio_out": (24000, audio.cpu().numpy()),
            "tokens": ps,
        }

    return {
        "audio_out": None,
        "tokens": "",
    }


@functools.wraps(tts)
@decorator_extension_outer
@decorator_apply_torch_seed
@decorator_save_metadata
@decorator_save_wav
@decorator_add_model_type("kokoro")
@decorator_add_base_filename
@decorator_add_date
@decorator_log_generation
@decorator_extension_inner
@log_function_time
def tts_decorated(*args, **kwargs):
    return tts(*args, **kwargs)


def tokenize_first(text, voice="af_heart"):
    """Get tokens for the text using the specified voice"""
    pipeline = get_pipeline(voice[0])
    for _, ps, _ in pipeline(text, voice):
        return ps
    return ""


def get_random_quote():
    """Get a random quote from the en.txt file"""
    with open(os.path.join(os.path.dirname(__file__), "samples", "en.txt"), "r") as r:
        random_quotes = [line.strip() for line in r]
    return random.choice(random_quotes)


def get_gatsby():
    """Get text from the Gatsby file"""
    with open(
        os.path.join(os.path.dirname(__file__), "samples", "gatsby5k.md"), "r"
    ) as r:
        return r.read().strip()


def get_frankenstein():
    """Get text from the Frankenstein file"""
    with open(
        os.path.join(os.path.dirname(__file__), "samples", "frankenstein5k.md"), "r"
    ) as r:
        return r.read().strip()


# Voice choices
TOKEN_NOTE = """
💡 Customize pronunciation with Markdown link syntax and /slashes/ like `[Kokoro](/kˈOkəɹO/)`

💬 To adjust intonation, try punctuation `;:,.!?—…"()""` or stress `ˈ` and `ˌ`

⬇️ Lower stress `[1 level](-1)` or `[2 levels](-2)`

⬆️ Raise stress 1 level `[or](+2)` 2 levels (only works on less stressed, usually short words)
"""


def ui():
    """Create the Gradio UI for the Kokoro extension"""
    CUDA_AVAILABLE = torch.cuda.is_available()

    gr.Markdown(
        """
    # Kokoro TTS
    ### 🎙️ Text-to-speech with Kokoro
                
    For certain tasks, might require espeak-ng: `sudo apt-get install espeak-ng` or `brew install espeak-ng` or `pacman -S espeak-ng`, more instructions:
                
    [Installation instructions](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md#installation)

    ### for Japanese generation

    ```python
    pip install 'fugashi[unidic]'
    python -m unidic download
    ```
    """
    )

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                lines=3,
                label="Text to generate",
                info="Arbitrarily many characters supported",
            )

            with gr.Row():
                voice = gr.Dropdown(
                    list(CHOICES.items()),
                    value="af_heart",
                    label="Voice",
                    info="Quality and availability vary by language",
                    allow_custom_value=True,
                )
                use_gpu = gr.Dropdown(
                    [("ZeroGPU 🚀", True), ("CPU 🐌", False)],
                    value=CUDA_AVAILABLE,
                    label="Hardware",
                    info="GPU is usually faster, but has a usage quota",
                    interactive=CUDA_AVAILABLE,
                )

            speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label="Speed")

            model_name = gr.Dropdown(
                [
                    ("Kokoro-82M", "hexgrad/Kokoro-82M"),
                    ("Kokoro-82M-v1.1-zh", "hexgrad/Kokoro-82M-v1.1-zh"),
                ],
                value="hexgrad/Kokoro-82M",
                label="Model",
                info="Select the Kokoro model to use",
            )

            generate_btn = gr.Button("Generate", variant="primary")

            with gr.Row():
                random_btn = gr.Button("🎲 Random Quote 💬", variant="secondary")
                gatsby_btn = gr.Button("🥂 Gatsby 📕", variant="secondary")
                frankenstein_btn = gr.Button("💀 Frankenstein 📗", variant="secondary")

            with gr.Row():
                unload_model_button("kokoro")
                seed, randomize_seed_callback = randomize_seed_ui()

        with gr.Column():
            # Output section
            audio_out = gr.Audio(label="Generated Audio", autoplay=True)

            with gr.Accordion("Output Tokens", open=True):
                tokens_out = gr.Textbox(
                    interactive=False,
                    show_label=False,
                    info="Tokens used to generate the audio, up to 510 context length.",
                )
                tokenize_btn = gr.Button("Tokenize", variant="secondary")
                gr.Markdown(TOKEN_NOTE)

    # Event handlers
    random_btn.click(fn=get_random_quote, inputs=[], outputs=[text])
    gatsby_btn.click(fn=get_gatsby, inputs=[], outputs=[text])
    frankenstein_btn.click(fn=get_frankenstein, inputs=[], outputs=[text])

    tokenize_btn.click(fn=tokenize_first, inputs=[text, voice], outputs=[tokens_out])

    generate_btn.click(
        **randomize_seed_callback,
    ).then(
        **dictionarize(
            fn=tts_decorated,
            inputs={
                text: "text",
                voice: "voice",
                speed: "speed",
                use_gpu: "use_gpu",
                model_name: "model_name",
                seed: "seed",
            },
            outputs={
                "audio_out": audio_out,
                "tokens": tokens_out,
                "metadata": gr.JSON(visible=False),
                "folder_root": gr.Textbox(visible=False),
            },
        ),
        api_name="kokoro",
    )


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        ui()
    demo.launch()
