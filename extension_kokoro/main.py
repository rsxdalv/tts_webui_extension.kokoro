import functools
from typing import TYPE_CHECKING
import gradio as gr
import torch
import numpy as np

from .CHOICES import CHOICES
from tts_webui.decorators import *
from tts_webui.extensions_loader.decorator_extensions import (
    decorator_extension_inner,
    decorator_extension_outer,
)
from tts_webui.utils.list_dir_models import unload_model_button
from tts_webui.utils.manage_model_state import manage_model_state
from tts_webui.utils.randomize_seed import randomize_seed_ui

if TYPE_CHECKING:
    from kokoro import KModel, KPipeline


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
def get_model(model_name="hexgrad/Kokoro-82M", use_gpu=False) -> "KModel":
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
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr-fr": "f",
    "hi": "h",
    "it": "i",
    "pt-br": "p",
    "ja": "j",
    "zh": "z",
}

LANG_CODES = dict(
    # pip install misaki[en]
    a="American English",
    b="British English",
    # espeak-ng
    e="es",
    f="fr-fr",
    h="hi",
    i="it",
    p="pt-br",
    # pip install misaki[ja]
    j="Japanese",
    # pip install misaki[zh]
    z="Mandarin Chinese",
)


def get_pipeline(lang_code) -> "KPipeline":
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


def get_voice(voice_name) -> torch.FloatTensor:
    """Lazily load a voice only when needed"""
    if voice_name not in _loaded_voices:
        pipeline = get_pipeline(voice_name[0])
        _loaded_voices[voice_name] = pipeline.load_voice(voice_name)
        # load_single_voice
    return _loaded_voices[voice_name]


def get_voice_from_formula(formula: str, normalize=True) -> torch.FloatTensor:
    """
    Parse a voice formula string and return the combined voice tensor.

    Args:
        formula: String like "af_heart * 0.333 + af_bella * 0.333 + af_nicole * 0.333"

    Raises:
        ValueError: If voice name is invalid or formula is malformed
    """
    if not formula.strip():
        raise ValueError("Formula cannot be empty")

    def get_voices():
        try:
            for term in formula.split("+"):
                voice_name, weight_str = term.strip().split("*", 1)
                voice_name = voice_name.strip()
                weight = float(weight_str.strip())

                if not voice_name or voice_name[0] not in LANG_CODES:
                    raise ValueError(f"Unknown voice: '{voice_name}'")

                yield get_voice(voice_name), weight

        except ValueError as e:
            raise ValueError(
                f"Invalid term format: '{formula}'. Expected 'voice_name * weight'"
            ) from e

    # Collect voices and weights
    voices_and_weights = list(get_voices())
    voices = [v for v, w in voices_and_weights]
    weights = torch.tensor([w for v, w in voices_and_weights], dtype=torch.float32)

    # Normalize weights so they sum to 1
    if normalize:
        weights = weights / weights.sum()
    voice_stack = torch.stack(voices)

    # Add dimensions to weights to match voice tensor dimensions
    for _ in range(voice_stack.dim() - 1):
        weights = weights.unsqueeze(-1)

    # Weighted average
    return torch.sum(voice_stack * weights, dim=0)


def tts(
    text,
    voice="af_heart",
    speed=1,
    use_gpu=True,
    model_name="hexgrad/Kokoro-82M",
    progress=gr.Progress(),
    **kwargs,
):
    CUDA_AVAILABLE = torch.cuda.is_available()
    use_gpu = use_gpu and CUDA_AVAILABLE

    progress(0, desc="Loading voices...")
    pipeline = get_pipeline(voice[0])
    # if voice is a formula
    if "+" in voice or "*" in voice:
        pack = get_voice_from_formula(voice)
    else:
        pack = get_voice(voice)

    progress(0.25, desc="Loading model...")
    model = get_model(model_name=model_name, use_gpu=use_gpu)

    def gen_chunks():
        for graphemes, phonemes, _ in pipeline(text, voice, speed):
            progress(0.5, desc=f"Generating audio: {graphemes[:30]}..")
            ref_s = pack[len(phonemes) - 1]
            print(f"Generating {graphemes}")
            audio = model(phonemes, ref_s, speed)

            yield {
                "audio_out": (24000, audio.cpu().numpy()),
                "tokens": phonemes,
            }

    results = list(gen_chunks())

    def combine_audio_out(results):
        return (
            results[0]["audio_out"][0],
            np.concatenate([r["audio_out"][1] for r in results]),
        )

    return {
        "audio_out": combine_audio_out(results),
        "tokens": " ".join([r["tokens"] for r in results]),
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
                
    For certain tasks, might require espeak-ng: `sudo apt-get install espeak-ng` or `brew install espeak-ng` or `pacman -S espeak-ng`, more instructions:
                
    [Installation instructions](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md#installation)

    """
    )

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                lines=3,
                label="Text to generate",
                info="Arbitrarily many characters supported",
            )
            generate_btn = gr.Button("Generate", variant="primary")

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

            with gr.Row():
                seed, randomize_seed_callback = randomize_seed_ui()

            with gr.Row():
                unload_model_button("kokoro")

        with gr.Column():
            audio_out = gr.Audio(label="Generated Audio", autoplay=True)

            with gr.Accordion("Output Tokens", open=True):
                tokens_out = gr.Textbox(
                    interactive=False,
                    show_label=False,
                    info="Tokens used to generate the audio, up to 510 context length.",
                )
                tokenize_btn = gr.Button("Tokenize", variant="secondary")
                gr.Markdown(TOKEN_NOTE)

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
