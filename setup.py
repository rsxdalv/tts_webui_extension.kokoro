import setuptools

setuptools.setup(
    name="tts_webui_extension.kokoro",
    version="0.3.1",
    packages=setuptools.find_namespace_packages(),
    install_requires=[
        "kokoro",
        "torch",
        "gradio",
        "misaki[en]",
        "misaki[ja]",
        "misaki[ko]",
        "misaki[zh]",
    ],
    package_data={
        "extension_kokoro": ["samples/*.txt", "samples/*.md"],
    },
    include_package_data=True,
)

