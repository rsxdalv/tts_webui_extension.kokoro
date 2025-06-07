from setuptools import setup, find_packages

setup(
    name="extension_kokoro",
    version="0.2.0",
    packages=find_packages(),
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
