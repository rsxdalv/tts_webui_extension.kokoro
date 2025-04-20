from setuptools import setup, find_packages

setup(
    name="extension_kokoro",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "kokoro",
        "torch",
        "gradio>=3.50.2",
    ],
    package_data={
        "extension_kokoro": ["samples/*.txt", "samples/*.md"],
    },
    include_package_data=True,
)
