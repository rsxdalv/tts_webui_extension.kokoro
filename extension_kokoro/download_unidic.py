import gradio as gr


def download_unidic():
    """Download unidic dictionary for Japanese language support"""
    try:
        import subprocess
        import sys
        import importlib.util

        # Check if unidic is installed
        if importlib.util.find_spec("unidic") is None:
            # Try to install unidic first
            print("UniDic not found. Attempting to install...")
            install_process = subprocess.Popen(
                [sys.executable, "-m", "pip", "install", "unidic"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            install_stdout, install_stderr = install_process.communicate()

            if install_process.returncode != 0:
                error_msg = f"Failed to install UniDic: {install_stderr}"
                print(f"[ERROR] {error_msg}")
                return f"Error: UniDic package is not installed and automatic installation failed.\n\nPlease run these commands manually:\n\npip install unidic\npython -m unidic download"

        # Run the unidic download command
        print("Downloading UniDic dictionary...")
        process = subprocess.Popen(
            [sys.executable, "-m", "unidic", "download"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout, stderr = process.communicate()

        if process.returncode == 0:
            success_msg = "UniDic dictionary downloaded successfully! Japanese TTS should now work properly."
            print(success_msg)
            return success_msg
        else:
            error_msg = f"Error downloading UniDic: {stderr}"
            print(f"[ERROR] {error_msg}")
            return f"Failed to download UniDic dictionary.\n\nError: {stderr}\n\nYou may need to run 'python -m unidic download' manually with administrator privileges."
    except Exception as e:
        error_msg = f"Error running unidic download: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return f"Error: {str(e)}\n\nTry running these commands manually:\n\npip install unidic\npython -m unidic download"


def download_ui():
    with gr.Accordion("Download UniDic Dictionary", open=False):
        unidic_btn = gr.Button(
            "Download UniDic Dictionary (Required)",
            variant="secondary",
        )
        unidic_output = gr.Textbox(label="UniDic Download Status", interactive=False)

        unidic_btn.click(fn=download_unidic, inputs=[], outputs=[unidic_output])
