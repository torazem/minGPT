from setuptools import setup

setup(
    name="minGPT",
    version="0.0.1",
    author="Andrej Karpathy",
    packages=["mingpt"],
    description="A PyTorch re-implementation of GPT",
    license="MIT",
    install_requires=[
        "torch",
        "regex",
        "requests",
        "transformers[torch]",
    ],
    extras_require={
        "streamlit": [
            "streamlit",
            "seaborn",
            "onnx",
            "onnxscript",
            "captum",
            "matplotlib==3.3.4",
        ],
    },
)
