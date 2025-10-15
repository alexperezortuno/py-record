from setuptools import setup, find_packages

setup(
    name="meeting-cli",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "faster-whisper>=0.10.0",
        "transformers>=4.30.0",
        "torch",
        "sentencepiece"
    ],
    entry_points={
        "console_scripts": [
            "meeting=meeting.cli:main",
        ],
    },
    author="Alexander Pérez Ortuño",
    author_email="alex@alexperezortuno.pro",
    description="Asistente de reuniones: graba, transcribe y resume tus reuniones automáticamente.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AlexPerezOrtuño/meeting-cli",
    python_requires=">=3.9",
)
