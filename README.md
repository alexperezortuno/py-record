# py-record


### System requirements

```shell
sudo apt update
```

```shell
sudo apt install ffmpeg pavucontrol cmake pkg-config libsentencepiece-dev -y
```

in conda or env you need first this

```shell
pip install -U pip setuptools wheel
```

```shell
pip install -r requirements.txt
```


### How to get audio devices

```shell
pactl list short sources
```

### Commands

```shell
python -m meeting.cli diarize --audio ~/meetings/meeting_2025-10-14_22-45-00.mp3

```
### Install this repo

Get the repo

```shell
git clone https://github.com/alexperezortuno/py-record
```

go to the repo

```shell
cd py-record
```

```shell
pip install -e .
```

# Usage

```shell
meeting -h
```

```shell
meeting record
```

```shell
meeting transcript -e --mode openai --audio ~/meetings/meeting_2025-10-14_22-45-00.mp3
```


```shell
meeting transcript -e --mode gemini --audio ~/meetings/meeting_2025-10-14_22-45-00.mp3
```