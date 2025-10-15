# py-record


### System requirements

```shell
sudo apt update
```

```shell
sudo apt install ffmpeg pavucontrol cmake pkg-config libsentencepiece-dev -y
```

### How to get audio devices

```shell
pactl list short sources
```

### Commands

```shell
python -m meeting.cli diarize --audio ~/meetings/meeting_2025-10-14_22-45-00.mp3
```
