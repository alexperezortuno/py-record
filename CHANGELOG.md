# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The device used for the local transcription is checked before the model is loaded: the free
  video memory of the GPU is compared with the memory needed by the model, so a busy GPU is
  detected before the transcription starts
- Local transcription with `--whisper-device auto` now retries on CPU when CUDA runs out of
  memory (for example when a local LLM server occupies the GPU), instead of aborting the run
- LARM provider: summarize transcriptions with a local model served by LARM through its
  OpenAI compatible API (`--model larm`, `--larm-model`)
- New environment variables: `LARM_BASE_URL`, `LARM_MODEL`, `LARM_TIMEOUT`, `LARM_API_KEY`
- REC-003: `meeting analyze` extracts topics, decisions, action items and open questions
  from an existing transcription with a local model served by LARM
  (`--transcript`, `--provider`, `--larm-model`, `--format`, `--output`, `--overwrite`)
- REC-003: the answer of the model is validated before it is exported: the structure of the
  payload is checked and every element must be backed by a literal quote of the transcript
  with matching timestamps; unverified elements are rejected and reported
- REC-003: the exported quote is always the fragment of the transcript that was verified,
  and the timestamps come from the segment of the transcript that holds that fragment
  (the ones declared by the model are checked against it and rejected when they do not
  correspond)
- REC-003: export of the analysis as JSON and/or Markdown (`reunion_analysis.json` and
  `reunion_analysis.md`), without overwriting existing results unless `--overwrite` is used

---

## [1.0.8] - 2025-10-28

### Added

- add UI for recording

### Changed

- Remove global variable Output path for cli param
- Mode cli param to model

---

## [1.0.7] - 2025-10-20

### Added

- add windows support

---

## [1.0.6] - 2025-10-17

### Changed
- Documentation

---

## [1.0.5] - 2025-10-17

### Added
- Prompt multilingual support es/en
- charset-normalization package for environments with non-utf8 characters

### Fixed
- Fix logging
- Fix start/stop recording
- Fix installer with toml

---

## [1.0.4] - 2025-10-16

### Added
- .env Template
- Installer in README
- Usage in README
- multilingual support and refine summary prompts

### Fixed
- Installer

---

## [1.0.0] - 2025-10-16

- Add recording
- Add transcript with local, OpenAI, Gemini
- Add diarize
