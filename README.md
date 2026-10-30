# iris-listen

Speech to text on your own GPU: [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)
behind OpenAI's `POST /v1/audio/transcriptions` and a bearer token. It is the
ears of [Iris](https://github.com/topics/iris); her voice is
[iris-voice](https://github.com/p4ulcristian/iris-voice).
[omarchy-dictation](https://github.com/p4ulcristian/omarchy-dictation)'s
push-to-talk types what it hears.

## How it fits

```
any program ── Bearer token, a clip (+ language, vocabulary) ──▶ iris-listen (server.py, Qwen3-ASR, GPU)
            ◀── {"text": "..."} ────────────────────────────────
```

- **The token**: every request needs `Authorization: Bearer <token>`, or the
  answer is 401. The token is in `~/.config/iris-listen/token`, made by the
  first install. So the service is safe on any address, with or without a
  proxy in front.
- **The model**: the best open model for English on the Open ASR Leaderboard
  (September 2026: 4.31% mean WER). It writes down what was said, never a
  translation, in 30 languages: ar cs da de el en es fa fi fil fr hi hu id it
  ja ko mk ms nl pl pt ro ru sv th tr vi yue zh.
- **Whole clips**: it hears a clip at once, not word by word, so there is no
  stream. A clip over 28 s is cut at its quietest moments and the pieces'
  texts joined.
- **Speed** on an RTX 5060 Ti: about 0.3 s for a 7 s sentence.
- **Memory**: about 4.6 GB of GPU memory while it runs.

## Install

```bash
git clone https://github.com/p4ulcristian/iris-listen && iris-listen/install.sh
```

It makes the token, sets up `.venv` here, downloads the model (3.9 GB, to
`~/.local/share/iris-listen/models/qwen3-asr-1.7b`, or `LISTEN_MODEL_DIR`) and
starts the user service `iris-listen` on `127.0.0.1:7861` (`LISTEN_PORT`).
`LISTEN_HOSTS="127.0.0.1 <address>"` makes it listen on more addresses, a
VPN's for a proxy in front; an address that is not up yet is bound anyway and
answers once it is. `LISTEN_TOKEN_FILE` puts the token somewhere else.

## API

`POST /v1/audio/transcriptions`, multipart form:

| field | |
|---|---|
| `file` | the clip: WAV, FLAC or OGG, any sample rate, up to 25 MB |
| `language` | optional, e.g. `en`: forces that language; without it the model tells |
| `prompt` | optional: names and terms the clip may hold ("Omarchy, Postgres"), which the model leans towards |
| `response_format` | `json` (default): `{"text": "..."}`; `text`: the text alone |
| `model` | accepted and ignored, for OpenAI's clients |

One clip at a time; others wait. `GET /health` answers `ok`.

```bash
curl -s -H "Authorization: Bearer $(cat ~/.config/iris-listen/token)" \
  -F file=@hello.wav -F language=en http://127.0.0.1:7861/v1/audio/transcriptions
```
