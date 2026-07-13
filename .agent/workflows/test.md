---
description: Test transcription endpoints
---

# Test Endpoints

// turbo-all

1. Health check:

```bash
curl http://localhost:9000/
```

1. Status:

```bash
curl http://localhost:9000/status
```

1. Transcribe local file:

```bash
curl -X POST "http://localhost:9000/asr?local_path=/movies/test.mp4&language=en"
```

1. Transcribe uploaded file:

```bash
curl -X POST -F "audio_file=@test.mp3" http://localhost:9000/asr
```

1. Transcribe with custom parameters:

```bash
curl -X POST -F "audio_file=@test.mp3" "http://localhost:9000/asr?initial_prompt=Technical%20discussion&vad_filter=true&word_timestamps=true&max_line_width=42&max_line_count=2"
```

1. Transcribe with speaker diarization:

```bash
curl -X POST -F "audio_file=@test.mp3" "http://localhost:9000/asr?diarize=true&min_speakers=2&max_speakers=5"
```

1. Detect language:

```bash
curl -X POST -F "audio_file=@test.mp3" http://localhost:9000/detect-language
```

1. Get VTT output:

```bash
curl -X POST -F "audio_file=@test.mp3" "http://localhost:9000/asr?output=vtt"
```
