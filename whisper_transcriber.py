import time
# from faster_whisper import WhisperModel
#
# model = WhisperModel(
#     "base",
#     device="cuda",
#     compute_type="float16"
# )

# import torch
#
# print("CUDA available:", torch.cuda.is_available())
# print("GPU:", torch.cuda.get_device_name(0))


from faster_whisper import WhisperModel

# model = WhisperModel("base", device="cuda", compute_type="float16")
# Transcription took 236.25s

# can be 7x slower - Transcription took 1507.76s
model = WhisperModel("base", device="cpu", compute_type="float32")

segments, _ = model.transcribe("china_podcast.mp3", language="pt", task="transcribe", vad_filter=True, word_timestamps=False)

# for s in segments:
#     print(s.start)
#     print(s.end)
#     print(s.text)

# for s in segments:
#       print(s.start, s.end, s.text)
#       if s.words:
#           for word in s.words:
#               print(f"  {word.start:.2f} → {word.end:.2f}  '{word.word}'  (prob: {word.probability:.2f})")

start = time.time()
for s in segments:
    print(f"{s.start} --> {s.text}")
end = time.time()

print(f"\nTranscription took {end - start:.2f}s")