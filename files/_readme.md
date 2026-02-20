Audio Transcriber 19 Fev 2026,current folder name python project is audio_to_text avoid env issues lets keep it

Transcriber: mp3 --> text (with or without sentences, word timestamps).
We keep data output not as .srt but different formats possible
- for subtitles with sentences timestamps
- karaoke with words timestamps
- for summaries with just text
should handle CPU and GPU options, languages ...

to be used by:
1) karaoke
2) subtitles
3) media to database
cut segment mp4 with some text or that contains text
(TEXT to video, depends on DB might not be here, database use case or server)
4) etc

- TODO
-- can this be reused somehow always same files picked from folder default like mp3, mp4 defined path in constants to be able to use others
-- use CPU or GPU
-- use only faster whisper remove old code keep just an archive with not used code
-- modules for subtitles, music, karaoke, etc
-- client calls all
-- menu modules with all text options consts
-- files: inputs, outputs, processed. Names with timestamps


////////////////// old readme /////////////////////////////////////////
This project is under refactoring 30 Jan 2026
Deleted subtitles generator from langum server
when stable in GIT as individual project
start commiting soon as this structure gets stable


- transcriber module should use whisper with GPU or CPU (threads), models should
be saved offline
- att_client (audio_to_text_client) maestro just call modules ffmpeg calls should be done in media_utils
- 

MAYBE series in other table? musics? videos??
MAYBE reviews: author, timestamp, is native...
MAYBE fts5 full text search