# MusicCaps Audio Coverage

Original dataset: 5521 clips (per spec and official release).
Successfully downloaded: 5278 clips (95.6%).
Unrecoverable: 243 clips (4.4%), broken down as:

- 189 clips: genuinely removed from YouTube (private videos, terminated
  accounts, copyright takedowns, ToS/policy violations) — permanent,
  unrecoverable regardless of method.
- 46 clips: blocked by YouTube's age/sign-in verification wall.
  Recoverable in principle via authenticated browser cookies, but
  blocked in practice by Chrome's App-Bound Encryption on Windows,
  which prevents yt-dlp's cookie extraction (DPAPI decryption failure,
  a known limitation: https://github.com/yt-dlp/yt-dlp/issues/10927).
  Not pursued further given the small fraction affected (0.8% of the
  full dataset).
- ~8 clips: transient "video is being processed" or uncategorized
  errors, not pursued given the small number involved.

This attrition is expected for any YouTube-sourced dataset accessed
years after its original release; 95.6% coverage is treated as the
practical ceiling for this project.