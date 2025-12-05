## Workflow
1. get word videos from UniSignMimic
2. extract frames from word videos
3. filter1: rtmpose detection (confidence, velocity)
4. filter2: frame deduction (remove redundant frames between words)
5. extract boundary frames: start + end
6. run Framer: generate interpolation frames from Framer
7. combine word images and interpolation gifs to resulk videos