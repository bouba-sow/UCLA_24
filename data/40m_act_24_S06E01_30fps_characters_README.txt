40m_act_24_S06E01_30fps_characters_README.txt

40m_act_24_S06E01_30fps_characters.csv contains labels for 15 characters (one per column) in the first episode of season 6 of the TV show 24.

The label 'Face' indicates a face being prominent on screen (regardless of the identity of the face).
The label 'Person' indicates a non-named character being prominent on screen.
The label 'No Characters' indicates tha no characters are on screen.

The labels are at frame-level resolution.

The fps of the video is 29.97002997002997 (according to opencv2). 

For the most accurate alignment of the labels with the neural data, the frame time found by n_frames/fps should be multiplied by the drift correction factor found in the *_audio_movie_start_time.json files for each recording (or the neural data shifted accordingly).
