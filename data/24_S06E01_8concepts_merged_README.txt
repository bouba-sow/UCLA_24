24_S04E01_8concepts_merged_README.txt

24_S04E01_8concepts_merged.csv contains labels for 8 concepts (one per column) in the first episode of season 4 of the TV show 24. These labels were used to train transformer models in the Ding, Dunn, Sakon et al. paper.

The labels have 1 second resolution.

There are 2435 rows, so this is labeling for the first 2435 seconds of the movie.

The actual video file length is 2479.39 seconds. The last portion of the movie file is the credits which is not included in the labels.

For the most accurate alignment of the labels with the neural data, the video time should be multiplied by the drift correction factor found in the *_audio_movie_start_time.json files for each recording (or the neural data shifted accordingly).

The concepts are either characters, settings (incl. the characters unique to those settings), or more abstract.
In the latter cases, the themes associated with the label are detailed below. 

WhiteHouse: White House/DC/President/presidential staff
CTU: counter terrorism agency (often mistaken by participants for CIA/FBI)/ CTU staff eg C.OBrian
Hostage: hostage/exchange/sacrifice
Handcuff: handcuff/chair/tied
J.Bauer
B.Buchanan
A.Fayed
A.Amar
