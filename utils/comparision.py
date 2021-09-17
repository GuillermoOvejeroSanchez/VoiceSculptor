import parselmouth  # https://parselmouth.readthedocs.io/en/stable/
from utils.syllabe_nuclei import speech_rate
from pathlib import Path
import os
import numpy as np
import json

path = Path("static/sounds/")


info_dict = {}

for file in os.listdir(path):
    if file.startswith("0") or file[0].isupper():
        print(file)
        snd = parselmouth.Sound(str(path / file))
        data = speech_rate(snd)
        intensity = snd.to_intensity().values.T
        pitch = snd.to_pitch()
        pitch = pitch.selected_array["frequency"]

        data["intensity mean"] = np.mean(intensity)
        data["intensity std"] = np.std(intensity)
        data["pitch mean"] = np.mean(pitch)
        data["pitch std"] = np.std(pitch)

        info_dict[file] = data


compare_dict = {}
# Con 0 y _ es Despues, solo nombre antes
for key in info_dict.keys():
    key_s = key.split("_")
    if len(key_s) > 1:
        compare_dict[key_s[0]] = {}
        compare_dict[key_s[0]]["AFTER"] = info_dict[key]
        compare_dict[key_s[0]]["BEFORE"] = info_dict[key_s[1]]


with open("compare_dict.json", "w") as outfile:
    json.dump(compare_dict, outfile)

with open("compare_dict.json", "r") as outfile:
    compare_dict = json.load(outfile)

table_latex = """
\subsubsection{{Student number {file}}}

\\begin{{table}}[H]
\centering
\\begin{{tabular}}{{|l|r|r|}}
\hline
&Before course &After course \\\\
\hline
Mean intensity &{before_mean_intensity}  &{after_mean_intensity} \\\\
\hline
Std intensity &{before_std_intensity}  &{after_std_intensity} \\\\
\hline
Mean pitch(Hz) &{before_mean_pitch}  &{after_mean_pitch} \\\\
\hline
Std pitch(Hz) &{before_std_pitch}  &{after_std_pitch} \\\\
\hline
Duration &{before_duration}  &{after_duration} \\\\
\hline
Number of pauses per minute &{before_pauses}  &{after_pauses} \\\\
\hline
Speech rate (syllabus/duration) &{before_srate} &{after_srate} \\\\
\hline
ASD(speakingtime / nsyll) &{before_asd} &{after_asd} \\\\
\hline
\end{{tabular}}
\caption{{Student number {file}}}
\label{{tab:{file}_table}}
\end{{table}}

"""

# "nsyll"
# "npause"
# "dur(s)"
# "phonationtime(s)"
# "speechrate(nsyll / dur)"
# "articulation rate(nsyll / phonationtime)"
# "ASD(speakingtime / nsyll)"


file_object = open("tables.txt", "a")
for k, v in compare_dict.items():
    before: dict = v.get("BEFORE")
    after: dict = v.get("AFTER")
    pauses_after = after.get("npause") / (after.get("dur(s)") / 60)
    pauses_before = before.get("npause") / (before.get("dur(s)") / 60)
    table_formatted = table_latex.format(
        file=k,
        before_mean_intensity=round(before.get("intensity mean"), 2),
        after_mean_intensity=round(after.get("intensity mean"), 2),
        before_std_intensity=round(before.get("intensity std"), 2),
        after_std_intensity=round(after.get("intensity std"), 2),
        before_mean_pitch=round(before.get("pitch mean"), 2),
        after_mean_pitch=round(after.get("pitch mean"), 2),
        before_std_pitch=round(before.get("pitch std"), 2),
        after_std_pitch=round(after.get("pitch std"), 2),
        before_duration=round(before.get("dur(s)"), 2),
        after_duration=round(after.get("dur(s)"), 2),
        before_phonation=round(before.get("phonationtime(s)"), 2),
        after_phonation=round(after.get("phonationtime(s)"), 2),
        before_pauses=round(pauses_before, 2),
        after_pauses=round(pauses_after, 2),
        before_srate=round(before.get("speechrate(nsyll / dur)"), 2),
        after_srate=round(after.get("speechrate(nsyll / dur)"), 2),
        before_arate=round(before.get("articulation rate(nsyll / phonationtime)"), 2),
        after_arate=round(after.get("articulation rate(nsyll / phonationtime)"), 2),
        before_asd=round(before.get("ASD(speakingtime / nsyll)"), 2),
        after_asd=round(after.get("ASD(speakingtime / nsyll)"), 2),
    )
    file_object.write(table_formatted)
