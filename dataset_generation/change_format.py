from tqdm import tqdm
from datasets import Dataset, Audio, load_dataset
from torchaudio import load, functional

# Suppose your dataset is loaded as a dict from JSON
dataset = load_dataset("Ansu/Instruct_200k_eu")
dataset = dataset['train']

with open('/scratch/andoni.sudupe/Instruct_S2S_eu/galdera_arazoak.txt', 'r') as file:
    # Read all lines into a list
    akatsak = file.read()

akatsak = akatsak.split("\n")[:-1]

akatsak = [eval(x)[0] for x in akatsak]
from tqdm import tqdm

#ids = [x.split('_')[-1] for x in akatsak]
dataset = dataset.filter(lambda x: x["id"] not in akatsak)
print(dataset)
# Flatten into rows
rows = []
for i, conv in enumerate(tqdm(dataset["conversation"])):
    for j in range(0, len(conv), 2):
        if conv[j]["from"] != "human" or conv[j+1]["from"] != "gpt":
            continue
        
        user_entry = conv[j]
        assistant_entry = conv[j+1]
        try:
            audio, sr = load(f'/scratch/andoni.sudupe/Instruct_S2S_eu/wavs/{user_entry["speech"]}')
        except:
            print(f"Error loading {user_entry['speech']}")
            continue
        audio = functional.resample(audio, orig_freq=sr, new_freq=16_000)
        rows.append({
            "id": dataset["id"][i],
            "round": j // 2 + 1,
            "question": f"<USER>: {user_entry['text']}",
            "answer": assistant_entry["text"],
            "answer_token": None,
            "question_audio": audio,  # HF will load waveform from path
        })

# Create a HF dataset
hf_dataset = Dataset.from_list(rows)

# Cast audio column to proper Audio type (HF handles loading as waveform)
# hf_dataset = hf_dataset.cast_column("question_audio", Audio(sampling_rate=16_000))  # set your SR
print(hf_dataset)
hf_dataset.push_to_hub('Ansu/Instruct_S2S_eu', token='...')
