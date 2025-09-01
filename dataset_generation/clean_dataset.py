from datasets import load_from_disk
import re

# === CONFIG ===
DATASET_PATH = "ICTNLP/InstructS2S-200K"  # Can be local folder or HF repo name
OUTPUT_PATH = "/content/cleaned_dataset"  # Output folder

# === Define filler patterns ===
fillers = [
    r"so\s*,",
    r"hey\s*,",
    r",\s*like,",
    r"\s*like,",
    r",\s*you know,",
    r"\s*you know,",
    r",\s*uh,",
    r"\s*uh,",
    r",\s*um,",
    r"um\s*,"
]

# The updated pattern should also handle fillers at the beginning of a sentence
pattern = re.compile(r"^\s*(?:" + "|".join(fillers) + r")\s*|(?:\s*(?:" + "|".join(fillers) + r"))\s*", flags=re.IGNORECASE)

def clean_text(text):
    if not isinstance(text, str):
        return text
    # Remove fillers
    text = pattern.sub(" ", text)
    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === Load dataset ===
dataset = load_from_disk(DATASET_PATH)

instruction_texts = [x['conversation'][0]["text"] for x in dataset]

# === Apply cleaning ===
def clean_split(split):
    def clean_conversation(example):
        cleaned_conversation = []
        for turn in range(0, len(example['conversation']), 2):
            cleaned_question = example['conversation'][turn].copy()
            cleaned_answer = example['conversation'][turn+1].copy()

            cleaned_question['text'] = clean_text(example['conversation'][turn]['text'])
            cleaned_conversation.append(cleaned_question)
            cleaned_conversation.append(cleaned_answer)
        return {"conversation": cleaned_conversation}
    return split.map(clean_conversation)

cleaned_dataset = clean_split(dataset)

# === Save cleaned dataset ===
cleaned_dataset.save_to_disk(f"{OUTPUT_PATH}")

print(f"✅ Dataset cleaned and saved to {OUTPUT_PATH}")
