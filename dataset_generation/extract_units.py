import argparse
import joblib
import torch
from transformers import Wav2Vec2Processor, HubertModel
from datasets import load_dataset
import numpy as np
from torchaudio import load, transforms

def process_audio(audio_array, sr, processor, model, kmeans, layer):
    """Extract HuBERT features and quantize into units with KMeans."""
    inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = inputs.to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    features = out.hidden_states[layer].squeeze(0).cpu().numpy()
    units = kmeans.predict(features)
    return units.tolist()

def main(args):
    dataset_repo = args.dataset_repo
    audio_path = args.audio_path
    kmeans_path = args.kmeans_path
    model_repo = args.model_repo
    layer = args.layer
    split = args.split
    push_repo = args.push_repo
    # Load kmeans model
    kmeans = joblib.load(kmeans_path)

    # Load pretrained HuBERT + processor
    processor = Wav2Vec2Processor.from_pretrained(model_repo)
    model = HubertModel.from_pretrained(model_repo)
    model.eval().to("cuda")

    # Load dataset
    dataset = load_dataset(dataset_repo, split=split)

    # Map function to add answer_token column
    def add_units(example):
        try:
            audio = f"{audio_path}{example['id']}_assistant_{example['round']-1}.wav"
            answer, old_sr = load(audio)
            sr = 16000
            answer = transforms.Resample(orig_freq=old_sr, new_freq=sr)(answer)
            answer = answer.squeeze(0).to("cuda")
            units = process_audio(answer, sr, processor, model, kmeans, layer)
            example["answer_token"] = units
            return example
        except Exception as e:
            print(f"{example['id']}: {e}")
            example["answer_token"] = [0]
            return example


    new_dataset = dataset.map(add_units)

    # Push dataset to HF Hub
    new_dataset.push_to_hub(push_repo, split=split, token='...')
    new_dataset.save_to_disk("/scratch/andoni.sudupe/Instruct_S2U_eu/hf")
    print(f"✅ Pushed dataset with units to {push_repo} (split: {split})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_repo", type=str, required=True, help="HF dataset repo to load from")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to audio files")
    parser.add_argument("--kmeans_path", type=str, required=True, help="Path to kmeans model")
    parser.add_argument("--model_repo", type=str, required=True, help="HF repo for HuBERT model")
    parser.add_argument("--layer", type=int, default=9, help="Hidden layer index for features")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/validation/test)")
    parser.add_argument("--push_repo", type=str, required=True, help="HF repo to push updated dataset to")
    args = parser.parse_args()

    main(args)

