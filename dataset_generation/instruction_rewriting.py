from transformers import pipeline
from datasets import load_dataset
import json
import argparse

def instruct_to_prompt(instruction):
    
    # prompt = f'''
    # Below is an instruction data containing the user's instruction. I would like to generate a speech version of
    # this instruction for training a large language model that supports speech input. Therefore, please rewrite my
    # instruction data according to the following requirements:
    # 1. Modify the instruction to simulate human speech, adding fillers as appropriate (but not too many "you
    # know", "like", etc.).
    # 2. The question should not contain content that cannot be synthesized by the TTS model. Numbers should
    # be written in English words rather than Arabic numerals.
    # 3. The question should be relatively brief without excessive verbiage.
    # [instruction]: {instruction}
    # Please output in JSON format as follows: "question": question.
    # '''

    prompt = f'''
    Hona hemen erabiltzailearen jarraibidea duen datu bat. Hizketazko bertsio bat sortu nahi nuke
    hizkuntza eredu handi bat entrenatzeko, ahots-sarrera onartzen duena. Horregatik, mesedez,
    berridatzi nire jarraibide-datua honako eskakizun hauei jarraituz:
    1. Galderak ez luke eduki behar TTS ereduak ezin dituen sintetizatu. Zenbakiak hitzez
    (ingelesez) idatzi behar dira, ez zenbaki arabiar gisa.
    2. Galdera laburra izan behar da, hitz alferrikako gehiegirik gabe.
    [jarraibidea]: {instruction}
    Mesedez, erantzun JSON formatuan honela: "galdera": galdera.
    '''
    return prompt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Path to input dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output JSON')
    parser.add_argument('--num_examples', type=int, default=10, help='Number of examples to process')
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(args.input_path)
    print(dataset)

    # Load model pipeline
    pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct", device_map="auto", do_sample=False, token='...')

    # Collect results
    output_data = []

    for i, example in enumerate(dataset['train']):
        if i >= 10:
            break
        # print(example)
        instruction = example['conversations'][0]['content']
        prompt = instruct_to_prompt(instruction)

        response = pipe(prompt, max_new_tokens=200, return_full_text=False)

        try:
            generated_text = response[0]['generated_text']
            output_data.append(generated_text)
        except Exception as e:
            print(f"Failed to parse response on example {i}: {e}")
            # print(response)

        # Save to file

        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump({"questions": output_data}, f, indent=2)

