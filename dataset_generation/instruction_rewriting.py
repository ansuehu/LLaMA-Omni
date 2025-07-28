from transformers import pipeline
from datasets import load_from_disk
import json
import argparse
from tqdm import tqdm

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

    # prompt = f'''
    # Hona hemen erabiltzailearen jarraibidea duen datu bat. Hizketazko bertsio bat sortu nahi nuke
    # hizkuntza eredu handi bat entrenatzeko, ahots-sarrera onartzen duena. Horregatik, mesedez,
    # berridatzi nire jarraibide-datua honako eskakizun hauei jarraituz:
    # 1. Galderak ez luke eduki behar TTS ereduak ezin dituen sintetizatu. Zenbakiak hitzez
    # (ingelesez) idatzi behar dira, ez zenbaki arabiar gisa.
    # 2. Galdera laburra izan behar da, hitz alferrikako gehiegirik gabe.
    # [jarraibidea]: {instruction}
    # Mesedez, erantzun JSON formatuan honela: "galdera": galdera.
    # '''

    prompt = f'''
        Hona hemen testu bat. Itzuli ezazu Euskarara.
        Testua: {instruction} 
    '''
    return prompt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Path to input dataset')
    parser.add_argument('--instruction_path', type=str, required=True, help='Path to save the output JSON')
    parser.add_argument('--answer_path', type=str, required=True, help='Path to save the output JSON')
    args = parser.parse_args()

    # Load dataset
    dataset = load_from_disk(args.input_path)
    # dataset = load_dataset("json", data_files="instruct_en_10_cleaned.json")
    print(dataset)

    # Load model pipeline
    pipe = pipeline("translation", model="HiTZ/mt-hitz-en-eu", batch_size=1000)
    # pipe = pipeline("translation", model="facebook/nllb-200-distilled-600M")
    # pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-eu")

    # pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct", device_map="auto", do_sample=False)
    # pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct", device_map="auto", do_sample=False, token='...')

    # Collect results
    instruct_data = []
    answer_data = []


    instruction_texts = [x['conversation'][0]["text"] for x in dataset]
    answer_texts = [x['conversation'][1]["text"] for x in dataset]

    instruct_data = pipe(instruction_texts, truncation='only_first')
    answer_data = pipe(answer_texts, truncation='only_first')

    # for i, example in enumerate(tqdm(dataset['conversation'])):
    #     # if i >= 10:
    #     #     break
    #     # print(example)
    #     for j in range(0, len(example), 2):
    #         try:
    #             instruction = example[j]['text']
    #             answer = example[j+1]['text']
    #             # instruc_prompt = instruct_to_prompt(instruction)
    #             # answer_prompt = instruct_to_prompt(answer)

    #             # instruct_response = pipe(instruc_prompt, max_new_tokens=200, return_full_text=False)
    #             # answer_response = pipe(answer_prompt, max_new_tokens=200, return_full_text=False)
    #             instruct_translation = pipe(instruction, src_lang = 'eng_Latn', tgt_lang = 'eus_Latn')
    #             answer_translation = pipe(answer, src_lang = 'eng_Latn', tgt_lang = 'eus_Latn')
            

    #             # generated_text = instruct_response[0]['generated_text']
    #             # instruct_data.append(generated_text)
    #             # generated_text = answer_response[0]['generated_text']
    #             # answer_data.append(generated_text)
    #             instruct_data.append(instruct_translation[0]['translation_text'])
    #             answer_data.append(answer_translation[0]['translation_text'])
    #         except Exception as e:
    #             print(f"Failed to parse response on example {i}: {e}")
    #             # print(response)

    #     # Save to file

    with open(args.instruction_path, "w") as f:
        for item in instruct_data:
            f.write(str(item) + "\n")
    
    with open(args.answer_path, "w") as f:
        for item in answer_data:
            f.write(str(item) + "\n")

