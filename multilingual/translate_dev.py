import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()
# Chinese
# Japanese
# Contonese
language = "Contonese"  # Change this to the target language you want to translate to
max_workers = 50  # Number of concurrent threads to use for translation
# Paths to your datasets
path_to_dev = os.path.join('..','..', 'dataset', 'bird-sql', 'dev', 'dev.json')
path_to_save = os.path.join('..','..', 'dataset', 'bird-sql', 'dev', 'dev_{}.json')

def get_abs_path_from_rel(rel_path):
    return os.path.join(os.path.dirname(__file__), rel_path)
path_to_dev = get_abs_path_from_rel(path_to_dev)
path_to_save = get_abs_path_from_rel(path_to_save)

def translate_to_english(data):
    question = data['question']
    try:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Translate the following question into {language}"},
                {"role": "user", "content": question}
            ],
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.5,
        )
        # Extract the translation from the response
        translation = response.choices[0].message.content.strip()
        data['question'] = translation
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        data['question'] = question  # Fallback to original question if translation fails
    return data

def main():
    # Load the dataset
    with open(path_to_dev, 'r', encoding='utf-8') as f:
        dev = json.load(f)

    # Use ThreadPoolExecutor for concurrent translation
    translated_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Adjust max_workers as needed
        futures = {executor.submit(translate_to_english, data): data for data in dev}
        for future in tqdm(as_completed(futures), total=len(futures)):
            translated_data.append(future.result())

    # Sort the results by question_id
    translated_data.sort(key=lambda x: x['question_id'])

    # Save the translated dataset
    with open(path_to_save.format(language), 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()