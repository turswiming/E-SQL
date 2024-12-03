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
language = "Chinese"  # Change this to the target language you want to translate to
max_workers = 10  # Number of concurrent threads to use for translation
# Paths to your datasets
path_to_dev = os.path.join('.', 'dataset', 'bird-sql', 'dev', 'column_meaning.json')
path_to_save = os.path.join('.', 'dataset', 'bird-sql', 'dev', 'column_meaning_{}.json')

def translate_to_english(data):
    question = data
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
        data = translation
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        data = question  # Fallback to original question if translation fails
    return data

def translate_column_meanings(column_meaning):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Adjust max_workers as needed
        futures = {executor.submit(translate_to_english, column_meaning[key]): key for key in column_meaning.keys()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Translating"):
            key = futures[future]
            try:
                column_meaning[key] = future.result()
            except Exception as e:
                print(f"An error occurred during translation for key {key}: {e}")
    return column_meaning

def main():
    # Load the dataset
    with open(path_to_dev, 'r', encoding='utf-8') as f:
        column_meaning = json.load(f)
    
    translated_column_meaning = translate_column_meanings(column_meaning)


    # Sort the results by question_id

    # Save the translated dataset
    with open(path_to_save.format(language), 'w', encoding='utf-8') as f:
        json.dump(translated_column_meaning, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()