import json
import pandas as pd

# ----------------------------------------------------
# PARAMETERS
# ----------------------------------------------------
INPUT_JSON_PATH = 'apple_news.json'
OUTPUT_JSON_PATH = 'apple_news_filtered.json'

# List of keywords that, when found near "Apple", suggest the article is NOT about the company
EXCLUSION_TERMS = ['Gwyneth Paltrow', 'Martin', 'pie', 'salad', 'cider', 'fruit', 'auction', 'dress', 'Paris',
                   'debutante', 'cookbook', 'XCX', 'hotel']

# List of keywords that suggest the article IS about the company
INCLUSION_TERMS = ['iPhone', 'iPad', 'Mac', 'Cook', 'stocks', 'share', 'earnings', 'Nasdaq', 'profit', 'Tim Cook',
                   'iOS', 'patent', 'supplier']


# ----------------------------------------------------
# FILTERING AND SAVING FUNCTION
# ----------------------------------------------------
def filter_and_save_news(input_path, output_path, exclusion_terms, inclusion_terms):
    # Loading data from JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df_raw = pd.DataFrame(data)

    # Combining title, description and content
    df_raw['combined_text'] = (
            df_raw['title'].astype(str).fillna('') +
            ' ' + df_raw['description'].astype(str).fillna('') +
            ' ' + df_raw['content'].astype(str).fillna('')
    )

    # FILTER LOGIC
    # 1. Article has to contain "Apple"
    must_contain_apple = df_raw['combined_text'].str.contains('Apple', case=False, na=False)

    # 2. Exclusion Mask: True for articles that contain exclusion words
    pattern_exclude = '|'.join([term.lower() for term in exclusion_terms])
    is_exclusion = df_raw['combined_text'].str.lower().str.contains(pattern_exclude, na=False)

    # 3. Boost Mask: True for articles that contain company keywords
    pattern_include = '|'.join([term.lower() for term in inclusion_terms])
    is_inclusion = df_raw['combined_text'].str.lower().str.contains(pattern_include, na=False)

    # Article is about the Apple company, if:
    # a) Contains "Apple" and doesn't contain exclusion words
    # OR
    # b) Contains "Apple" and contains company keywords

    final_mask = must_contain_apple & (~is_exclusion | is_inclusion)

    df_filtered = df_raw[final_mask].copy()

    # SAVING DATA
    # Converting DataFrame back to a list of JSON dictionaries
    # Using orient='records' to get the list of JSON objects format
    filtered_data = df_filtered.to_dict(orient='records')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)

    print(f"--- FILTRATION SUMMARY ---")
    print(f"Loaded: {len(df_raw)} articles.")
    print(f"Remaining after filtration (about Apple): {len(df_filtered)} articles.")
    print(f"Filtered data saved to file: {output_path}")


# ----------------------------------------------------
# EXECUTION
# ----------------------------------------------------
filter_and_save_news(INPUT_JSON_PATH, OUTPUT_JSON_PATH, EXCLUSION_TERMS, INCLUSION_TERMS)
