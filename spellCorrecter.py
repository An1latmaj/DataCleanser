import pandas as pd
import nltk
from textblob import TextBlob

nltk.download("punkt_tab")
file_path = "misspelled.csv"
df = pd.read_csv(file_path)


def correct_spelling(text):
    if isinstance(text, str):
        blob = TextBlob(text)
        corrected_text = blob.correct()
        return str(corrected_text)
    return text


def find_spelling_errors(text):
    if isinstance(text, str):
        blob = TextBlob(text)
        errors = {
            word: word.correct()
            for word in blob.words
            if word != word.correct()
        }
        return errors
    return {}


spelling_errors = []
corrected_data = df.copy()

for column in df.columns:
    if df[column].dtype == object:
        df[f"{column}_SpellingErrors"] = df[column].apply(find_spelling_errors)
        corrected_data[column] = df[column].apply(correct_spelling)

corrected_data.to_csv("corrected_file.csv", index=False)
df.to_csv("spelling_errors.csv", index=False)

print("Spelling corrections saved successfully! Check 'corrected_file.csv' and 'spelling_errors.csv' for details.")
