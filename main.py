from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
import os
from vosk import Model, KaldiRecognizer
import pyaudio
import json
import spacy
from spacy_langdetect import LanguageDetector
from spacy.language import Language

# Load the English and French models
nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")

# # Add language detection (optional if you want to handle multilingual input)
# @Language.factory("language_detector")
# def get_lang_detector(nlp, name):
#     return LanguageDetector()
#
# nlp_en.add_pipe('language_detector', last=True)
# nlp_fr.add_pipe('language_detector', last=True)

# List of common filler words in English and French
filler_words = {
    'en': ["okay", "um", "uh", "well", "like", "you know", "so"],
    'fr': ["d'accord", "euh", "ben", "alors", "hein", "en fait"]
}

# List of interrogative pronouns and auxiliary verbs for English and French
interrogative_words = {
    'en': ["what", "why", "how", "who", "when", "where", "which"],
    'fr': ["quoi", "pourquoi", "comment", "qui", "quand", "où", "lequel"]
}

auxiliary_verbs = {
    'en': ["can", "could", "will", "would", "shall", "should", "may", "might", "do", "does", "did", "is", "are", "was", "were"],
    'fr': ["peux", "puis", "voudrais", "pouvons", "fais", "dois", "es", "est", "sont", "étais", "serais"]
}


def call_llm(content):
    client = ChatNVIDIA(
        model="mistralai/mixtral-8x22b-instruct-v0.1",
        api_key=os.getenv("NVIDIA_NIM_API_KEY"),
        # temperature=0.5,
        # top_p=1,
        # max_tokens=1024,
    )

    for chunk in client.stream([{"role": "user", "content": content}]):
        print(chunk.content, end="")

    # add a new line after stream finishes
    print()


def init_pyaudio_stream():
    paa = pyaudio.PyAudio()

    devices = []
    device_count = paa.get_device_count()
    for i in range(device_count):
        device = paa.get_device_info_by_index(i)
        devices.append(device)
        print(device)

    device_indexes = list(map(lambda d: d['index'], devices))
    device_names = tuple(map(lambda d: d['name'], devices))

    selection = input(f"Please select a device ({device_indexes})...\n")
    print(f"Selected index: {selection} -> {device_names[int(selection)]}")

    try:
        selectionInt = int(selection)
    except LookupError:
        selectionInt = -1
        print("Input needs to be a whole number (0, 1, 2, ..etc)")
        quit()

    if selectionInt not in device_indexes:
        print(f"{selection} not available, options are {device_indexes}")
        quit()

    selectedDevice = devices[selectionInt]
    sampleRate = int(selectedDevice['defaultSampleRate'])
    print(f"Selection index: {selection}, name: {selectedDevice['name']}")

    stream = paa.open(format=pyaudio.paInt16, channels=1, rate=sampleRate, input=True, input_device_index=selectionInt, frames_per_buffer=8192)
    return stream, sampleRate


def remove_filler_words(text, language):
    filler_list = filler_words.get(language, [])
    tokens = text.split()
    # Remove filler words
    filtered_tokens = [token for token in tokens if token.lower() not in filler_list]
    return " ".join(filtered_tokens)


def format_text(text, language, intent):
    text = text.capitalize()

    if intent.get('has_question'):
        punctuation = "?"
    else:
        punctuation = "."

    if language == 'en':
        doc = nlp_en(text)
    elif language == 'fr':
        doc = nlp_fr(text)

    formatted_text = []
    for token in doc:
        # Capitalize proper nouns and the start of the sentence
        if token.pos_ == "PROPN" or token.i == 0:
            formatted_text.append(token.text.capitalize())
        else:
            formatted_text.append(token.text)

    formatted_sentence = " ".join(formatted_text)
    formatted_sentence = formatted_sentence.rstrip(".?") + punctuation

    return formatted_sentence.strip()


def detect_intent(doc, language):
    has_question = False
    is_action_request = False
    action_request_verb = None

    # Get the interrogative words and auxiliary verbs based on the language
    interrogatives = interrogative_words.get(language, [])
    auxiliaries = auxiliary_verbs.get(language, [])

    # Check for interrogative pronouns anywhere in the sentence
    for token in doc:
        if token.lemma_.lower() in auxiliaries:
            has_question = True
        if token.lemma_.lower() in interrogatives:
            has_question = True
            break  # Once we find an interrogative word, we assume it's a question

    # Check for action requests by detecting imperative verb forms
    if language == 'en':
        # English: Check for imperative verb at the start of the sentence
        if doc[0].pos_ == "VERB" and doc[0].tag_ == "VB":
            is_action_request = True
            action_request_verb = doc[0].text
    elif language == 'fr':
        # French: Check for imperative verbs (starting verbs in the infinitive form)
        if doc[0].pos_ == "VERB" and doc[0].tag_ == "VERB":
            is_action_request = True
            action_request_verb = doc[0].text

    return {
        "text": doc.text,
        "language": language,
        "has_question": has_question,
        "is_action_request": is_action_request,
        "action_request_verb": action_request_verb
    }


def detect_lang_and_route(text, language='en'):
    filtered_text = remove_filler_words(text, language)
    if language=='en':
        doc = nlp_en(filtered_text)
        return detect_intent(doc, 'en')
    elif language=='fr':
        doc = nlp_fr(filtered_text)
        return detect_intent(doc, 'fr')


def run_with_vosk(stream, sampleRate):
    model = Model(r"vosk-model-en-us-0.42-gigaspeech")
    model_language = 'en'
    recognizer = KaldiRecognizer(model, sampleRate)
    print('vosk model loaded')

    stream.start_stream()

    while True:
        data = stream.read(1024, exception_on_overflow=False)
        if len(data)==0:
            break
        if recognizer.AcceptWaveform(data):
            resultText = recognizer.Result()
            resultJson = json.loads(resultText)
            if len(resultJson['text']) > 0:
                text = resultJson['text']
                print(f"\npause detected for: {text}")
                intent = detect_lang_and_route(text, model_language)
                formatted_intent = format_text(text, model_language, intent)
                print(f"Intent: {intent}")
                print(f"Formatted Intent: {formatted_intent}")
                if intent.get('has_question') or intent.get('is_action_request'):
                    call_llm(formatted_intent)

                if resultJson['text']=='exit':
                    quit()
        else:
            resultText = recognizer.PartialResult()
            resultJson = json.loads(resultText)
            partialText = resultJson['partial']
            if len(partialText) > 0:
                print(f"\r{partialText}", end='')


def main():
    load_dotenv()

    stream, sampleRate = init_pyaudio_stream()
    run_with_vosk(stream, sampleRate)




if __name__=='__main__':
    main()