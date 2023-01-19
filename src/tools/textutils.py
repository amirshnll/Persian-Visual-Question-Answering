from deep_translator import GoogleTranslator


def translate(text, source='en', target='fa'):
    translator = GoogleTranslator(source=source, target=target)
    translated = translator.translate(text=text)
    return translated

