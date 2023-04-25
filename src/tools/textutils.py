from deep_translator import GoogleTranslator


def translate(text, source='en', target='fa'):
    translator = GoogleTranslator(source=source, target=target)
    return translator.translate(text=text)

