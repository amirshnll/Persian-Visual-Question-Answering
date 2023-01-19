import re
import os
from hazm import Normalizer
from deep_translator import GoogleTranslator
from src.tools.pyutils import save_file, load_file


def translate(text, source='en', target='fa'):
    translator = GoogleTranslator(source=source, target=target)
    translated = translator.translate(text=text)
    return translated


def translate_data(data, save_path, dictionary_name):

    normalizer = Normalizer(
        remove_extra_spaces=False,
        persian_style=False,
        persian_numbers=True,
        remove_diacritics=False,
        affix_spacing=False,
        token_based=False,
        punctuation_spacing=False
    )

    if os.path.exists(path=os.path.join(save_path, dictionary_name)):
        dictionary = load_file(path=os.path.join(save_path, dictionary_name))
        print('dictionary found!')
    else:
        dictionary = {}

    for d in data:
        eq = d['en_question']
        ea = d['en_answer']

        # Translating Question Text
        if eq in dictionary:
            pq = dictionary[eq]
        elif eq.isdigit():
            pq = eq
        else:
            pq = translate(text=eq)
            dictionary[eq] = pq
            save_file(data=dictionary, path=save_path, file_name=dictionary_name, file_type='pickle')

        # Translating Answer Text
        if ea in dictionary:
            pa = dictionary[ea]
        elif ea.isdigit():
            pa = ea
        else:
            pa = translate(text=ea)
            dictionary[ea] = pa
            save_file(data=dictionary, path=save_path, file_name=dictionary_name, file_type='pickle')

        pq = normalizer.normalize(text=pq)
        pa = normalizer.normalize(text=pa)

        d['fa_question'] = pq
        d['fa_answer'] = pa

    return data


def remove_repeated_data(data):

    que_ans_img = {}

    for d in data:
        q = d['fa_question']
        a = d['fa_answer']
        i = d['image_id']
        d['repeated_data'] = False

        qai = f'{q}|||{a}|||{i}'

        if qai not in que_ans_img:
            que_ans_img[qai] = ''

        else:
            d['repeated_data'] = True

    data = [d for d in data if d['repeated_data'] is False]

    return data


def remove_english_qas(data):

    data = [d for d in data if (
        (
            re.search(pattern='[a-zA-Z]', string=d['fa_question']) is None
        )
        and
        (
            re.search(pattern='[a-zA-Z]', string=d['fa_answer']) is None
        )
    )]

    return data
