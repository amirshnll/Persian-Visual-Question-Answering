from src.tools.pyutils import save_file


def tokenize_questions(data):

    for d in data:
        q = d['fa_question']

        # Remove Punctuations
        q = q.replace(u'\u200c', ' ')
        q = q.replace(u'\u200b', ' ')
        q = q.replace("?", ' ')
        q = q.replace("؟", ' ')
        q = q.replace("«", ' ')
        q = q.replace("»", ' ')
        q = q.replace("،", ' ')
        q = q.replace("(", ' ')
        q = q.replace(")", ' ')
        q = q.replace('ً', ' ')
        q = q.replace('"', ' ')
        q = q.replace('‌', ' ')
        q = q.replace('‍', ' ')
        q = q.replace('!', ' ')
        q = q.replace('/', ' ')
        q = q.replace('#', ' شماره ')
        q = q.replace(': ', ' ').replace(' :', ' ')
        q = q.replace('...', ' ')
        q = q.replace('..', ' ')
        q = q.replace('. ', ' ').replace(' .', ' ')
        q = q.replace('-', ' ')
        q = q.replace('_', ' ')
        q = q.replace("'", ' ')
        q = q.replace('ِ', ' ')
        q = q.replace('ّ', ' ')
        q = q.replace(' *', ' ').replace('* ', ' ')
        q = q.replace('×', '*')

        # q = normalizer.affix_spacing(text=q)
        # q = normalizer.character_refinement(text=q)
        # q = normalizer.normalize(text=q)

        # q = word_tokenize(sentence=q)
        # q = normalizer.token_spacing(tokens=q)
        q = q.split(sep=' ')
        # Preprocess Tokens
        # for i in range(len(q)):
        #     q[i] = lemmatizer.lemmatize(word=q[i])

        # Remove Conjugation characters
        q = list(filter(lambda x: x != 'و', q))
        q = list(filter(lambda x: x != 'ی', q))
        q = list(filter(lambda x: x != 'ها', q))
        q = list(filter(lambda x: x != 'های', q))
        q = list(filter(lambda x: x != 'می', q))
        q = list(filter(lambda x: x != 'اند', q))
        q = list(filter(lambda x: x != 'است', q))
        q = list(filter(lambda x: x != 'ای', q))
        q = list(filter(lambda x: x != 'چیزی', q))
        q = list(filter(lambda x: x != 'این', q))
        q = list(filter(lambda x: x != 'از', q))
        q = list(filter(lambda x: x != 'را', q))
        q = list(filter(lambda x: x != 'ایا', q))
        q = list(filter(lambda x: x != 'آیا', q))
        q = list(filter(lambda x: x != '', q))
        q = list(filter(lambda x: x != 'آن', q))
        q = list(filter(lambda x: x != 'پس', q))
        q = list(filter(lambda x: x != 'نوع', q))
        q = list(filter(lambda x: x != 'شده', q))

        # Remove Pre-position and Post-position characters

        # if 'در' in q:
        #     q = list(filter(lambda x: x != 'در', q))
        # if 'به' in q:
        #     q = list(filter(lambda x: x != 'به', q))
        # if 'با' in q:
        #     q = list(filter(lambda x: x != 'با', q))
        # if 'برای' in q:
        #     q = list(filter(lambda x: x != 'برای', q))
        # if 'بی' in q:
        #     q = list(filter(lambda x: x != 'بی', q))
        # if 'بر' in q:
        #     q = list(filter(lambda x: x != 'بر', q))
        # if 'درباره' in q:
        #     q = list(filter(lambda x: x != 'درباره', q))
        # if 'تا' in q:
        #     q = list(filter(lambda x: x != 'تا', q))

        # if 'جز' in q:
        #     q = list(filter(lambda x: x != 'جز', q))
        # if 'بدون' in q:
        #     q = list(filter(lambda x: x != 'بدون', q))
        # if 'چون' in q:
        #     q = list(filter(lambda x: x != 'چون', q))
        # if 'مانند' in q:
        #     q = list(filter(lambda x: x != 'مانند', q))
        # if 'مثل' in q:
        #     q = list(filter(lambda x: x != 'مثل', q))
        # if 'مگر' in q:
        #     q = list(filter(lambda x: x != 'مگر', q))
        # if 'الا' in q:
        #     q = list(filter(lambda x: x != 'الا', q))

        # Remove Unnecessary words


        d['tokens'] = q

    return data


def get_tokens_length_frequency(data):
    temp = {}

    for i in range(len(data)):
        ql = len(data[i]['tokens'])
        if ql not in temp:
            temp[ql] = 1
        else:
            temp[ql] += 1

    temp = sorted(temp.items(), key=lambda x: x[0])
    return {x[0]: x[1] for x in temp}


def filter_by_tokens_length(data, tokens_length):

    for d in data:
        d['tokens'] = d['tokens'][-tokens_length:]
    return data


def get_tokens_frequency(data):

    tokens_frequency = {}

    for d in data:
        for t in d['tokens']:
            if t not in tokens_frequency:
                tokens_frequency[t] = 1
            else:
                tokens_frequency[t] += 1

    return tokens_frequency


def get_useful_tokens(tokens_frequency, min_occ):
    return {k: v for k, v in tokens_frequency.items() if v >= min_occ}


def get_tokens_dictionary(useful_tokens, save_path, idx_to_token_save_name, token_to_idx_save_name):

    token_to_idx = {}

    i = 1
    for key, value in useful_tokens.items():
        token_to_idx[key] = i
        i += 1

    token_to_idx['UKN'] = i

    idx_to_token = {v: k for k, v in token_to_idx.items()}

    save_file(data=idx_to_token, path=save_path, file_name=idx_to_token_save_name, file_type='pickle')
    save_file(data=token_to_idx, path=save_path, file_name=token_to_idx_save_name, file_type='pickle')

    return idx_to_token, token_to_idx

