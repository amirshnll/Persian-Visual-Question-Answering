from src.tools.pyutils import save_file


def get_answers_frequency(data):

    answers_frequency = {}

    for d in data:
        answer = d['fa_answer']
        if answer not in answers_frequency:
            answers_frequency[answer] = 1
        else:
            answers_frequency[answer] += 1

    return answers_frequency


def get_k_frequent_answers(answers_frequency, k):
    k_frequent_answers = sorted(answers_frequency.items(), key=lambda x: x[1], reverse=True)[:k]
    k_frequent_answers = {x[0]: x[1] for x in k_frequent_answers}
    return k_frequent_answers


def filter_by_k_frequent_answers(data, answers):
    data = [d for d in data if d['fa_answer'] in answers]
    return data


def get_answers_dictionary(answers, save_path, idx_to_answer_save_name, answer_to_idx_save_name):

    answer_to_idx = {}

    i = 0
    for k, v in answers.items():
        answer_to_idx[k] = i
        i += 1

    idx_to_answer = {v: k for k, v in answer_to_idx.items()}

    save_file(data=idx_to_answer, path=save_path, file_name=idx_to_answer_save_name, file_type='pickle')
    save_file(data=answer_to_idx, path=save_path, file_name=answer_to_idx_save_name, file_type='pickle')

    return idx_to_answer, answer_to_idx


