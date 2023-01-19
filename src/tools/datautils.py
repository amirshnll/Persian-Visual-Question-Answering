import cv2
import matplotlib.pyplot as plt


def visualize_data(data):

    img = cv2.imread(filename=data['image_path'])
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)

    plt.imshow(X=img)
    plt.show()
    print(f'EN Q: {data["question"]}')
    print(f'FA Q {data["que_freq"]}: {data["fa_question"]}')
    print(f'EN A: {data["multiple_choice_answer"]}')
    print(f'FA A {data["ans_freq"]}: {data["fa_answer"]}')
