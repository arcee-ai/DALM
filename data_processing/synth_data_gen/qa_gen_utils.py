def generate_question(passage, question_generator):
    question_list = []
    generated_texts = question_generator(passage, max_length=50, num_return_sequences=1)
    for text in generated_texts:
        question_list.append(text['generated_text'])
    return question_list