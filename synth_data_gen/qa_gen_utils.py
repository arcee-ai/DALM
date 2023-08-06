def generate_question(passage, question_generator):
    questions = question_generator(passage, max_length=50, num_return_sequences=1)
    return questions[0]['generated_text']