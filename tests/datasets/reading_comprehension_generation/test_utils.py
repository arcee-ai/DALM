from dalm.datasets.reading_comprehension_generation.utils import _raw_question_and_answer_extractor, question_and_answer_extractor
import pdb

def test_question_and_answer_extractor():
    chat_completions = question_and_answer_extractor(
        whole_text=""" 
1. QUESTION: Can you summarize the purpose of population imaging studies and how they contribute to preventing or treating disease?
ANSWER: Population imaging studies generate data for developing and implementing personalized health strategies to prevent or more effectively treat disease. These studies acquire imaging for pre-symptomatic populations to discover alterations due to impending disease and identify individuals at risk, which enables early intervention.

2. QUESTION: How does the UK Biobank study stand out in terms of size and availability of expert annotation?
ANSWER: The UK Biobank study stands out for its sheer size, careful implementation, and availability of top quality expert annotation. The resulting massive imaging datasets targeting around 100,000 subjects have posed new challenges requiring automatic image analysis, and this study has put published approaches for cardiac image quantification to the test.

3. QUESTION: What does the proposed cardiac magnetic resonance (CMR) image analysis pipeline do, and how does it differ from previous published approaches for cardiac image quantification?
ANSWER: The proposed CMR image analysis pipeline performs end-to-end image analytics from multi-view cine CMR images all the way to anatomical and functional bi-ventricular quantification without manual user interactions. It provides fully automated extraction of global and regional reference ranges of all key functional cardiovascular indexes from both left and right cardiac ventricles for a population of 20,000 subjects imaged at 50 time frames per subject, for a total of one million CMR volumes. This is the first published attempt to fully automate the extraction of global and regional reference ranges of all key functional cardiovascular indexes for such a large population.

4. QUESTION: How does the proposed CMR analytics pipeline compare in terms of segmentation accuracy with respect to human experts, and what are the results of its validation against manual expert readings on a reference cohort of 4620 subjects?
ANSWER: The proposed pipeline shows broad significant agreement between the manually obtained reference indexes and those automatically computed via the framework. Around 80.67% of subjects were processed with mean contour distance of less than 1 pixel, and around 17.50% with mean contour distance between 1 and 2 pixels. The comparison with a recently published approach reporting on UKB data, which is based on deep learning, shows similar performance in terms of segmentation accuracy with respect to human experts.
        """,
        context=""" 
Population imaging studies generate data for developing and implementing personalised health strategies to prevent, or more effectively treat disease. Large prospective epidemiological studies acquire imaging for pre-symptomatic populations. These studies enable the early discovery of alterations due to impending disease, and enable early identification of individuals at risk. Such studies pose new challenges requiring automatic image analysis. To date, few large-scale population-level cardiac imaging studies have been conducted. One such study stands out for its sheer size, careful implementation, and availability of top quality expert annotation; the UK Biobank (UKB). The resulting massive imaging datasets (targeting ca. 100,000 subjects) has put published approaches for cardiac image quantification to the test. In this paper, we present and evaluate a cardiac magnetic resonance (CMR) image analysis pipeline that properly scales up and can provide a fully automatic analysis of the UKB CMR study. Without manual user interactions, our pipeline performs end-to-end image analytics from multi-view cine CMR images all the way to anatomical and functional bi-ventricular quantification. All this, while maintaining relevant quality controls of the CMR input images, and resulting image segmentations. To the best of our knowledge, this is the first published attempt to fully automate the extraction of global and regional reference ranges of all key functional cardiovascular indexes, from both left and right cardiac ventricles, for a population of 20,000 subjects imaged at 50 time frames per subject, for a total of one million CMR volumes. In addition, our pipeline provides 3D anatomical bi-ventricular models of the heart. These models enable the extraction of detailed information of the morphodynamics of the two ventricles for subsequent association to genetic, omics, lifestyle habits, exposure information, and other information provided in population imaging studies. We validated our proposed CMR analytics pipeline against manual expert readings on a reference cohort of 4620 subjects with contour delineations and corresponding clinical indexes. Our results show broad significant agreement between the manually obtained reference indexes, and those automatically computed via our framework. 80.67% of subjects were processed with mean contour distance of less than 1 pixel, and 17.50% with mean contour distance between 1 and 2 pixels. Finally, we compare our pipeline with a recently published approach reporting on UKB data, and based on deep learning. Our comparison shows similar performance in terms of segmentation accuracy with respect to human experts.
        """,
    )
    print(chat_completions) 

    # The first chat completion item should be a user prompt, and it should start with "Based on the following text:"
    assert chat_completions[0]["content"].startswith("Based on the following text:")
    assert chat_completions[0]["role"] == "user"

    # We should have 9 chat completion items for this input
    assert len(chat_completions) == 9

    # Now it should alternate between user and assistant roles.
    # Odd numbers should be user prompts, and even numbers should be assistant responses.
    for i, chat_completion in enumerate(chat_completions):
        
        # Skip the first row, which was already verified as having a user role above.
        if i == 0:
            continue
        
        if i % 2 == 0:
            assert chat_completion["role"] == "assistant"
        else:
            assert chat_completion["role"] == "user"





def test_raw_question_and_answer_extractor():

    inputs = [
        {
            "whole_text": """
                            QUESTION: What is the focus?
                            ANSWER: The focus is health strategies.

                            QUESTION: What is unique about the UK?
                            ANSWER: The UK Biobank (UKB).

                            QUESTION: What is the focus of the proposed?
                            ANSWER: The focus of the proposed CMR image imaging studies.

                            QUESTION: How was the proposed CMR analytics?
                            ANSWER: The proposed CMR analytics pipeline was validated.""",
            "expected_output": [
                {
                    "question": "What is the focus?",
                    "answer": "The focus is health strategies."
                },
                {
                    "question": "What is unique about the UK?",
                    "answer": "The UK Biobank (UKB)."
                },
                {
                    "question": "What is the focus of the proposed?",
                    "answer": "The focus of the proposed CMR image imaging studies."
                },
                {
                    "question": "How was the proposed CMR analytics?",
                    "answer": "The proposed CMR analytics pipeline was validated."
                }
            ]
        },        
        {
            "whole_text": """1. QUESTION: What are thoracic diseases?
                                ANSWER: Thoracic diseases refer to health problems.
                                
                                2. QUESTION: How is chest X-ray currently?
                                ANSWER: Chest X-ray is currently one.
                                
                                3. QUESTION: Why is reading chest X-ray images?
                                ANSWER: Reading chest X-ray images.
                                
                                4. QUESTION: What is the proposed solution?
                                ANSWER: To make a deep architecture.""",
            "expected_output": [
                {
                    "question": "What are thoracic diseases?",
                    "answer": "Thoracic diseases refer to health problems."
                },
                {
                    "question": "How is chest X-ray currently?",
                    "answer": "Chest X-ray is currently one."
                },
                {
                    "question": "Why is reading chest X-ray images?",
                    "answer": "Reading chest X-ray images."
                },
                {
                    "question": "What is the proposed solution?",
                    "answer": "To make a deep architecture."
                }
            ]
        },
        {
            "whole_text": """1. [QUESTION:] What are thoracic diseases?
                                [ANSWER:] Thoracic diseases refer to health problems.
                                
                                2. [QUESTION:] How is chest X-ray currently?
                                [ANSWER:] Chest X-ray is currently one.
                                
                                3. [QUESTION:] Why is reading chest X-ray images?
                                [ANSWER:] Reading chest X-ray images .
                                
                                4. [QUESTION:] What is the proposed solution?
                                [ANSWER:] To make a deep architecture.""",
            "expected_output": [
                {
                    "question": "What are thoracic diseases?",
                    "answer": "Thoracic diseases refer to health problems."
                },
                {
                    "question": "How is chest X-ray currently?",
                    "answer": "Chest X-ray is currently one."
                },
                {
                    "question": "Why is reading chest X-ray images?",
                    "answer": "Reading chest X-ray images ."
                },
                {
                    "question": "What is the proposed solution?",
                    "answer": "To make a deep architecture."
                }
            ]
        },
        {
            "whole_text": """ 1. [QUESTION: Complete-the-sentence Q&A] What are thoracic diseases?
                                ANSWER: Thoracic diseases refer to health problems.
                                
                                2. [QUESTION: True/false Q&A] How is chest X-ray currently?
                                ANSWER: Chest X-ray is currently one.""",
            "expected_output": [
                {
                    "question": "What are thoracic diseases?",
                    "answer": "Thoracic diseases refer to health problems."
                },
                {
                    "question": "How is chest X-ray currently?",
                    "answer": "Chest X-ray is currently one."
                }
            ]
        },
        {
            "whole_text": """1. Question (type: normal q&a): What are thoracic diseases?
                                Answer: Thoracic diseases refer to health problems.
                                
                               2. Question (type: complete-the-sentence): How is chest X-ray currently?
                                Answer: Chest X-ray is currently one.
                                """,
            "expected_output": [
                {
                    "question": "(type: normal q&a): What are thoracic diseases?",
                    "answer": "Thoracic diseases refer to health problems."
                },
                {
                    "question": "(type: complete-the-sentence): How is chest X-ray currently?",
                    "answer": "Chest X-ray is currently one."
                }
            ]
        },


    ]

    for input in inputs:
        result_qa_pairs = _raw_question_and_answer_extractor(whole_text=input["whole_text"])
        expected_qa_pairs = input["expected_output"]
        for result, expected in zip(result_qa_pairs, expected_qa_pairs):
            result_question = result["question"].strip().lower()
            expected_question = expected["question"].strip().lower()
            result_answer = result["answer"].strip().lower()
            expected_answer = expected["answer"].strip().lower()
            assert result_question == expected_question, f"result_question: {result_question} != expected_question: {expected_question}"
            assert result_answer == expected_answer, f"result_answer: {result_answer} != expected_answer: {expected_answer}"
        


