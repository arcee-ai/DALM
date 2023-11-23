# Modified version of code from https://github.com/microsoft/LMOps/blob/main/adaptllm/utils/read.py

# ruff: noqa: E501

import argparse
import copy
import json
import logging
import os
import random
import re
import typing
from typing import Any, Dict, Iterator, List, Optional, Tuple

import datasets
import numpy as np
import sentencepiece as spm  # type: ignore[import-untyped]
from pysbd import Segmenter  # type: ignore[import-untyped]
from tqdm.contrib.concurrent import process_map

from dalm.datasets.reading_comprehension_generation.utils import (
    create_domain_tokenizer,
    create_domain_tokenizer_from_files,
    input_generator,
)

logger = logging.getLogger(__name__)

TYPES = ["nli", "common_reason", "paraphrase", "word2text", "summarize", "text_completion"]


def remove_double_space(string: str) -> str:
    return re.sub("[ ]{2,}", " ", string)


class App:
    def __init__(self) -> None:
        self.cls_dic: Dict[str, Any] = {}

    @typing.no_type_check
    def add(self, key: str):
        def adder(cls: Any):
            self.cls_dic[key] = cls
            return cls

        return adder


type_map = App()


def chatml_format(question: str, answer: str | None = None) -> List[Dict[str, str]]:
    result = [{"role": "user", "content": question}]
    if answer is not None:
        result.append({"role": "assistant", "content": answer})
    return result


@type_map.add("basetype")
class BaseType(object):
    def __init__(self) -> None:
        self.max_subcategory_num = 2  # limit the number of examples per subcategory
        self.max_seq_len = 2048
        self.mine_regex: Dict[str, Any] = {}

    def collect_mined(self, tup: List[str], class_name: str) -> Dict[str, Any]:
        raise NotImplementedError

    def get_all_templates(self, entry: Dict[str, Any], random_seed: int) -> List[Tuple[str, str]] | List[Tuple[str]]:
        raise NotImplementedError

    @typing.no_type_check
    def get_template(self, entry: Dict[str, Any], random_seed: int) -> Tuple[str] | Tuple[str, str]:
        """
        random sample a template for each entry
        """
        random.seed(random_seed)  # fix random seed for reproduction
        template = random.choice(self.get_all_templates(entry, random_seed))
        return template

    # TODO: refactor
    def fill_in_the_template(
        self, template: Tuple[str] | Tuple[str, str], kw_dic: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Account for:
        1. length 1 template and no qa_demos
        2. length 1 template and qa_demos
        3. length 2 template and no qa_demos
        4. length 2 template and qa_demos

        """
        qa_demos = kw_dic.get("qa_demos", [])

        if "qa_demos" in kw_dic.keys():
            qa_demos = kw_dic["qa_demos"]

        question = template[0].format(**kw_dic)

        if len(template) == 1 and len(qa_demos) > 1:
            qa_demos[0]["content"] = question + qa_demos[0]["content"]
            return qa_demos
        elif len(template) == 1 and len(qa_demos) == 0:
            return chatml_format(question)
        elif len(template) == 2:
            answer = template[1].format(**kw_dic)

            result = chatml_format(question, answer)
            if qa_demos is not None:
                result = qa_demos + result

            return result
        else:
            raise ValueError("template length must be 1 or 2")

    def truncate_sentence(self, text: str, max_len: int) -> List[str]:
        tokenized_example = self.ori_spm.encode(text)
        example_length = len(tokenized_example)

        if example_length > max_len:
            truncated_text_list = []
            chunked_list = [tokenized_example[i : i + max_len] for i in range(0, len(tokenized_example), max_len)]
            # input_ids = tokenized_example[:max_len]
            # truncated_text = self.ori_spm.decode(input_ids)
            for truncated_tokens in chunked_list:
                truncated_text_list.append(self.ori_spm.decode(truncated_tokens))
            return truncated_text_list
        else:
            return [text]

    def init_spm(self, ori_spm: spm.SentencePieceProcessor, domain_spm: spm.SentencePieceProcessor) -> None:
        self.ori_spm = ori_spm
        self.domain_spm = domain_spm

        ori_tokens = set([self.ori_spm.id_to_piece(i) for i in range(len(self.ori_spm))])
        domain_tokens = set([self.domain_spm.id_to_piece(i) for i in range(len(self.domain_spm))])
        specific_tokens_set = domain_tokens - (ori_tokens & domain_tokens)
        specific_tokens = [token for token in list(specific_tokens_set) if (token.startswith("▁") and len(token) > 10)]
        self.specific_token_set = set(specific_tokens)

    def compile_regex(self) -> None:
        """
        Does nothing more than compile regexes
        """
        self.regex_dic = {}
        for class_name, pattern in self.mine_regex.items():
            self.regex_dic[class_name] = re.compile(pattern, re.IGNORECASE)

    def mine(self, text: str, **kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        mined_dic: Dict[str, Any] = {}
        mined_num = 0
        for class_name, regex in self.regex_dic.items():
            mined_dic[class_name] = []
            x = regex.findall(text)
            if len(x) > 0:
                for tup in x:
                    collected = self.collect_mined(tup, class_name)
                    mined_dic[class_name].append(collected)
            mined_num += len(mined_dic[class_name])
        return mined_dic, mined_num


@type_map.add("nli")
class nli(BaseType):
    def __init__(self) -> None:
        super().__init__()
        # init regex
        self.mine_regex = {
            "Entail": r"([.!?]+[\s]+)([^.!?\n,]{50,}[.!?]+)([\s]+)(Yes|Therefore|Thus|Accordingly|Hence|For this reason)([\s]*,[\s]+)([^.!?\n,]{50,}[.!?]+)([\s]+)",
            "Contradict": r"([.!?]+[\s]+)([^.!?\n,]{50,}[.!?]+)([\s]+)(No|However|But|On the contrary|In contrast|Whereas)([\s]*,[\s]+)([^.!?\n,]{50,}[.!?]+)([\s]+)",
            "Neutral": r"([.!?]+[\s]+)([^.!?\n,]{50,}[.!?]+)([\s]+)(Maybe|Also|Furthermore|Secondly|Additionally|Moreover|In addition)([\s]*,[\s]+)([^.!?\n,]{50,}[.!?]+)([\s]+)",
        }
        self.compile_regex()

    def collect_mined(self, tup: List[str], class_name: str) -> Dict[str, Any]:
        dic = {
            "label": class_name,
            "verbalizer": tup[3],
            "premise": tup[1],
            "hypothesis": tup[-2],
        }
        return dic

    def get_all_templates(self, entry: Dict[str, Any], random_seed: int) -> List[Tuple[str, str]]:
        np.random.seed(random_seed)
        type = np.random.choice(["generate", "classify"], p=[0.2, 0.8])
        if type == "classify":
            return [
                # Basic Templates
                ('{premise}\nBased on the sentence above can we infer that "{hypothesis}"?', "{answer}"),
                (
                    "{premise}\nBased on this sentence can we infer that the following sentence is true?\n{hypothesis}\nAnswer:",
                    "{answer}",
                ),
                ("{premise}\nCan we draw the following hypothesis?\n{hypothesis}\n{options_}", "{answer}"),
                (
                    "{premise}\nDoes this next sentence follow, given the preceding text?\n{hypothesis}\nAnswer:",
                    "{answer}",
                ),
                (
                    "Can we draw the following hypothesis from the context?\nContext: {premise}\nHypothesis: {hypothesis}\nAnswer:",
                    "{answer}",
                ),
                (
                    "{hypothesis}\nDetermine if the sentence is true based on the text below:\n{premise}\nAnswer:",
                    "{answer}",
                ),
                ("Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis?", "{answer}"),
                (
                    "Premise: {premise}\nHypothesis: {hypothesis}\nIs the hypothesis entailed by the premise?",
                    "{answer}",
                ),
                (
                    "Here is a premise:\n{premise}\nHere is a hypothesis:\n{hypothesis}\nIs it possible to infer that if the premise is true, then so is the hypothesis?",
                    "{answer}",
                ),
                (
                    "Sentence 1: {premise}\nSentence 2: {hypothesis}\nIs this second sentence entailed by the first sentence?\n{options_}",
                    "{answer}",
                ),
                ('Based on the premise "{premise}", can we infer the hypothesis "{hypothesis}" is true?', "{answer}"),
                (
                    'Premise:\n"{premise}" Based on this premise, is the hypothesis "{hypothesis}" true?\n{options_}',
                    "{answer}",
                ),
                ('If {premise}, can we infer that "{hypothesis}"?', "{answer}"),
                ('{premise}\nDoes it follow that "{hypothesis}"?\n{options_}', "{answer}"),
                ('Question: If "{premise}", does this mean that "{hypothesis}"?\nAnswer:', "{answer}"),
                ('If "{premise}", can we infer "{hypothesis}"?', "{answer}"),
                ('If "{premise}", does it logically follow that "{hypothesis}"?', "{answer}"),
                ('Based on the sentence "{premise}", is the sentence "{hypothesis}" a true sentence?', "{answer}"),
                (
                    "Premise: {premise}\nHypothesis: {hypothesis}\nCan we infer that the hypothesis is true if the premise is true?",
                    "{answer}",
                ),
                (
                    'Here is a premise: "{premise}"\nHere is a hypothesis: "{hypothesis}"\nDoes the premise tell us whether the hypothesis is true?',
                    "{answer}",
                ),
                ('Is the premise "{premise}" true if "{hypothesis}"?\n{options_}', "{answer}"),
                ('If "{premise}", can we infer that "{hypothesis}"?\n{options_}', "{answer}"),
                ('If "{premise}", is "{hypothesis}" correct?', "{answer}"),
                ('Let\'s say that "{premise}"\nCan we now say that "{hypothesis}"?', "{answer}"),
                ('Does "{hypothesis}" appear to be an accurate statement based on "{premise}"?', "{answer}"),
                ('Is it possible to draw the statement that "{hypothesis}" if "{premise}"?', "{answer}"),
                ('Is "{hypothesis}" true if "{premise}"?\n{options_}', "{answer}"),
                (
                    'Sentence 1: "{premise}"\nSentence 2: "{hypothesis}"\nIs sentence 2 true, based on sentence 1?',
                    "{answer}",
                ),
                # fill-in-the-blank:
                (
                    'Sentence 1: "{premise}"\nSentence 2: "{hypothesis}"\nWhich word is the best to connect them? Therefore, However, or Moreover?',
                    "{connect_answer}",
                ),
                (
                    "Choose the most suitable word to link the following sentences:\n1. {premise}\n2. {hypothesis}\nOptions:\n- Therefore\n- However\n- Moreover",
                    "{connect_answer}",
                ),
                (
                    'Connect the following sentence: {premise}\nChoose the appropriate word to link it with: "{hypothesis}"\nOptions: Therefore, However, Moreover',
                    "{connect_answer}",
                ),
                (
                    'Given the sentence: {premise}\nChoose the appropriate word from the options (Therefore, However, Moreover) to connect it with: "{hypothesis}"\nWord:',
                    "{connect_answer}",
                ),
                (
                    'Connect the sentence: {premise}\nFrom the choices (Therefore, However, Moreover), select the word that best links it to: "{hypothesis}"\nAnswer:',
                    "{connect_answer}",
                ),
                # relation classification
                (
                    'Assess the relationship between Sentence 1: "{premise}"\nSentence 2: "{hypothesis}"\nIs it characterized as Entailment, Neutral, or Contradictory?',
                    "{relation_answer}",
                ),
                (
                    'Given Sentence 1: "{premise}"\nSentence 2: "{hypothesis}"\nHow would you describe the relationship between these two sentences? Entailment, Neutral, or Contradictory?',
                    "{relation_answer}",
                ),
                (
                    'Considering Sentence 1: "{premise}"\nSentence 2: "{hypothesis}"\nHow do you perceive the connection between these two sentences in terms of their relationship?',
                    "{relation_answer}",
                ),
                (
                    'Assess the relationship between Sentence 1: "{premise}"\nSentence 2: "{hypothesis}"\nWould you categorize their connection as Entailment, Neutral, or Contradictory?',
                    "{relation_answer}",
                ),
            ]
        elif type == "generate":
            if entry["label"] == "Entail":
                return [
                    ("Complete the following sentence\n{premise} Accordingly,", "{hypothesis}"),
                    ("{premise} Therefore:", "{hypothesis}"),
                    ("{premise} Thus?", "{hypothesis}"),
                    (
                        'Based on the statement "{premise}", provide a continuation using the word "Hence" to express the following idea.\nContinuation:',
                        "{hypothesis}",
                    ),
                    (
                        'Question: Complete the following statement using the word "Therefore" in relation to "{premise}"\nAnswer:',
                        "{hypothesis}",
                    ),
                    ("{premise} {verbalizer}?", "{hypothesis}"),
                    ("{premise} {verbalizer}:", "{hypothesis}"),
                    # more variations
                    (
                        "{premise}\nProduce a sentence that encompasses the concept from the above statement. Sentence:",
                        "{hypothesis}",
                    ),
                    (
                        '"{premise}" Generate a sentence that follows from the notion presented in the previous statement.',
                        "{hypothesis}",
                    ),
                    (
                        "{premise}\nCraft a sentence that develops the idea put forth in the preceding statement.",
                        "{hypothesis}",
                    ),
                    (
                        "{premise}\nCreate a sentence that is a logical extension of the idea in the previous statement.\nAnswer:",
                        "{hypothesis}",
                    ),
                    (
                        '"{premise}" Formulate a sentence that is consistent with the concept presented in the prior statement.',
                        "{hypothesis}",
                    ),
                    (
                        "{premise}\nDevelop a sentence that builds upon the thought conveyed in the above statement.",
                        "{hypothesis}",
                    ),
                ]
            elif entry["label"] == "Neutral":
                return [
                    ("Complete the following sentence: {premise} {verbalizer},", "{hypothesis}"),
                    ("Complete the following sentence\n{premise} {verbalizer}:", "{hypothesis}"),
                    ("{premise} {verbalizer}?", "{hypothesis}"),
                    (
                        'Based on the statement {premise}, provide a continuation using the word "{verbalizer}" to express the following idea.\nContinuation:',
                        "{hypothesis}",
                    ),
                    (
                        'Question: Complete the following statement using the word "{verbalizer}" in relation to "{premise}"\nAnswer:',
                        "{hypothesis}",
                    ),
                ]
            elif entry["label"] == "Contradict":
                return [
                    ("Complete the following sentence: {premise} On the contrary,", "{hypothesis}"),
                    ("{premise} But,\nWhat is a completion for it?", "{hypothesis}"),
                    ("Complete the following sentence\n{premise} However?", "{hypothesis}"),
                    ("Sentence: {premise} {verbalizer},\nHow do you finish this sentence?", "{hypothesis}"),
                    ("{premise} {verbalizer}:", "{hypothesis}"),
                    (
                        'Based on the statement {premise}, provide a continuation using "In contrast" to express the following idea.',
                        "{hypothesis}",
                    ),
                    (
                        'Complete the following statement using the word "But" in relation to "{premise}".',
                        "{hypothesis}",
                    ),
                ]
            else:
                raise ValueError("label must be Entail, Neutral or Contradict")
        else:
            raise ValueError("type must be generate or classify")

    def format_single_demo(self, entry: Dict[str, Any], random_seed: int) -> List[Dict[str, str]]:
        kw_dic = {}
        kw_dic["premise"] = entry["premise"]
        hypothesis = entry["hypothesis"]
        kw_dic["hypothesis"] = hypothesis[0].upper() + hypothesis[1:]
        kw_dic["options_"] = "- Yes\n- No\n- Maybe"

        kw_dic["verbalizer"] = entry["verbalizer"]
        if entry["label"] == "Entail":
            kw_dic["answer"] = "Yes"
            kw_dic["connect_answer"] = "Therefore"
            kw_dic["relation_answer"] = "Entailment"
        elif entry["label"] == "Contradict":
            kw_dic["answer"] = "No"
            kw_dic["connect_answer"] = "However"
            kw_dic["relation_answer"] = "Contradictory"
        elif entry["label"] == "Neutral":
            kw_dic["answer"] = "Maybe"
            kw_dic["connect_answer"] = "Moreover"
            kw_dic["relation_answer"] = "Neutral"

        template = self.get_template(entry, random_seed)
        return self.fill_in_the_template(template, kw_dic)


@type_map.add("common_reason")
class common_reason(BaseType):
    def __init__(self) -> None:
        super().__init__()
        self.mine_regex = {
            "Cause-effect": r"([.!?]+[\s]+)([^.!?\n,]{50,}[.!?]+)([\s]+)(Thus|Therefore|Accordingly|Hence|For this reason)([\s]*,[\s]+)([^.!?\n,]{50,}[.!?]+)([\s]+)",
            "Effect-cause": r"([.!?]+[\s]+)([^.!?;\n,]{50,}[.!?]+)([\s]+)(due to|on account of|owing to)([\s]+)([^.!?;\n,]{50,}[.!?]+)([\s]+)",
        }
        self.compile_regex()

    def collect_mined(self, tup: List[str], class_name: str) -> Dict[str, Any]:
        dic = {
            "relation": class_name,
            "verbalizer": tup[3],
            "sentence1": tup[1],
            "sentence2": tup[-2],
        }
        return dic

    def get_all_templates(self, entry: Dict[str, Any], random_seed: int) -> List[Tuple[str, str]]:
        if entry["relation"] == "Cause-effect":
            return [
                # Basic templates
                ('Question: What is the effect of "{cause}"? Answer:', "{effect}"),
                ("Here is a premise: {cause}\nWhat is the effect?", "{effect}"),
                ('Q: What is the result of "{cause}"? A:', "{effect}"),
                ('What is a plausible effect of "{cause}"?', "{effect}"),
                ('Based on "{cause}", what is the result?', "{effect}"),
                ("{cause}\nEffect:", "{effect}"),
                ("What is the result of the following sentence?\n{cause}\nResult:", "{effect}"),
                ('Q: What happens after "{cause}"? A:', "{effect}"),
                ("{cause}\nWhat happens next?", "{effect}"),
                # More varaiations
                ("Considering the cause: {cause}\nWhat could be the resulting effect?", "{effect}"),
                ("Given that: {cause}\nWhat do you anticipate as the outcome?", "{effect}"),
                ('What could stem from "{cause}"?', "{effect}"),
                ("Explore the consequences of: {cause}\nAnswer:", "{effect}"),
                ('What might follow from "{cause}"?', "{effect}"),
                ('Based on the cause: "{cause}"\nWhat is likely to be the effect?', "{effect}"),
                ('If "{cause}" occurs, what is the probable effect?', "{effect}"),
                ('Imagine "{cause}" taking place; what would be the resultant effect?', "{effect}"),
                ("Given the scenario: {cause}\nWhat effect could be expected?", "{effect}"),
                ('Examine the potential outcomes of "{cause}"\nOutcome:', "{effect}"),
                ("Anticipating the result of: {cause}\nWhat could be the effect?", "{effect}"),
                ('What is the expected effect of "{cause}"?', "{effect}"),
                ("Considering the event: {cause}\nWhat could be an outcome?", "{effect}"),
                ('If "{cause}" happens, what could be the subsequent effect?', "{effect}"),
                ('Explore the aftermath of: "{cause}"\nWhat could be the effect?', "{effect}"),
            ]
        elif entry["relation"] == "Effect-cause":
            return [
                # Basic templates
                ('Q: "{effect}" What is the cause? A:', "{cause}"),
                ("Here is a result: {effect}\nWhat is the cause?", "{cause}"),
                ('What is the reason of "{effect}"?', "{cause}"),
                ('What is a plausible reason for "{effect}"?', "{cause}"),
                ('what is the cause of "{effect}"?', "{cause}"),
                ("{effect}\nCause:", "{cause}"),
                ("Question: What is the reason of the following sentence?\n{effect}\nAnswer:", "{cause}"),
                ('What happens before "{effect}"?', "{cause}"),
                ("{effect}\nWhat happens before?", "{cause}"),
                # More variations:
                ("Given the outcome: {effect}\nWhat could have led to this result?", "{cause}"),
                ('Uncover the cause behind: "{effect}".', "{cause}"),
                ("What might be responsible for {effect}?", "{cause}"),
                ("Identify a probable cause for: {effect}\nCause:", "{cause}"),
                ('What event or circumstance could explain "{effect}"?', "{cause}"),
                ("When observing: {effect}\nWhat should we consider as the cause?", "{cause}"),
                ("What events or factors contributed to: {effect}?", "{cause}"),
                ('Considering the effect: "{effect}"\nWhat could be the underlying cause?', "{cause}"),
                ('Before "{effect}" occurred, what factor might have caused it?', "{cause}"),
                ('What do you think led to the occurrence of: "{effect}"?', "{cause}"),
                ("Analyze the occurrence of: {effect}\nWhat could be identified as the cause?", "{cause}"),
                ("Given that: {effect}\nWhat was the triggering cause?", "{cause}"),
                ('Explore the background of: "{effect}"\nWhat could have instigated it?', "{cause}"),
                ("What played a role in bringing about: {effect}?", "{cause}"),
                (
                    'Delve into the circumstances behind "{effect}"\nWhat could be the originating cause? Answer:',
                    "{cause}",
                ),
                ("Complete the following sentence\n{effect} because of", "{cause}"),
                ("Your task is to complete the following sentence: {effect} due to", "{cause}"),
                ("{effect} owing to\nHow would you complete it:", "{cause}"),
                (
                    'Based on the statement {effect}, provide a continuation using "{verbalizer}" to express the following idea.\nContinuation:',
                    "{cause}",
                ),
                (
                    'Question: Complete the following statement using "{verbalizer}" in relation to "{effect}".',
                    "{cause}",
                ),
                ("Answer the question...{effect} {verbalizer}?", "{cause}"),
                ("{effect} {verbalizer}:", "{cause}"),
            ]
        else:
            raise ValueError("relation must be Cause-effect or Effect-cause")

    def format_single_demo(self, entry: Dict[str, Any], random_seed: int) -> List[Dict[str, str]]:
        kw_dic = {}
        kw_dic["verbalizer"] = entry["verbalizer"]
        if entry["relation"] == "Cause-effect":
            kw_dic["cause"] = entry["sentence1"]
            kw_dic["effect"] = entry["sentence2"][0].upper() + entry["sentence2"][1:]
        elif entry["relation"] == "Effect-cause":
            kw_dic["cause"] = entry["sentence2"][0].upper() + entry["sentence2"][1:]
            kw_dic["effect"] = entry["sentence1"]
        elif entry["relation"] == "Explanantion":
            kw_dic["sentence1"] = entry["sentence1"]
            kw_dic["sentence2"] = entry["sentence2"][0].upper() + entry["sentence2"][1:]

        template = self.get_template(entry, random_seed)
        return self.fill_in_the_template(template, kw_dic)


@type_map.add("paraphrase")
class paraphrase(BaseType):
    def __init__(self) -> None:
        super().__init__()
        self.mine_regex = {
            "Similar": r"([.!?]+[\s]+)([^.!?\n]{50,}[.!?]+)([\s]+)(In other words|In other word|Namely|That is to say|i.e.|Scilicet|Similarly|Equally)([\s]*,[\s]+)([^.!?\n]{50,}[.!?]+)([\s]+)",
            "Different": r"([.!?]+[\s]+)([^.!?\n]{50,}[.!?]+)([\s]+)(No|However|But|On the contrary|In contrast|Whereas)([\s]*,[\s]+)([^.!?\n]{50,}[.!?]+)([\s]+)",
        }

        self.compile_regex()

    def collect_mined(self, tup: List[str], class_name: str) -> Dict[str, Any]:
        dic = {
            "label": class_name,
            "verbalizer": tup[3],
            "sentence1": tup[1],
            "sentence2": tup[-2],
        }
        return dic

    def get_all_templates(self, entry: Dict[str, Any], random_seed: int) -> List[Tuple[str, str]]:
        if entry["label"] == "Different":
            return [
                (
                    '"{sentence1}" Generate a sentence that expresses a contrasting idea to the previous statement.',
                    "{sentence2}",
                ),
                ('Can you create a sentence that contradicts the meaning of "{sentence1}"?', "{sentence2}"),
                (
                    'Given the sentence "{sentence1}", can you come up with a statement that contradicts its meaning?',
                    "{sentence2}",
                ),
                (
                    'Here is a sentence: "{sentence1}". Now, provide a sentence that contradicts its meaning.',
                    "{sentence2}",
                ),
                (
                    'Your challenge is to create a sentence that expresses the opposite of "{sentence1}". Answer:',
                    "{sentence2}",
                ),
                ('Contradict the meaning of the sentence "{sentence1}" by crafting another sentence.', "{sentence2}"),
                ('Compose a sentence that contradicts the idea conveyed in "{sentence1}".', "{sentence2}"),
                (
                    'Can you generate a sentence that has a conflicting meaning compared to "{sentence1}"?',
                    "{sentence2}",
                ),
                (
                    'In opposition to the sentence "{sentence1}", create a sentence with a contradictory meaning.',
                    "{sentence2}",
                ),
                (
                    'Your task is to provide a sentence that negates or contradicts the message of "{sentence1}".',
                    "{sentence2}",
                ),
                (
                    'Given the sentence "{sentence1}", come up with a different sentence that contradicts its meaning?',
                    "{sentence2}",
                ),
                ('Craft a sentence that goes against the meaning of the sentence "{sentence1}".', "{sentence2}"),
            ]
        elif entry["label"] == "Similar":
            return [
                ("Complete the following sentence: {sentence1} Namely,", "{sentence2}"),
                ("{sentence1} In other words\nProvide the missing portion of the above sentence:", "{sentence2}"),
                ("Q: {sentence1} That is to say?", "{sentence2}"),
                (
                    'Question: Complete the following statement using "{verbalizer}" in relation to "{sentence1}"\nAnswer:',
                    "{sentence2}",
                ),
                ("Question: {sentence1} {verbalizer}?", "{sentence2}"),
                ("{sentence1} {verbalizer},\nHow do you finish this sentence?", "{sentence2}"),
                ("Extend the thought in this sentence: {sentence1} To elaborate further:", "{sentence2}"),
                (
                    'Build upon the statement {sentence1} by utilizing "{verbalizer}" to express the following concept.',
                    "{sentence2}",
                ),
                (
                    '"{sentence1}" Generate a sentence that expresses a further elaboration to the previous statement.',
                    "{sentence2}",
                ),
                ('"{sentence1}" Expand on the previous statement:', "{sentence2}"),
                ("{sentence1}\nProvide an explanatory sentence:", "{sentence2}"),
            ]
        else:
            raise ValueError("label must be Similar or Different")

    def format_single_demo(self, entry: Dict[str, Any], random_seed: int) -> List[Dict[str, str]]:
        kw_dic = {}
        kw_dic["verbalizer"] = entry["verbalizer"]
        kw_dic["sentence1"] = entry["sentence1"]
        kw_dic["sentence2"] = entry["sentence2"][0].upper() + entry["sentence2"][1:]

        template = self.get_template(entry, random_seed)
        return self.fill_in_the_template(template, kw_dic)


@type_map.add("word2text")
class word2text(BaseType):
    def __init__(self) -> None:
        super().__init__()
        self.mine_regex = {
            "definition": r"([\s]+)([^.!?,;\s\"]{10,})([\s]+)(is defined as|\'s definition is)([\s]+)([^.!?\n]{20,}[.!?]+)([\s]+)",
            "topic": r"([.!?]+[\s]+)([^.!?,;\n]{20,})([\s]+)(was about|talks about|is about|\'s topic is)([\s]+)([^.!?\n]{20,}[.!?]+)([\s]+)",
        }
        # `topic` is defined as a summaization task in our paper,
        # here we categorize it to word2text for simple code implementation

        self.compile_regex()

        self.min_kw_num = 3  # requires at least 3 domain-specific keywords,
        self.max_sent_len = 100  # with fewer than 100 sent tokens.
        self.max_collect_sent = 2  # early break when find enough task examples.

    def collect_mined(self, tup: List[str], class_name: str) -> Dict[str, Any]:
        if class_name == "definition":
            dic = {
                "relation": class_name,
                "verbalizer": tup[3],
                "word": tup[1],
                "definition": tup[-2],
            }
        elif class_name == "topic":
            dic = {
                "relation": class_name,
                "verbalizer": tup[3],
                "sentence": tup[1],
                "topic": tup[-2],
            }
        return dic

    @typing.no_type_check
    def mine(self, text: str, sents: List[str], **kwargs) -> Tuple[Dict[str, Any], int]:
        def mine_regex(text):
            mined_dic = {}
            mined_num = 0
            for class_name, regex in self.regex_dic.items():
                mined_dic[class_name] = []
                x = regex.findall(text)
                if len(x) > 0:
                    for tup in x:
                        collected = self.collect_mined(tup, class_name)
                        mined_dic[class_name].append(collected)
                mined_num += len(mined_dic[class_name])
            return mined_dic, mined_num

        mined_dic, mined_num = mine_regex(text)

        random.seed(len(text))  # fix random seed for reproduction
        random.shuffle(sents)

        mined_dic["word2text"] = []  # wrap as a list to align with other task types
        for sent in sents:
            if len(mined_dic["word2text"]) == self.max_collect_sent:
                break
            sent_tokens = set(self.domain_spm.encode(sent, out_type=str))
            specific_tokens_in_sent = list(self.specific_token_set & sent_tokens)
            if len(specific_tokens_in_sent) >= self.min_kw_num and len(sent_tokens) <= self.max_sent_len:
                tokens = [
                    self.domain_spm.decode(token) for token in specific_tokens_in_sent
                ]  # transfer tokens back to normal words
                dic = {
                    "relation": "word2text",
                    "token_set": tokens,
                    "sent": sent.strip(),
                }
                mined_dic["word2text"].append(dic)
        mined_num += len(mined_dic["word2text"])
        return mined_dic, mined_num

    def get_all_templates(self, entry: Dict[str, Any], random_seed: int) -> List[Tuple[str, str]]:
        if entry["relation"] == "word2text":
            return [
                ("Concepts: {tripleset}\nWrite a sentence that includes all these words.\nSentence:", "{target}"),
                (
                    "Concepts: {tripleset}\nFind a sentence in the article that includes all these words.\nSentence:",
                    "{target}",
                ),
                ("Keywords: {tripleset}\nWhat is a sentence that includes all these keywords?", "{target}"),
                (
                    "Here are some concepts: {tripleset}\nWhat is a sentence about these concepts in the article?",
                    "{target}",
                ),
                ("Produce a sentence which mentions all of these concepts: {tripleset}\nAnswer:", "{target}"),
                ("Write a sentence about the following things:\n{tripleset}\nAnswer:", "{target}"),
                ("Generate a sentence that includes all the following words: {tripleset}. Sentence:", "{target}"),
                ("Sentence: {target}\nWhat are the keywords in this sentence?", "{tripleset}"),
                ("What are the most important words in the following sentence\n{target}\nWords:", "{tripleset}"),
                ("{target}\nIdentify the most salient words in the above sentence.", "{tripleset}"),
                ("Concepts: {tripleset}\nWhat would a sentence about these concepts be like?", "{target}"),
                ("Here are some words: {tripleset}.\nWrite a sentence that describes them.", "{target}"),
                (
                    "Here are some words: {tripleset}.\nTell me a sentence that describes them in the article.",
                    "{target}",
                ),
                (
                    "Here are some concepts: {tripleset}.\nGenerate a detailed description of them.\nDescription:",
                    "{target}",
                ),
                ("Generate a sentence about: {tripleset}\nSentence:", "{target}"),
                ("Write a sentence about [{tripleset}].", "{target}"),
                ("Produce a long descriptive sentence that uses all these words: {tripleset}.\nSentence:", "{target}"),
                ("Create a set of three concepts in the following sentence.\n{target}\nConcepts:", "{tripleset}"),
                ("{tripleset}\nWhat is the sentence in the article that verbalizes these concepts?", "{target}"),
                (
                    "Keywords: {tripleset}\nTell me the sentence in the article about these concepts.\nSentence:",
                    "{target}",
                ),
                ("Here are some keywords: {tripleset}.\nWrite a sentence that includes them.", "{target}"),
                ("Generate a sentence that includes these keywords [{tripleset}].", "{target}"),
                ("Find a sentence in the above article that includes the following words: [{tripleset}].", "{target}"),
                ("Produce a long descriptive  sentence that uses all these words: {tripleset}\nAnswer:", "{target}"),
                ("Sentence: {target}\nWhat keywords can be extracted from this sentence?", "{tripleset}"),
            ]
        elif entry["relation"] == "definition":
            return [
                ("Q: {word} {verbalizer}? A:", "{definition}"),
                ("Next question: {word} {verbalizer}:", "{definition}"),
                ("{word} {verbalizer}?", "{definition}"),
                ("{word} {verbalizer}:", "{definition}"),
                ("What is the definition of {word}?", "{definition}"),
                ("How to define {word}?", "{definition}"),
                ('Explain the meaning of "{word}".', "{definition}"),
                ('What does "{word}" refer to?', "{definition}"),
                ("Please elucidate the concept of {word}\nAnswer:", "{definition}"),
                ('What is the meaning of the term "{word}"?', "{definition}"),
                ("Could you offer a definition for {word}?", "{definition}"),
                ("Could you offer a definition for {word}?\nDefinition:", "{definition}"),
            ]
        elif entry["relation"] == "topic":
            return [
                ("{sentence} {verbalizer}?", "{topic}"),
                ("{sentence} {verbalizer}:", "{topic}"),
                ("Q: {sentence} {verbalizer}? A:", "{topic}"),
                ("Answer the question\n{sentence} {verbalizer}?", "{topic}"),
                ("Answer the question\n{sentence} {verbalizer}:", "{topic}"),
                ("Answer the following question:\n{sentence} {verbalizer}?\nAnswer:", "{topic}"),
                ("Answer this question:\n{sentence} {verbalizer}?", "{topic}"),
                ("Please answer this question: {sentence} {verbalizer}?\nAnswer:", "{topic}"),
                ("Answer the question...{sentence} {verbalizer}?", "{topic}"),
                ('Can you tell me the answer to "{sentence} {verbalizer}?"?', "{topic}"),
                ("Next question: {sentence} {verbalizer}:", "{topic}"),
                ("Q: {sentence} {verbalizer}:", "{topic}"),
                ("Please answer this question: {sentence} {verbalizer}:", "{topic}"),
                ("Write the answer: {sentence} {verbalizer}?\nAnswer:", "{topic}"),
                ('What is the answer to "{sentence} {verbalizer}:"?', "{topic}"),
                ("Answer this question.\n{sentence} {verbalizer}:", "{topic}"),
                ("Answer the following question. {sentence} {verbalizer}:", "{topic}"),
                ("Question: {sentence} {verbalizer}?", "{topic}"),
                ("{sentence} {verbalizer}??", "{topic}"),
            ]
        else:
            raise ValueError("relation must be word2text, definition or topic")

    def format_single_demo(self, entry: Dict[str, Any], random_seed: int) -> List[Dict[str, str]]:
        kw_dic = {}
        if entry["relation"] == "word2text":
            kw_dic["tokens"] = entry["token_set"]
            kw_dic["tripleset"] = ", ".join(kw_dic["tokens"][: self.min_kw_num])
            kw_dic["target"] = entry["sent"].strip()
        elif entry["relation"] == "definition" or entry["relation"] == "topic":
            kw_dic = entry

        template = self.get_template(entry, random_seed)
        return self.fill_in_the_template(template, kw_dic)


@type_map.add("summarize")
class summarize(BaseType):
    def __init__(self) -> None:
        super().__init__()

    @typing.no_type_check
    def mine(self, text: str, title, **kwargs):
        # seems redundant but has to do so to align with other task types
        mined_dic = {"title": title}
        mined_num = 1 if title is not None else 0
        return mined_dic, mined_num

    def get_all_templates(self, entry: Dict[str, Any], random_seed: int) -> List[Tuple[str]]:
        # those are templates when summarization is conducted but text completion is NOT conducted
        return [
            # summary_templates
            (
                "{context_wo_title}\n\nWhat is a potential title for this context in the {domain} domain? \nTitle: {title}",
            ),
            ("{domain} article: {context_wo_title}{qa_demos}\n\nWhat is the title of this article? {title}",),
            ("Article: {context_wo_title}{qa_demos}\n\nGenerate a title for this {domain} paragraph.\nTitle: {title}",),
            ("{context_wo_title}\n\nWrite a title for the above {domain} article. {title}",),
            ("{context_wo_title}\nBriefly summarize this {domain} text? {title}",),
            (
                "Article in the {domain} domain: {context_wo_title}\n\nGenerate a short summary for this article.\nAnswer: {title}",
            ),
            (
                "{context_wo_title}{qa_demos}\n\nSummarize the aforementioned {domain} text in a single sentence. {title}",
            ),
            (
                "{context_wo_title}\nCan you generate a short summary of the above {domain} paragraph? {title}{qa_demos}",
            ),
            (
                "{context_wo_title}\nPlease write a short summary for the above article in the {domain} domain. {title}{qa_demos}",
            ),
            ("Context: {context_wo_title}{qa_demos}\n\nWhat was this {domain} article about? {title}",),
            # write based on title
            (
                "Write an article about {domain} domain, using the following title: {title}.\nArticle: {context_wo_title}{qa_demos}",
            ),
            (
                "Title: {title}\nWrite a an article about {domain} domain based on this title. {context_wo_title}{qa_demos}",
            ),
            ('Use the title "{title}" to write a {domain} article.\nArticle: {context_wo_title}{qa_demos}',),
            (
                "Craft an informative article about the {domain} domain, drawing from the following summary: {title}\nArticle: {context_wo_title}{qa_demos}",
            ),
            (
                "Create a {domain} article inspired by the provided title: {title}\nOutput: {context_wo_title}{qa_demos}",
            ),
            ('Can you develop an engaging {domain} article using the title "{title}"? {context_wo_title}{qa_demos}',),
            (
                "Write an informative piece on the {domain} domain, using the provided title: {title}. {context_wo_title}{qa_demos}",
            ),
            (
                "Craft an article focused on {domain}, utilizing the provided title: {title}.\nArticle: {context_wo_title}{qa_demos}",
            ),
            (
                "Compose an in-depth {domain} article based on the title: {title}\nArticle: {context_wo_title}{qa_demos}",
            ),
            (
                'Can you create an article delving into the {domain} domain, incorporating the given title "{title}"? {context_wo_title}{qa_demos}',
            ),
        ]

    def format_single_demo(self, entry: Dict[str, Any], random_seed: int) -> List[Dict[str, str]]:
        sents = entry.pop("sents")
        template = self.get_template(entry, random_seed)

        entry["context_wo_title"] = "".join(sents).strip()
        final_demo = self.fill_in_the_template(template, entry)
        return final_demo


@type_map.add("text_completion")
class text_completion(BaseType):
    def __init__(self) -> None:
        super().__init__()

    @typing.no_type_check
    def mine(self, sents: Any, **kwargs) -> Tuple[Dict[str, Any], int]:
        # seems redundant but has to do so to align with other task types
        mined_dic = {"sents": sents}
        mined_num = 1 if len(sents) >= 4 else 0
        return mined_dic, mined_num

    def get_all_templates(self, entry: Dict[str, Any], random_seed: int) -> List[Tuple[str, str]]:
        # those are templates when text completion is conducted but summarization is NOT conducted
        return [
            ("Please complete an article: {context_1st_half}", "{context_2nd_half}"),
            (
                "Here is the first part of an article: {context_1st_half}\n\nHow would you continue the article?",
                "{context_2nd_half}",
            ),
            (
                "Explore the initial section of an article: {context_1st_half}\nWhat could be the next part?",
                "{context_2nd_half}",
            ),
            ("Read the beginning of an article {context_1st_half}\n\nWrite the subsequent part?", "{context_2nd_half}"),
            (
                "In this article snippet, you will find the first part: {context_1st_half}\nHow would you compose the remaining section?",
                "{context_2nd_half}",
            ),
            (
                "Take a look at the introductory part of an article: {context_1st_half}\n\nYour challenge is to write the following segment",
                "{context_2nd_half}",
            ),
            (
                "Review the initial portion of an article: {context_1st_half}\nWhat would you include in the rest of the article?",
                "{context_2nd_half}",
            ),
            (
                "Consider the first segment of an article: {context_1st_half}\nContinuation of the article:",
                "{context_2nd_half}",
            ),
            (
                "Examine the first segment of an article: {context_1st_half}\n\nQuestion: Complete the article?\nCompletion:",
                "{context_2nd_half}",
            ),
            (
                "Read the beginning of an article: {context_1st_half}\n\nHow would you extend the article?",
                "{context_2nd_half}",
            ),
        ]

    def format_single_demo(self, entry: Dict[str, Any], random_seed: int) -> List[Dict[str, str]]:
        sents = entry.pop("sents")
        entry["context_1st_half"] = entry["title"] + "\n" if entry["title"] is not None else ""

        cut_index = random.Random(random_seed).randint(1, len(sents) - 1)

        entry["context_1st_half"] += "".join(sents[:cut_index]).strip()
        entry["context_2nd_half"] = "".join(sents[cut_index:]).strip()
        template = self.get_template(entry, random_seed)
        final_demo = self.fill_in_the_template(template, entry)
        return final_demo


# NOTE: useless if we don't have the title
@type_map.add("summarize_completion")
class summarize_completion(BaseType):
    def __init__(self) -> None:
        super().__init__()

    def get_all_templates(self, entry: Dict[str, Any], random_seed: int) -> List[Tuple[str]]:
        # applicable to both text completion and summarization:
        return [
            (
                "Please complete an article about {domain}: {context_1st_half} {context_2nd_half}{qa_demos}\n\nWhat was this article about?\nAnswer: {title}",
            ),
            (
                "Here is the first part of an article about {domain}: {context_1st_half}\n\nPlease complete it.\nCompletion: {context_2nd_half}{qa_demos}\n\nWhat was this article about? {title}",
            ),
            (
                "Explore the initial section of an article on {domain}: {context_1st_half}\n\nProvide the text ending? {context_2nd_half}\n\nPropose a title for this context? {title}{qa_demos}",
            ),
            (
                "Read the beginning of an article about {domain}: {context_1st_half}\n\nYour task is to add the subsequent part. {context_2nd_half}\n\nBriefly summarize this text. Summary: {title}{qa_demos}",
            ),
            (
                "In this article snippet about {domain}, you will find the first half: {context_1st_half}\n\nCompose the remaining section: {context_2nd_half}\n\nWrite a title for it.\nTitle: {title}{qa_demos}",
            ),
            (
                "Take a look at the first part of an article on {domain}: {context_1st_half}\n\nYour challenge is to write the following segment. {context_2nd_half}\n\nWhat is a very short summary of the above text? {title}{qa_demos}",
            ),
            (
                "Review the initial portion of an article discussing {domain}: {context_1st_half}\n\nWhat would you include in the rest of the article? {context_2nd_half}\n\nWhat is a shorter version of this article?\nShort version: {title}{qa_demos}",
            ),
            (
                "Consider the opening of an article centered around {domain}: {context_1st_half}\n\nNow, provide the continuation of the article.\nContinuation: {context_2nd_half}\n\nWhat was this article about? {title}{qa_demos}",
            ),
            (
                "Examine the first segment of an article exploring {domain}: {context_1st_half}\n\nComplete the article? {context_2nd_half}\nCan you generate a short summary of the above paragraph?\nAnswer: {title}{qa_demos}",
            ),
            (
                "Read the beginning of an article on {domain}: {context_1st_half}\n\nHow would you extend the article? {context_2nd_half}\n\nPlease write a short summary for the above article. {title}{qa_demos}",
            ),
        ]

    def format_single_demo(self, entry: Dict[str, Any], random_seed: int) -> List[Dict[str, str]]:
        sents = entry.pop("sents")
        template = self.get_template(entry, random_seed)
        cut_index = random.Random(random_seed).randint(1, len(sents) - 1)

        entry["context_1st_half"] = "".join(sents[:cut_index]).strip()
        entry["context_2nd_half"] = "".join(sents[cut_index:]).strip()
        final_demo = self.fill_in_the_template(template, entry)
        return final_demo


@type_map.add("no_summarize_completion")
class no_summarize_completion(BaseType):
    def __init__(self) -> None:
        super().__init__()

    def get_all_templates(self, entry: Dict[str, Any], random_seed: int) -> List[Tuple[str]]:
        # applicable to having no summarization and no completion
        return [
            ("Please answer some questions about the following article:\n{context}\n",),
            ("Read this article and answer questions\n{context}\n",),
            ("{context}\n",),
            ("Answer some questions about this article:\n{context}\n",),
            ("Here are some questions about this article: {context}\n",),
            ("Article: {context}\n",),
            ("Read this article: {context}\n",),
            ("Given the rticle: {context}\n",),
            ("Context: {context}\n",),
            ("Article: {context}\n",),
            ("Use this article to answer the questions: {context}\n",),
            ("Answer based on context :\n{context}\n",),
        ]

    def format_single_demo(self, entry: Dict[str, Any], random_seed: int) -> List[Dict[str, str]]:
        sents = entry.pop("sents")
        entry["context"] = entry["title"] + "\n" if entry["title"] is not None else ""

        template = self.get_template(entry, random_seed)

        entry["context"] += "".join(sents).strip()
        final_demo = self.fill_in_the_template(template, entry)
        return final_demo


@type_map.add("overall")
class overall(BaseType):
    def __init__(self) -> None:
        super().__init__()
        self.demo_deliminator = "\n\n"
        self.intro_deliminators = [  # connect raw text with the followed QAs
            ("\nPlease answer some questions about the above article:\n\n",),
            ("\nAnswer some questions about the above article :\n\n",),
            ("\n\nWhat are the answers to these questions?\n",),
            ("\n\nNow answer these questions:\n\n",),
            ("\nNow answer the following questions:\n\n",),
            ("\n\nWhat are the answers to the questions or completions:\n",),
            ("\nHow would one answer these questions in the domain:\n\n",),
            ("\n\nUse evidence from the article to answer these questions:\n\n",),
            ("\n\nUse this above article to answer the questions:\n",),
            ("\nAnswer the following questions based on the article:\n\n",),
            ("\nAnswer these questions:\n",),
            ("\n\nBased on the above article, answer questions.\n\n",),
            ("\nWrite some question-answer pairs about the above article:\n\n",),
            ("\nRespond to the following questions based on the above article:\n\n",),
            ("\n\nUpon reading the article, answer the following questions:\n\n",),
            ("\nEvaluate your understanding of the article by answering the following questions:\n\n",),
        ]

    def format_recomprehension(
        self, overall_entry: Dict[str, Any], insert_types: List[str] = TYPES
    ) -> Tuple[str, Dict[str, Any]]:
        qa_demo_list = []
        seed = overall_entry["text_id"]
        count_dict: Dict[str, Any] = {}
        for type in list(set(insert_types) & set(["nli", "common_reason", "paraphrase", "word2text"])):
            type_cls = type_map.cls_dic[type]()
            type_examples = []
            count_dict[type] = {}
            for subcategory, examples in overall_entry[type].items():
                if len(examples) == 0:
                    continue
                random.Random(seed).shuffle(examples)
                type_examples += examples[: type_cls.max_subcategory_num]
                count_dict[type][subcategory] = len(examples[: type_cls.max_subcategory_num])
            if len(type_examples) == 0:
                continue
            # ensure examples of one type altogether, to imitate the few-shot setting

            qa_demo_list += [type_cls.format_single_demo(example, seed) for example in type_examples]

        if len(qa_demo_list) > 0:
            random.Random(seed).shuffle(qa_demo_list)
            intro = random.Random(seed).choice(self.intro_deliminators)[0]
            qa_demos: List[Dict[str, str]] = sum(qa_demo_list, [])
            qa_demos[0]["content"] = intro + qa_demos[0]["content"]
        else:
            qa_demos = []

        def summaize_only(count_dict: Dict[str, int]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
            count_dict["summarize"] = 1
            count_dict["text_completion"] = 0
            overall_cls = summarize()
            entry = overall_entry["summarize"]
            entry["sents"] = overall_entry["text_completion"]["sents"]
            entry["qa_demos"] = qa_demos
            entry["spm"] = self.ori_spm
            read_compre_demo = overall_cls.format_single_demo(entry, seed)
            return read_compre_demo, count_dict

        def completion_only(count_dict: Dict[str, int]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
            count_dict["summarize"] = 0
            count_dict["text_completion"] = 1
            overall_cls = text_completion()
            entry = overall_entry["text_completion"]
            entry["qa_demos"] = qa_demos
            entry["title"] = overall_entry["summarize"]["title"]
            entry["spm"] = self.ori_spm
            read_compre_demo = overall_cls.format_single_demo(entry, seed)
            return read_compre_demo, count_dict

        def summarize_and_completion(count_dict: Dict[str, int]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
            count_dict["summarize"] = 1
            count_dict["text_completion"] = 1
            overall_cls = summarize_completion()
            entry = overall_entry["text_completion"]
            entry["qa_demos"] = qa_demos
            entry["title"] = overall_entry["summarize"]["title"]
            entry["spm"] = self.ori_spm
            read_compre_demo = overall_cls.format_single_demo(entry, seed)
            return read_compre_demo, count_dict

        def no_summarize_or_completion(count_dict: Dict[str, int]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
            count_dict["summarize"] = 0
            count_dict["text_completion"] = 0
            overall_cls = no_summarize_completion()
            entry = overall_entry["text_completion"]
            entry["qa_demos"] = qa_demos
            entry["title"] = overall_entry["summarize"]["title"]
            entry["spm"] = self.ori_spm
            read_compre_demo = overall_cls.format_single_demo(entry, seed)
            return read_compre_demo, count_dict

        if ("summarize" in insert_types and overall_entry["summarize"]["title"] is not None) and (
            "text_completion" in insert_types and len(overall_entry["text_completion"]["sents"]) >= 2
        ):
            np.random.seed(seed)
            read_func = np.random.choice(  # type: ignore
                [summaize_only, completion_only, summarize_and_completion, no_summarize_or_completion],  # type: ignore
                p=[0.4, 0.1, 0.4, 0.1],  # type: ignore
            )  # type: ignore
        elif "summarize" in insert_types and overall_entry["summarize"]["title"] is not None:
            np.random.seed(seed)
            read_func = np.random.choice([summaize_only, no_summarize_or_completion], p=[0.5, 0.5])  # type: ignore
        if "text_completion" in insert_types and len(overall_entry["text_completion"]["sents"]) >= 2:
            np.random.seed(seed)
            if len(qa_demos) == 0:
                read_func = completion_only
            else:
                read_func = np.random.choice([completion_only, no_summarize_or_completion], p=[0.5, 0.5])  # type: ignore
        else:
            read_func = no_summarize_or_completion

        return read_func(count_dict)


class RegexBasedReadingComprehension:
    def __init__(self, general_spm: spm.SentencePieceProcessor, domain_spm: spm.SentencePieceProcessor) -> None:
        self.inited_type_map = {}

        for type in TYPES:
            type_cls = type_map.cls_dic[type]()
            type_cls.init_spm(general_spm, domain_spm)
            self.inited_type_map[type] = type_cls

        self.overall_cls = type_map.cls_dic["overall"]()
        self.overall_cls.init_spm(general_spm, domain_spm)

        # to chunk text to sentences
        self.segmenter = Segmenter(language="en", clean=False)

    def generate(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        # NOTE: if the context has no title, use the following code:
        title = None
        context_wo_title = entry["text"]

        # truncate the context to meet the max_seq_len
        # context_wo_title = overall_cls.truncate_sentence(context_wo_title, max_len=overall_cls.max_seq_len-200)
        context_wo_title_list = self.overall_cls.truncate_sentence(
            context_wo_title, max_len=self.overall_cls.max_seq_len - 200
        )

        read_compre_list = []
        for context_wo_title in context_wo_title_list:
            sents = self.segmenter.segment(context_wo_title)
            overall_entry = {"text_id": entry["text_id"]}
            for type in TYPES:
                type_cls = self.inited_type_map[type]
                overall_entry[type], mined_num = type_cls.mine(
                    text=context_wo_title, title=title, sents=copy.deepcopy(sents)
                )

        # create the reading comprehension text
        read_compre, count_dict = self.overall_cls.format_recomprehension(copy.deepcopy(overall_entry))
        # count_dict includes the number of comprehension tasks per task type
        # you may use `mined_num` and `count_dict` for data analysis
        read_compre_list.append(read_compre)

        return {"read_compre": read_compre_list, "file_name": entry["file_name"]}

    def dataset_generator(
        self, input_dir_or_file: str, column: Optional[str], workers: int = 1
    ) -> Iterator[Tuple[int, str, str]]:
        generator = input_generator(input_dir_or_file, column)

        raw_texts = []
        for text_id, (filename, content) in enumerate(generator):
            text = content.strip()
            raw_texts.append({"text": text, "text_id": text_id, "file_name": filename})

        logger.info("transferring raw texts into reading comprehension...")
        read_compre = list(process_map(self.generate, raw_texts, max_workers=workers, chunksize=8192))

        logger.info("saving reading comprehension texts...")
        # sort by text_id to align with the order of raw texts
        for entry in read_compre:
            for index, read_compre_example in enumerate(entry["read_compre"]):
                file_name = entry["file_name"]
                yield index, file_name, read_compre_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Directory containing the input files OR a CSV file")
    parser.add_argument("--csv_column", type=str, help="Column to read from the CSV file")
    parser.add_argument(
        "--debug_output_dir",
        type=str,
        help="directory of the output reading comprehension texts",
    )
    parser.add_argument("--general_spm_path", type=str, help="path of the general sentencepiece model", required=True)
    parser.add_argument(
        "--domain_spm_path",
        type=str,
        help="path of the domain sentencepiece model",
    )
    parser.add_argument(
        "--domain_tokenizer_training_text",
        type=str,
        help="path of the domain sentencepiece model",
    )
    parser.add_argument(
        "--output_dataset_name",
        type=str,
        required=True,
        help="name of the output dataset",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if os.path.isfile(args.input) and not args.csv_column:
        raise ValueError("a CSV column must be specified if the input is a file")

    if not (args.domain_spm_path or args.domain_tokenizer_training_text):
        # warn user that the domain tokenizer will be created from the input files
        logger.warning(
            "No domain tokenizer is provided nor explicit file for training domain tokenizer is provided, "
            "the domain tokenizer will be created from the input files, "
        )

    if args.domain_tokenizer_training_text:
        # train domain tokenizer
        domain_spm = create_domain_tokenizer(args.domain_tokenizer_training_text)
    elif args.domain_spm_path:
        domain_spm = spm.SentencePieceProcessor(model_file=args.domain_spm_path)
    else:
        domain_spm = create_domain_tokenizer_from_files(args.input, args.csv_column)

    general_spm = spm.SentencePieceProcessor(model_file=args.general_spm_path)

    # get max worker for multi-process
    max_workers = min((os.cpu_count() or 1) // 2, 1)

    logger.info(f"max_workers for generation of regex data: {max_workers}")

    rc = RegexBasedReadingComprehension(general_spm, domain_spm)
    dataset_generator = rc.dataset_generator(args.input, column=args.csv_column, workers=max_workers)

    if args.debug_output_dir:
        in_memory_dataset = []

        logger.info("saving debug data...")
        os.makedirs(args.debug_output_dir, exist_ok=True)

        for index, filename, read_compre_example in dataset_generator:
            with open(os.path.join(args.debug_output_dir, f"{filename}_{index}.json"), "w", encoding="utf8") as f:
                json.dump({"messages": read_compre_example}, f)
            in_memory_dataset.append({"messages": read_compre_example})
    else:
        in_memory_dataset = [{"messages": rc_text} for _, _, rc_text in dataset_generator]

    # make dataset from reading comprehension texts
    logger.info("making dataset...")

    regex_dataset = datasets.Dataset.from_list(in_memory_dataset)
    regex_dataset.save_to_disk(args.output_dataset_name)

    logger.info(f"Done. Dataset saved to disk at {args.output_dataset_name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
