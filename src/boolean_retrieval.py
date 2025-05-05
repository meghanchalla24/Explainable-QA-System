import json
import re
import math
import psutil
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

### ------------------ Preprocessing & Inverted Index ------------------ ###

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return sorted(set(tokens))

def build_inverted_index(pdf_data):
    inverted_index = defaultdict(list)
    all_doc_ids = []

    for pdf_file, data in pdf_data.items():
        for page_num, page in enumerate(data["extracted_text"], start=1):
            doc_id = (pdf_file, page_num)
            all_doc_ids.append(doc_id)

            text = page.get("text", "")
            terms = preprocess_text(text)

            for term in terms:
                if doc_id not in inverted_index[term]:
                    inverted_index[term].append(doc_id)

    return dict(inverted_index), all_doc_ids

### ------------------ Boolean Operators with Skip Pointers ------------------ ###

def and_operation(list1, list2):
    if not list1 or not list2:
        return []

    p1, p2 = 0, 0
    result = []

    skip1 = max(1, int(math.sqrt(len(list1))))
    skip2 = max(1, int(math.sqrt(len(list2))))

    while p1 < len(list1) and p2 < len(list2):
        if list1[p1] == list2[p2]:
            result.append(list1[p1])
            p1 += 1
            p2 += 1
        elif list1[p1] < list2[p2]:
            if p1 + skip1 < len(list1) and list1[p1 + skip1] <= list2[p2]:
                p1 += skip1
            else:
                p1 += 1
        else:
            if p2 + skip2 < len(list2) and list2[p2 + skip2] <= list1[p1]:
                p2 += skip2
            else:
                p2 += 1
    return result

def or_operation(list1, list2):
    if not list1:
        return list2
    if not list2:
        return list1

    result = []
    p1, p2 = 0, 0

    while p1 < len(list1) and p2 < len(list2):
        if list1[p1] == list2[p2]:
            result.append(list1[p1])
            p1 += 1
            p2 += 1
        elif list1[p1] < list2[p2]:
            result.append(list1[p1])
            p1 += 1
        else:
            result.append(list2[p2])
            p2 += 1

    result.extend(list1[p1:])
    result.extend(list2[p2:])
    return result

def not_operation(posting, all_doc_ids):
    return sorted(set(all_doc_ids) - set(posting))

### ------------------ Boolean Query Processing ------------------ ###

def evaluate_query(query, inverted_index, all_doc_ids):
    ps = PorterStemmer()
    tokens = re.findall(r'\(|\)|\w+|AND|OR|NOT', query.lower())

    precedence = {"not": 3, "and": 2, "or": 1}
    operators = []
    operands = []

    def apply_operator():
        op = operators.pop()
        if op == "not":
            operand = operands.pop()
            operands.append(not_operation(operand, all_doc_ids))
        else:
            right = operands.pop()
            left = operands.pop()
            if op == "and":
                operands.append(and_operation(left, right))
            elif op == "or":
                operands.append(or_operation(left, right))

    for token in tokens:
        token = token.lower()
        if token in {"and", "or", "not"}:
            while operators and operators[-1] != "(" and precedence[operators[-1]] >= precedence[token]:
                apply_operator()
            operators.append(token)
        elif token == "(":
            operators.append(token)
        elif token == ")":
            while operators and operators[-1] != "(":
                apply_operator()
            operators.pop()
        else:
            stemmed = ps.stem(token)
            operands.append(inverted_index.get(stemmed, []))

    while operators:
        apply_operator()

    return operands[-1] if operands else []


def paginate_results(results, page=1, page_size=10):
    start = (page - 1) * page_size
    end = start + page_size
    return results[start:end]
