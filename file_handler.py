import os
import subprocess
import xml.etree.ElementTree as ET
import re
from collections import Counter

import nltk
from nltk import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm

from xml_handler import clean_xml, extract_valid_xml
from ufal.morphodita import Tagger, Forms, TaggedLemmas, TokenRanges

# Load once at module level
MORPHODITA_MODEL_PATH = os.path.join("morfflex", "czech-morfflex2.0-pdtc1.0-220710.tagger")
tagger = Tagger.load(MORPHODITA_MODEL_PATH)
morpho = tagger.getMorpho()
tokenizer = tagger.newTokenizer()

nltk.download('stopwords')
stopword_en = set(stopwords.words("english"))

stopword_cs = {
    "a", "aby", "aj", "ale", "anebo", "ani", "asi", "aspoň",
    "bez", "bude", "by", "byl", "byla", "byli", "bylo",
    "co", "což", "či", "do", "ho", "i", "jak", "je", "jeho",
    "její", "jejich", "jen", "jenž", "jsme", "jsou",
    "jste", "k", "kam", "kde", "když", "ke", "který",
    "má", "málo", "mi", "mně", "můj", "na", "nad",
    "ne", "nebo", "něco", "někdo", "některý",
    "o", "od", "po", "pod", "pro", "proto", "se",
    "si", "s", "ta", "tak", "také", "takže",
    "ten", "tento", "to", "tohle", "tohoto",
    "toto", "tu", "tuto", "ty", "tý", "u",
    "už", "v", "ve", "vy", "z", "za", "ze"
}


def download_and_extract():
    print("Downloading and extracting data files...")
    subprocess.run([
        "wget",
        "--user", "npfl103",
        "--password", "npfl103",
        "http://ufal.mff.cuni.cz/~pecina/courses/npfl103/data/A1.tgz"
    ], check=True)

    subprocess.run([
        "tar", "xf", "A1.tgz"
    ], check=True)

def nltk_download(resource):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

def read_documents_list(lst_name=None):
    lst_path = os.path.join("A1", lst_name)

    filenames = []
    with open(lst_path, "r", encoding="utf-8") as f:
        for line in f:
            filename = line.strip()
            if not filename:
                continue
            filenames.append(filename)

    return filenames

def load_all_documents(language, run, documents_list=None):
    print("Loading all documents...")

    docs_dir = os.path.join("A1", f"documents_{language}")
    parsed_docs = {}

    if documents_list:
        filenames = read_documents_list(documents_list)
    else:
        filenames = [f for f in os.listdir(docs_dir) if f.endswith(".xml")]

    for filename in tqdm(filenames, desc="Loading documents"):
        if not filename.endswith(".xml"):
            continue

        path = os.path.join(docs_dir, filename)

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        content = clean_xml(content)
        content = extract_valid_xml(content)

        try:
            # print(f"Processing {filename}...")
            file_docs = parse_all_docs(content, run, language)
            duplicate_keys = parsed_docs.keys() & file_docs.keys()

            if duplicate_keys:
                print("Duplicate key:", next(iter(duplicate_keys)))
            else:
                parsed_docs.update(file_docs)
        except ET.ParseError as e:
            print(f"Skipping {filename}. error: {e}")

    print("Finished loading all documents.")
    return parsed_docs

def remove_stop_words(tokens, language):
    stop_words = stopword_en if language == "en" else stopword_cs
    return [t for t in tokens if t not in stop_words]

def lemmatize_cs(docs_dict):
    # lemmatize using MorphoDiTa
    forms = Forms()
    lemmas = TaggedLemmas()
    tokens = TokenRanges()

    result = {}

    for docno, token_list in docs_dict.items():
        # rejoin tokens for its tokenizer
        text = " ".join(token_list)

        tokenizer.setText(text)
        doc_lemmas = []

        while tokenizer.nextSentence(forms, tokens):
            tagger.tag(forms, lemmas)

            for i in range(len(lemmas)):
                raw_lemma = lemmas[i].lemma

                # strip suffixes from lemma
                clean = raw_lemma.split("_")[0].split("-")[0].casefold().strip()

                if not clean:
                    continue
                if not re.search(r"\w", clean):
                    continue
                if clean in stopword_cs:
                    continue

                doc_lemmas.append(clean)

        result[docno] = doc_lemmas

    return result

def stem_tokens(tokens, language):
    if language == "en":
        stemmer = SnowballStemmer("english")
        return [stemmer.stem(t) for t in tokens]
    # no stemming for cs
    return tokens

def lemmatize_en(tokens):
    nltk_download("wordnet")
    nltk_download("omw-1.4")
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]

def preprocess_text(text, run, lang):
    if not text:
        return []

    # run-0 (baseline)
    if run == 0:
        return re.split(r"\s+", text)

    # run-1
    elif run == 1:
        # split on spaces, commas, punctuation, parentheses, hyphens
        tokens = re.split(r"[^\w]+", text.casefold())

        tokens = remove_stop_words(tokens, lang)

        tokens = [t for t in tokens if t]

        # cs has lemmatization separately
        if lang == "en":
            tokens = lemmatize_en(tokens)

        # equivalence classes used only for Czech because it didn't improve much for English. (improved P_10 but hurt MAP)
        if lang == "cs":
            tokens = normalize_equivalence_classes(tokens, lang)

        # stemming for en only
        if lang == "en":
            tokens = stem_tokens(tokens, lang)

        return tokens

    else:
        raise ValueError(f"Unsupported run: {run}")

def parse_single_doc_voc(doc, run, lang):
    docno = None
    tokens = []

    for child in doc:
        tag = child.tag
        text = (child.text or "").strip()

        if not text:
            continue

        if tag == "DOCNO":
            docno = text
        else:
            tokens.extend(preprocess_text(text, run, lang))

    if docno is None:
        return {}

    return docno, tokens

# converts all xml files into a dictionary <docno:list_tokens>
def parse_all_docs(xml_text, run, lang):

    root = ET.fromstring(xml_text)
    docs_dict = {}

    for doc in root.findall(".//DOC"):
        result = parse_single_doc_voc(doc, run, lang)

        if not result:
            continue

        docno, tokens = result
        docs_dict[docno] = tokens

    # MorphoDiTa lemmatization
    if run == 1 and lang == "cs":
        docs_dict = lemmatize_cs(docs_dict)

    return docs_dict

def query_constructor(lang, run, train, queries_path=None):
    print("Constructing queries...")
    if queries_path:
        path = os.path.join(os.getcwd(), "A1", queries_path)
    else:
        if train:
            xml_path = f"topics-train_{lang}.xml"
        else:
            xml_path = f"topics-test_{lang}.xml"

        path = os.path.join(os.getcwd(), "A1", xml_path)

    queries = {}

    tree = ET.parse(path)
    root = tree.getroot()

    for topic in root.findall("top"):
        qid = topic.find("num").text.strip()
        title = topic.find("title").text.strip()

        # applies the baseline for run-0 or additional improvements for run-1
        tokens = preprocess_text(title, run, lang)
        queries[qid] = tokens

    if run == 1 and lang == "cs":
        queries = lemmatize_cs(queries)

    return queries

def normalize_equivalence_classes(tokens, lang):
    map_en = {
        "u.s.": "usa", "u_s": "usa",
        "u.k.": "uk", "britain": "uk", "england": "uk",
        "ussr": "russia", "soviet": "russia",
        "eu": "europe", "european": "europe",
        "govt": "government", "gov": "government",
        "corp": "corporation", "inc": "corporation", "ltd": "corporation",
        "dept": "department",
        "pct": "percent", "%": "percent",
        "vs": "versus", "dr": "doctor", "mr": "mister", "mrs": "missus",
        "jan": "january", "feb": "february",
        "apr": "april", "jun": "june", "jul": "july",
        "oct": "october", "nov": "november", "dec": "december",
    }

    map_cs = {
        "čr": "česko", "česká republika": "česko",
         "slovenská republika": "slovensko",
        "usa": "amerika", "spojené státy": "amerika",
        "eu": "evropa", "evropská unie": "evropa",
        "vl": "vláda", "č": "číslo",
        "mld": "miliarda", "mil": "milion",
        "tis": "tisíc", "tj": "to jest", "tzv": "takzvaný",
        "např": "například", "atd": "a tak dále",
    }

    mapping = map_en if lang == "en" else map_cs
    normalized = []

    for t in tokens:
        # spelling normalization
        t = mapping.get(t, t)

        # number normalization
        if re.fullmatch(r"\d+([.,]\d+)?", t):
            t = "<NUM>"

        # collapse repeated characters, "goood" to: "good"
        if lang == "en":
            t = re.sub(r"(.)\1{2,}", r"\1\1", t)

        normalized.append(t)

    return normalized

# gets a dict and returns the collection of it (run-0)
def dict_to_counter(dict):
    print("Converting dictionary to counter...")
    return {qid: Counter(tokens) for qid, tokens in dict.items()}

def write_results(results, output_file, run):
    print("Writing results to " + output_file)
    with open(output_file, "w", encoding="iso-8859-1") as f:
        for qid, ranked_docs in results.items():
            for rank, (docno, score) in enumerate(ranked_docs):
                f.write(f"{qid}\t0\t{docno}\t{rank}\t{score:.6f}\t{run}\n")
