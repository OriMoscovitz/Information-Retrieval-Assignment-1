import re


def clean_xml(text):
    return re.sub(
        r"[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD]",
        "",
        text
    )

def extract_valid_xml(text):
    start = text.find("<LATIMES2002>")
    end = text.rfind("</LATIMES2002>")

    if start == -1 or end == -1:
        return text

    end += len("</LATIMES2002>")
    return text[start:end]