import argparse

from utils import run

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--run", required=True, type=int, default=None, help="type of run", choices=[0, 1])
parser.add_argument("-q", "--queries", required=True, type=str, default=None, help="path to topics xml file")
parser.add_argument("-d", "--documents_list", required=True, type=str, default=None, help="path to documents list")
parser.add_argument("-o", "--output", required=True, type=str, default=None, help="path to output file ")


# ./run -q topics.xml -d documents.lst -r run -o sample.res

def main(args):
    # mismatch of documents
    if "en" in args.queries and "cs" in args.documents_list:
        print(f"Error. Got queries and documents in different languages.")
        exit(1)

    train = True if "train" in args.queries else False
    language = "en" if "en" in args.queries else "cs"

    run(
        language,
        args.run,
        train,
        args.queries,
        args.documents_list,
        args.output,
    )

if __name__ == '__main__':
    args = parser.parse_args()
    print(
    f"""
    ========================================
    Running Information Retrieval System
    ----------------------------------------
    Language : {"en" if "en" in args.queries else "cs"}
    Run      : run-{args.run}
    Dataset  : {"train" if "train" in args.queries else "test"}
    Queries  : {args.queries}
    Documents: {args.documents_list}
    Output   : {args.output}
    ========================================
    """
    )
    main(args)
