# iterate through all files in data json folder and load json
import csv
import glob
import json
from typing import Dict, List, Optional

from tqdm import tqdm

# output csv file
csv_filename = "abstract_output.csv"


def get_date(patent: Dict) -> str:
    try:
        date = patent["bibliographic_information"]["document_date"]
    except Exception:
        try:
            date = patent["bibliographic_information"]["date"]
        except Exception:
            try:
                date = patent["bibliographic_information"]["Issue Date"]
            except Exception:
                print(patent["bibliographic_information"].keys())
    return date


def get_ipc(patent: Dict) -> str:
    try:
        ipc = patent["classifications"]["main_or_locrano_class"]
    except Exception:
        try:
            ipc = patent["classifications"]["us_classifications_cpc_text"]
        except Exception:
            try:
                ipc = patent["classifications"][0]["ICL"][0]
            except Exception:
                try:
                    ipc = (
                        patent["classifications"]["section"]
                        + patent["classifications"]["class"]
                        + patent["classifications"]["subclass"]
                    )
                except Exception:
                    print("WARNING: no ipc found")
                    ipc = None

    if isinstance(ipc, list):
        ipc = ipc[0]

    return ipc


def get_abstract(patent: Dict) -> Optional[str]:
    try:
        abstract = patent["abstract"]
        if isinstance(abstract, list):
            abstract = abstract[0]
    except Exception:
        abstract = None

    return abstract


def get_title(patent: Dict) -> str:
    try:
        title = patent["bibliographic_information"]["invention_title"]
    except Exception:
        try:
            title = patent["bibliographic_information"]["Title of Invention"]
        except Exception:
            title = patent["bibliographic_information"]["title_of_invention"]
    return title


def get_claims(patent: Dict) -> Optional[str]:
    try:
        claims = patent["claim_information"]
    except Exception:
        claims = None

    return claims


def extract_batch(batch: List) -> tuple[list, list, int]:
    abstract_list = []
    title_list = []

    skipped_abstracts_num = 0

    for filename in tqdm(batch):
        patent = json.load(open(filename))

        abstract = get_abstract(patent)

        title = get_title(patent)

        if (abstract and title) is not None:
            abstract_list.append(abstract)
            title_list.append(title)

        else:
            skipped_abstracts_num += 1

    return abstract_list, title_list, skipped_abstracts_num


files_to_load = glob.glob("./json_data/*.json")
# increment files to load in batches of 100 and print progress and collect total skipped abstracts and claims

total_skipped_abstracts = 0
total_skipped_claims = 0
total_skipped_batches = 0


with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
    # Create a CSV writer
    csv_writer = csv.writer(csv_file, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Write the header row
    csv_writer.writerow(["Title", "Abstract"])

    for i in tqdm(range(0, len(files_to_load), 10)):
        files_to_load_batch = files_to_load[i : i + 10]

        abstract_list, title_list, skipped_abstracts_num = extract_batch(files_to_load_batch)

        # Write data from the lists
        for title, abstract in zip(title_list, abstract_list, strict=True):
            csv_writer.writerow([title, abstract])
