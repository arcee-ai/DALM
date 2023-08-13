#iterate through all files in data json folder and load json
import glob
import json
from tqdm import tqdm
import csv

csv_filename = "abstract_output.csv"

def get_date(patent):
    try:
        date = patent["bibliographic_information"]["document_date"]
    except:
        try:
            date = patent["bibliographic_information"]["date"]
        except:
            try:
                date = patent["bibliographic_information"]["Issue Date"]
            except:
                
                print(patent["bibliographic_information"].keys())
    return date

def get_ipc(patent):
    try:
        ipc = patent["classifications"]["main_or_locrano_class"]
    except:
        try:
            ipc = patent["classifications"]["us_classifications_cpc_text"]
        except:
            try:
                ipc = patent["classifications"][0]["ICL"][0]
            except:
                try:
                    ipc = patent["classifications"]["section"] + patent["classifications"]["class"] + patent["classifications"]["subclass"]
                except:
                    print("WARNING: no ipc found")
                    ipc = None
                    
    if type(ipc) == list:
        ipc = ipc[0]
        
                
    return ipc

def get_abstract(patent):
    try:
        abstract = patent["abstract"]
        if type(abstract) == list:
            abstract = abstract[0]
    except:
        abstract = None    
    
    return abstract

def get_title(patent):
    try:
        title = patent["bibliographic_information"]["invention_title"]
    except:
        try:
            title = patent["bibliographic_information"]["Title of Invention"]
        except:
            title = patent["bibliographic_information"]["title_of_invention"]
    return title
        
def get_claims(patent):
    try:
        claims = patent["claim_information"]
    except:
        claims = None
        
    return claims

def extract_batch(batch):
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
    

files_to_load = glob.glob('./json_data/*.json')
#increment files to load in batches of 100 and print progress and collect total skipped abstracts and claims

total_skipped_abstracts = 0
total_skipped_claims = 0
total_skipped_batches = 0


with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
    # Create a CSV writer
    csv_writer = csv.writer(csv_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    # Write the header row
    csv_writer.writerow(['Title', 'Abstract'])

    for i in tqdm(range(0, len(files_to_load), 10)):
        files_to_load_batch = files_to_load[i:i+10]

        abstract_list, title_list, skipped_abstracts_num = extract_batch(files_to_load_batch)

        # Write data from the lists
        for title, abstract in zip(title_list, abstract_list):
            csv_writer.writerow([title, abstract])







     
    