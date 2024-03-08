# Importing nessesary libraries and modules. 
import os # Importing os module to interact with the operating system.
import multiprocessing as mp # Importing multiprocessing module to run multiple processes.
import io # Importing io module to deal with various types of I/O.
import spacy # Importing spacy module to deal with natural language processing.
import pprint # Importing pprint module to pretty print the output.
from spacy.matcher import Matcher # Importing Matcher from spacy.matcher module.
from . import utils # Importing utils module from the current package.

# Class for parsing resumes to extract basic details like name, email, mobile number, skills, and education degree.
#    resume (str): Path to the resume file or an instance of io.BytesIO.
#    skills_file (str): Path to the skills file containing a list of skills.
#    custom_regex (str): Custom regular expression pattern for extracting mobile numbers.
class ResumeParser(object):
    def __init__(
        self,
        resume,
        skills_file=None,
        custom_regex=None
    ):
        nlp = spacy.load('en_core_web_sm') # Loading the en_core_web_sm model.
        custom_nlp = spacy.load(os.path.dirname(os.path.abspath(__file__))) # Loading the custom model.
        self.__skills_file = skills_file
        self.__custom_regex = custom_regex
        self.__matcher = Matcher(nlp.vocab) # Creating a matcher object.
        self.__details = {
            'name': None,
            'email': None,
            'mobile_number': None,
            'skills': None,
            'degree': None,
            'no_of_pages': None,
        }
        self.__resume = resume # Assigning the resume to the instance variable.
        if not isinstance(self.__resume, io.BytesIO): # Checking if the resume is not an instance of io.BytesIO.
            ext = os.path.splitext(self.__resume)[1].split('.')[1] # Extracting the extension of the resume file.
        else:
            ext = self.__resume.name.split('.')[1]
        self.__text_raw = utils.extract_text(self.__resume, '.' + ext) # Extracting the text from the resume.
        self.__text = ' '.join(self.__text_raw.split()) # Joining the text and removing extra spaces.
        self.__nlp = nlp(self.__text) # Creating a spacy nlp object.
        self.__custom_nlp = custom_nlp(self.__text_raw) # Creating a custom nlp object.
        self.__noun_chunks = list(self.__nlp.noun_chunks) # Extracting noun chunks from the resume.
        self.__get_basic_details() # Extracting basic details from the resume.
    
    # Method to return the extracted details.
    def get_extracted_data(self): 
        return self.__details 
    def __get_basic_details(self): # Method to extract basic details from the resume.
        cust_ent = utils.extract_entities_wih_custom_model(
                            self.__custom_nlp ) 
        name = utils.extract_name(self.__nlp, matcher=self.__matcher) # Extracting name from the resume.
        email = utils.extract_email(self.__text)
        mobile = utils.extract_mobile_number(self.__text, self.__custom_regex)
        skills = utils.extract_skills(
                    self.__nlp,
                    self.__noun_chunks,
                    self.__skills_file)

        entities = utils.extract_entity_sections_grad(self.__text_raw) # Extracting entities using custom model.

        try:
            self.__details['name'] = cust_ent['Name'][0]
        except (IndexError, KeyError):
            self.__details['name'] = name
        self.__details['email'] = email
        self.__details['mobile_number'] = mobile
        self.__details['skills'] = skills
        self.__details['no_of_pages'] = utils.get_number_of_pages(self.__resume)
        
        # Extracting degree
        try:
            self.__details['degree'] = cust_ent['Degree']
        except KeyError:
            pass

        return

def resume_result_wrapper(resume): # Wrapper method to extract details from the resume.
    parser = ResumeParser(resume) # Creating an instance of ResumeParser.
    return parser.get_extracted_data() 

if __name__ == '__main__': # Main method to run the program.
    pool = mp.Pool(mp.cpu_count()) # Creating a pool of processes.
    resumes = [] # List to store the resumes.
    data = [] # List to store the extracted details.
    for root, directories, filenames in os.walk('resumes'): # Looping through the resumes directory.
        for filename in filenames:
            file = os.path.join(root, filename)
            resumes.append(file)
    results = [
        pool.apply_async(
            resume_result_wrapper,
            args=(x,)
        ) for x in resumes
    ]
    results = [p.get() for p in results] # Extracting the results from the processes.
    pprint.pprint(results) # Pretty printing the extracted details.
