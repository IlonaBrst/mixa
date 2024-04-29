import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer, models,util
from transformers import pipeline 
import openai

openai.api_key = "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"
model = SentenceTransformer('all-MiniLM-L6-v2')

context = {
     
        "rare disease in pediatric neurologic disease" : "Drug in development in phase I-II for rare disease in pediatric neurologic disease",
        "Main symptoms are epileptic seizure and loss of functional skills (motor, language)" : "Main symptoms are epileptic seizure and loss of functional skills (motor, language)",
        "Treatment is administered intrathecally with requirement of 2 days of hospitalization." : "Treatment is administered intrathecally with requirement of 2 days of hospitalization.",
        "Phase I/II study" : "Phase I/II study",
        "open-label trial" : "open-label trial",
        "pediatric neurologic condition" : "rare disease in pediatric neurologic disease",
        "rare neurological disorder children" : "rare disease in pediatric neurologic disease",
        "epileptic seizures" : "epileptic seizures",
        "clinical development phase I-II" : "clinical development phase I-II",
        "no comparator open label" : "no comparator open label",
        "primary endpoint" : "no comparator with primary endpoint is change in seizures frequency",
        "secondary endpoints" : "econdary endpoints are neurodevelopment scales (VINELAND II, GMFCS and GMFM-88), standard safety monitoring"
     
}
keywords = [
        "rare disease in pediatric neurologic disease",
        "Main symptoms are epileptic seizure and loss of functional skills (motor, language)", 
        "Treatment is administered intrathecally with requirement of 2 days of hospitalization.","early phase clinical trial",
        "Phase I/II study",
        "open-label trial",
        "pediatric neurologic condition",
        "rare neurological disorder children",
        "epileptic seizures",
        "clinical development phase I-II",
        "no comparator open label",
        "primary endpoint",
        "secondary endpoints"
        ]

#plus petit chunk size
#gpt => demander a optimiser le contexte base avec forte recurrence de mots techniques
#postSql bgvector
#faire degager les criteres qui ont de tros de reccurent
#chunk par section : dans ma db je donne un poids pour chaque section => et quand je fais requete je met plus de valeur 
def check_answer(answer, keyword):

    similarity = 0

    if keyword in context:
       
       context_embedding = model.encode(context[keyword])
       answer_embedding = model.encode(answer)
       similarity = util.pytorch_cos_sim(context_embedding, answer_embedding)
       similarity = similarity.item()

    return answer, similarity
def chat_with_gpt3(prompt):

    response = openai.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                 messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                ],
                temperature=0, #pour eviter hallucination
                max_tokens=1000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

    return response.choices[0].message.content.strip()

def rephrase(answer):
    prompt = "Rephrase the answer, keep all the information" + answer
    response = openai.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                 messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                ],
                temperature=0.6,
                max_tokens=1000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

    return response.choices[0].message.content.strip()

def get_keywords(keyword,retriever):
     
    try :
        docs = retriever.get_relevant_documents(keyword, k=2)
        print(f"Keyword: {keyword}, Documents Found: {len(docs)}")
        return [doc.page_content for doc in docs]
    except:
        return "No documents found"
          
     


def process_document(file_path):
    # Initialize the PDF loader and text splitter
    #if error not a pdf file

    

    if not file_path.lower().endswith('.pdf'):
        return f"Skipped non-PDF file: {file_path}"

    try:
        # Initialize the PDF loader and text splitter
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    except Exception as e:
        return f"Failed to load PDF: {e}"


    text_splitter = CharacterTextSplitter(separator="\n",chunk_size=256)
    document_fragments = text_splitter.split_documents(documents)
    # Extract texts and generate embeddings
    texts = [doc.page_content for doc in document_fragments]

    emb_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=emb_model,
        cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME')
    )
    

    #embeddings = HuggingFaceEmbeddings()
    
  

    # Initialize Chroma with embeddings
    db = Chroma.from_documents(document_fragments, embedding=embeddings)
    retriever = db.as_retriever(
    search_type="mmr"
    )

    # Define keywords and retrieve relevant text
    
    
    
    

    #Use case de questionner : passe les chunk dans un prompt => important de overlap => toujours lui en envoyer 2=> generer beaucoup de question et de la reponse et lie reponse => 
    #Usecase summarize => trouver l'equation
    #Finetune 
    """

    keywords = [
    "early phase clinical trial",
    "Phase I/II study",
    "open-label trial",
    "pediatric neurologic condition",
    "rare neurological disorder children",
    "epileptic seizures",
    "motor skill loss",
    "language skill deterioration",
    "functional skill loss",
    "treatment administration",
    "hospitalization requirement",
    "two-day hospital stay",
    "incidence rate pediatric",
    "annual diagnosis rate",
    "early mortality rate",
    "clinical development phase I-II",
    "no comparator open label",
    "primary endpoint seizure frequency",
    "secondary endpoints neurodevelopmental",
    ]

    keywords = [
        "Therapeutic Indication", "Orphan drug","Chronic indication","Acute indication","rare pediatric disease","Main symptoms","rare pediactric","Mode of administration","Clinical assessment","Market Size/incidence"
    ]

    """
    #summarizer = pipeline("summarization", model="google/bigbird-pegasus-large-pubmed")

    results = {}
    results1 = {}

    
    for keyword in keywords:
        
        results[keyword] = get_keywords(keyword,retriever)

        #memory
        results1[keyword] = results[keyword]

        if keyword=="rare disease in pediatric neurologic disease":
            mem = [0]
            sims = [0]
            count = 0
            prompt = "Please provide in one sentence a description of the condition Information source: " + results[keyword][0] + " " + results[keyword][1] + " " + results[keyword][2] + " " + results[keyword][3]
            results[keyword] = chat_with_gpt3(prompt)
            while min(sims) < 0.8:
                
                answer, similarity = check_answer(results[keyword], keyword)
                mem.append(answer)
                
                sims.append(similarity)
                count += 1

                results[keyword] = rephrase(answer)
                if count == 4:
                    break
            print(mem)
            print(sims)
            index= sims.index(max(sims))
            results[keyword] = mem[index]
    



        if keyword=="Main symptoms are epileptic seizure and loss of functional skills (motor, language)":
             
            prompt = "Please provide in one sentence description the symptoms Information source: " + results[keyword][0] + " " + results[keyword][1] + " " + results[keyword][2] + " " + results[keyword][3]
            results[keyword] = chat_with_gpt3(prompt)
            mem = [0]
            sims = [0]
            count = 0

            while min(sims) < 0.4:

                answer, similarity = check_answer(results[keyword], keyword)
                mem.append(answer)
                
                sims.append(similarity)
                count += 1

                results[keyword] = rephrase(answer)
                if count == 4:
                    break


            index= sims.index(max(sims))
            results[keyword] = mem[index]



        if keyword=="Treatment is administered intrathecally with requirement of 2 days of hospitalization.":
             
            prompt = "Please provide in one sentence description of treatment Information source: " + results[keyword][0] + " " + results[keyword][1] + " " + results[keyword][2] + " " + results[keyword][3]
            results[keyword] = chat_with_gpt3(prompt)

            mem = [0]
            sims = [0]
            count = 0

            while min(sims) < 0.4:
                    
                    answer, similarity = check_answer(results[keyword], keyword)
                    mem.append(answer)
                    
                    sims.append(similarity)
                    count += 1
    
                    results[keyword] = rephrase(answer)
                    if count == 4:
                        break

            index= sims.index(max(sims))
            results[keyword] = mem[index]

        if keyword=="early phase clinical trial":
             
            prompt = "Please provide in one sentence description of early phase clinical trial Information source: " + results[keyword][0] + " " + results[keyword][1] + " " + results[keyword][2] + " " + results[keyword][3]
            results[keyword] = chat_with_gpt3(prompt)




        if keyword=="Phase I/II study":
             
            prompt = "Please provide in one sentence description of Phase I/II study Information source: " + results[keyword][0] + " " + results[keyword][1] + " " + results[keyword][2] + " " + results[keyword][3]
            results[keyword] = chat_with_gpt3(prompt)

        if keyword=="open-label trial":
             
            prompt = "Please provide in one sentence description of open-label trial Information source: " + results[keyword][0] + " " + results[keyword][1] + " " + results[keyword][2] + " " + results[keyword][3]
            results[keyword] = chat_with_gpt3(prompt)

        if keyword=="pediatric neurologic condition":

            prompt = "Please provide a brief description of condition, including the symptoms Information source: " + results[keyword][0] + " " + results[keyword][1] + " " + results[keyword][2] + " " + results[keyword][3]
            results[keyword] = chat_with_gpt3(prompt)

            mem = [0]
            sims = [0]
            count = 0

            while min(sims) < 0.4:
                    
                    answer, similarity = check_answer(results[keyword], keyword)
                    mem.append(answer)
                    
                    sims.append(similarity)
                    count += 1
    
                    results[keyword] = rephrase(answer)
                    if count == 4:
                        break

            index= sims.index(max(sims))
            results[keyword] = mem[index]



        if keyword=="rare neurological disorder children":
                
            prompt = "Please provide a brief description of the disorder state, which type of population the disordor affect and the rarity Information source: " + results[keyword][0] + " " + results[keyword][1] + " " + results[keyword][2] + " " + results[keyword][3]
            results[keyword] = chat_with_gpt3(prompt)

            mem = [0]
            sims = [0]
            count = 0

            while min(sims) < 0.4:
                    
                    answer, similarity = check_answer(results[keyword], keyword)
                    mem.append(answer)
                    
                    sims.append(similarity)
                    count += 1
    
                    results[keyword] = rephrase(answer)
                    if count == 4:
                        break
            print(mem)
            print(sims)
            index= sims.index(max(sims))
            results[keyword] = mem[index]

        if keyword=="epileptic seizures":
                    
            prompt = "Please provide a brief description of the type of seizures, if there is Information source: " + results[keyword][0] + " " + results[keyword][1] + " " + results[keyword][2] + " " + results[keyword][3]
            results[keyword] = chat_with_gpt3(prompt)

        if keyword=="clinical development phase I-II":
             
            prompt = "Please provide in one sentence description of clinical development phase  Information source: " + results[keyword][0] + " " + results[keyword][1] + " " + results[keyword][2] + " " + results[keyword][3]
            results[keyword] = chat_with_gpt3(prompt)

        if keyword=="comparator open label":
             
            prompt = "Please provide in one sentence description of the comparator open label Information source: " + results[keyword][0] + " " + results[keyword][1] + " " + results[keyword][2] + " " + results[keyword][3]
            results[keyword] = chat_with_gpt3(prompt)

        if keyword=="primary endpoint":
                 
                prompt = "Please provide in one sentence description of primary endpoint, give all key information Information source: " + results[keyword][0] + " " + results[keyword][1] + " " + results[keyword][2] + " " + results[keyword][3]
                results[keyword] = chat_with_gpt3(prompt)
                mem = [0]
                sims = [0]
                count = 0


                while min(sims) < 0.4:
                    
                    answer, similarity = check_answer(results[keyword], keyword)
                    mem.append(answer)
                    
                    sims.append(similarity)
                    count += 1
    
                    results[keyword] = rephrase(answer)
                    if count == 4:
                        break

                index= sims.index(max(sims))
                results[keyword] = mem[index]
        if keyword=="secondary endpoints":
             
            prompt = "Please provide in one sentence description of secondary endpoints, give all key information Information source: " + results[keyword][0] + " " + results[keyword][1] + " " + results[keyword][2] + " " + results[keyword][3]
            results[keyword] = chat_with_gpt3(prompt)

            mem = [0]
            sims = [0]
            count = 0


            while min(sims) < 0.4:
                    
                    answer, similarity = check_answer(results[keyword], keyword)
                    mem.append(answer)
                    
                    sims.append(similarity)
                    count += 1
    
                    results[keyword] = rephrase(answer)
                    if count == 4:
                        break   

            index= sims.index(max(sims))
            results[keyword] = mem[index] 
        #empty db

    for collection in db._client.list_collections():
            ids = collection.get()['ids']
            print('REMOVE %s document(s) from %s collection' % (str(len(ids)), collection.name))
            if len(ids): 
                collection.delete(ids)

  
    # Save results to a JSON file

    with open(f'{os.path.splitext(os.path.basename(file_path))[0]}_info2.json', 'w') as json_file:
        json.dump(results, json_file, indent=4, ensure_ascii=False)
    
    with open(f'{os.path.splitext(os.path.basename(file_path))[0]}_info4.json', 'w') as json_file:
        json.dump(results1, json_file, indent=4, ensure_ascii=False)
    
    return f"Information extracted and saved for {file_path}"

if __name__ == "__main__":
    
    #do all the files in the folder
    #for ten files

    file_path = "files/document-ta614.pdf"
    print(process_document(file_path))

    """
    
    file_path = "files/document-hst12.pdf"
    print(process_document(file_path))

    file_path = "files/document-ta808.pdf"
    print(process_document(file_path))
   
    #for all files in the folder

    filenames = os.listdir("filestest")
    print(len(filenames))
    count = 0


  
    for filename in filenames:
    
            file_path = os.path.join("filestest", filename)
            print(file_path)
            print(process_document(file_path))

   """