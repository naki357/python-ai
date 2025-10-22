from pyexpat import model
import pandas as pd
import numpy as np
from sklearn import datasets, model_selection, metrics
import torch
import tensorflow as tf
import igraph as ig
from transformers import pipeline
from langchain_community.llms import OpenAI
#import openai


mongoCloudURI = "mongodb+srv://fullmongo:q7xe7WYjN5zwkC3@mongocluster0.skcwy.mongodb.net/?retryWrites=true&w=majority&appName=MongoCluster0"

# Data analysis example
def data_analysis():
    df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
    print('DataFrame head:')
    print(df.head())

# Machine learning example
def sklearn_example2(start_message):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_wine # Example dataset
    from sklearn.metrics import accuracy_score

    print(start_message)

    # Load a dataset (e.g., Iris dataset)
    dataset = load_wine()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)

    # Create and train a Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=50, random_state=40)
    rf_classifier.fit(X_train, y_train)

    # Make predictions and evaluate accuracy
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    from sklearn.tree import export_graphviz
    import pydotplus
    from IPython.display import Image # For displaying the image in Jupyter notebooks

    # Select one tree from the forest (e.g., the first tree)
    estimator = rf_classifier.estimators_[10]

    # Export the decision tree to a DOT file
    dot_data = export_graphviz(estimator,
        out_file=None,
        feature_names=dataset.feature_names,
        class_names=dataset.target_names,
        filled=True, rounded=True,
        special_characters=True)

    # Create a graph from the DOT data and render it as a PNG image
    graph = pydotplus.graph_from_dot_data(dot_data)
    # Image(graph.create_png())
    Image(graph.write_png("wine_tree10.png"))

    # Get feature importances
    importances = rf_classifier.feature_importances_
    indices = np.argsort(importances)[::-1] # Sort in descending order

    # Plot feature importances
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.show()

# Deep learning example (PyTorch)
def pytorch_example():
    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self, in_features=13, h1=8, h2=9, out_features=3):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(in_features, h1)
            self.fc2 = nn.Linear(h1, h2)
            self.out = nn.Linear(h2, out_features)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.out(x)

            return x
        
    torch.manual_seed(42)  # For reproducibility
    model = Model()

    # Load the wine dataset 
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split

    wineSet = load_wine()
    X = torch.tensor(wineSet.data, dtype=torch.float32)
    y = torch.tensor(wineSet.target, dtype=torch.long)
    # Example forward pass
    output = model(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    epochs = 100
    losses = []
    model.train()
    for epoch in range(epochs):  # Example: 100 epochs
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()    
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

    import matplotlib.pyplot as plt
    plt.plot(range(epochs), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    print('Input shape:', X.shape)
    print('Output shape:', output.shape)
    print('PyTorch model created with architecture:', model)

# Deep learning example (TensorFlow)
def tensorflow_example():
    print('TensorFlow version:', tf.__version__)

# Hugging Face Transformers example
def hf_example():
    classifier = pipeline('sentiment-analysis')
    classifier.model.config.id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    # Example usage
    result = classifier('Everything here sucks and I want to go home.  This is ' \
    'kind of not ok.')
    print(result)

    # from vllm import LLM
    # from ocrflux.inference import parse

    # file_path = 'test.pdf'
    # # file_path = 'test.png'
    # llm = LLM(model="model_dir/OCRFlux-3B",gpu_memory_utilization=0.8,max_model_len=8192)
    # result = parse(llm,file_path)
    # document_markdown = result['document_text']
    # with open('test.md','w') as f:
    #     f.write(document_markdown) 

# LangChain + OpenAI example
def langchain_openai_example():
    # This requires OPENAI_API_KEY to be set in your environment
    llm = OpenAI()
    print(llm('What are the latest trends in AI?'))

def object_detection_example():
    from transformers import DetrImageProcessor, DetrForObjectDetection
    import torch
    from PIL import Image
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # url = "https://www.istockphoto.com/photo/administration-teamwork-office-documents-or-people-review-financial-data-finance-gm1473508665-503623178"
    image = Image.open(requests.get(url, stream=True).raw)

    # you can specify the revision tag if you don't want the timm dependency
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        )

def Google_Gemini_example():
    # This requires GEMINI_API_KEY to be set in your environment
    from google import genai
    # from google.genai import types

    # import google.generativeai as genai
    # from google.generativeai import types
    
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client()
    client.max_output_tokens = 100

    # Test Gemini's image understanding capabilities
    with open('./images/wall-st-1.webp', 'rb') as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        content=[
            genai.types.Content(
                image=genai.types.Image(
                    image_bytes=image_bytes,
                    mime_type='image/webp'
                )
            ),
            genai.types.Content(
                text='What is in this image?'
            )
        ],
    )

    print(response.text)

def RAG_Generate_Embeddings():
    # This function is a placeholder for a future RAG example using MongoDB
    # First connect to MongoDB
    from pymongo.mongo_client import MongoClient
    from pymongo.server_api import ServerApi
    
    # Create a new client and connect to the server
    client = MongoClient(mongoCloudURI, server_api=ServerApi('1'))
    db = client["sample_restaurants"]
    rest_collection = db["restaurants"]
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Define a function that generates embeddings for a piece of data
    def generate_embedding(data, precision="float32"):
        return model.encode(data, convert_to_tensor=True)

    # Iterate through the collection and generate embeddings for movie title, cast, genre and directors
    # embeddings = []
    iCount = 0

    for restaurant in rest_collection.find().skip(0):
        restaurantId = restaurant.get("_id")
        name = restaurant.get("name", "")
        cuisine = restaurant.get("cuisine", "")

        # Generate embeddings for each field
        name_embedding = generate_embedding(name)
        cuisine_embedding = generate_embedding(cuisine)

        queryClause = {"_id": restaurantId}
        setClause = {
            "$set": 
            { "name_embedding": name_embedding.tolist(), 
             "cuisine_embedding": cuisine_embedding.tolist() } 
        }

        # Perform the update_one operation
        rest_collection.update_one(queryClause, setClause)

        iCount += 1
        # Print out the iteration number every 50 movies
        if iCount % 25 == 0:
            print(f"Created embeddings for {iCount} restaurants.")

def RAG_Run_Query():
    # This function is a placeholder for a future RAG example using MongoDB
    from pymongo.mongo_client import MongoClient
    from pymongo.server_api import ServerApi
    from sentence_transformers import SentenceTransformer, util
    from langchain_core.documents import Document
    from typing_extensions import List, TypedDict
    from langchain import hub
    # from langchain_core.llms import LLM
    from langgraph.graph import START, StateGraph
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DistilBertTokenizer, DistilBertForQuestionAnswering
    from IPython.display import Image, display

    uri = "mongodb+srv://fullmongo:q7xe7WYjN5zwkC3@mongocluster0.skcwy.mongodb.net/?retryWrites=true&w=majority&appName=MongoCluster0"
    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client["sample_restaurants"]
    collection = db["restaurants"]

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    # Define the query embedding 
    # This embedding should be generated using the same model (Voyage AI's voyage-3-large)
    # that was used to embed the 'plot_embedding_voyage_3_large' field in the database.
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with your model if different

    # Allow the user to continuously input queries
    print("Follow the prompts to search the restaurant listings. Type 'exit' to quit.")
    while True:
        # First get the type of query which will determine the path of the embedding field to use
        query_type = input("Enter the type of query (name/cuisine): ")
        if query_type.lower() == 'name':
            query_path = "name_embedding"
        elif query_type.lower() == 'cuisine':
            query_path = "cuisine_embedding"
        else:
            query_type = "name"
            query_path = "name_embedding"

        print(f"Searching against: {query_path}") 

        # Get the query from the user
        query = input(f"Describe the restaurant {query_type} that you'd like to find: ")
        if query.lower() == 'exit':
            break
        query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=True)
        query_embedding = query_embedding.tolist()  # Convert to list for MongoDB compatibility

        # region Rankfusion (combining the results of searches against various query paths)
        # Rankfusion is not currently supported in my MongoDB Atlas free tier, so using single vector search instead
        # pipeline1 = [
        #     {
        #         "$vectorSearch": {
        #             "queryVector": query_embedding,
        #             "dim": 384,  # Dimension of the embeddings
        #             "path": "plot_embedding",  # Use the path based on the query type
        #             "similarity": "cosine",  # Use cosine similarity
        #             "numCandidates": 150,  # Number of nearest neighbors to consider
        #             "limit": 10,           # Number of documents to return in the results
        #             "index": "idx-movies-embeddings" # Name of your vector search index
        #         }
        #     }
        # ]
        # pipeline2 = [
        #     {
        #         "$vectorSearch": {
        #             "queryVector": query_embedding,
        #             "dim": 384,  # Dimension of the embeddings
        #             "path": "title_embedding",  # Use the path based on the query type
        #             "similarity": "cosine",  # Use cosine similarity
        #             "numCandidates": 150,  # Number of nearest neighbors to consider
        #             "limit": 10,           # Number of documents to return in the results
        #             "index": "idx-movies-embeddings" # Name of your vector search index
        #         }
        #     }
        # ]
        # pipeline3 = [
        #     {
        #         "$vectorSearch": {
        #             "queryVector": query_embedding,
        #             "dim": 384,  # Dimension of the embeddings
        #             "path": "cast_embedding",  # Use the path based on the query type
        #             "similarity": "cosine",  # Use cosine similarity
        #             "numCandidates": 150,  # Number of nearest neighbors to consider
        #             "limit": 10,           # Number of documents to return in the results
        #             "index": "idx-movies-embeddings" # Name of your vector search index
        #         }
        #     }
        # ]

        # pipeline = [
        #     {
        #         "$rankFusion": {
        #             "input": [
        #                 {"pipeline": [pipeline1], "weight": 0.4}, # Adjust weights as needed
        #                 {"pipeline": [pipeline2], "weight": 0.3},
        #                 {"pipeline": [pipeline3], "weight": 0.3}
        #             ]
        #         }
        #     },
        #     # Add any further aggregation stages like $project, $limit, etc.
        #     {
        #         "$project": {
        #             "title": 1,
        #             "plot": 1,
        #             "genres": 1,
        #             "cast": 1,
        #             "directors": 1,
        #             "score": { "$meta": "vectorSearchScore" } # Include the search score
        #         }
        #     }
        # ]
        
        # Run a vector search query across multiple paths using the rankfusion operator
        # movies = list(collection.aggregate(pipeline))
        # endregion

        # Perform the Atlas Vector Search query based on the query type
        restaurants = collection.aggregate([
            {
                "$vectorSearch": {
                    "queryVector": query_embedding,
                    "dim": 384,
                    "path": query_path,  # Use the path based on the query type-
                    "numCandidates": 150,  # Number of nearest neighbors to consider
                    "limit": 10,           # Number of documents to return in the results
                    "index": "vector_index_restaurants" # Name of your vector search index
                }
            },
            {
                "$project": {
                    "address": 1,
                    "borough": 1,
                    "name": 1,
                    "cuisine": 1,
                    "score": { "$meta": "vectorSearchScore" } # Include the search score
                }
            }
        ])

        # Create a State-typed dictionary for the current query
        queryState: State = {"question": query, "context": restaurants.to_list(), "answer": ""}

        restCount = 0
        for restaurant in queryState["context"]:
             restCount+=1
             print(f"Match({restCount}): ")
             print(f"Restaurant: {restaurant['name']}")
             print(f"Cuisine: {restaurant['cuisine']}")
             print(f"Borough: {restaurant['borough']}")
             print(f"Query match score: {restaurant['score']}\n")
             # Print out the address if found:
             if 'address' in restaurant:
                currAddress = restaurant["address"]
                print(f"Street Address: {currAddress['building']} {currAddress['street']}")
                print(f"Zip Code: {currAddress['zipcode']}")
             
        # Now add reranking and work with an LLM to generate a more refined response
        prompt = hub.pull("rlm/rag-prompt")

        def retrieve(state: State):        
            return {"context": queryState["context"]}

        def generate(state: State):
            result_content = ",".join(restaurant["name"] for restaurant in queryState["context"])
            messages = prompt.invoke({"question": queryState["question"], "context": queryState["context"]})
            # llm = Anthropic(model="claude-2")
            llm_name = "distilbert-base-uncased-distilled-squad"
            tokenizer = AutoTokenizer.from_pretrained(llm_name)
            llm = DistilBertForQuestionAnswering.from_pretrained(llm_name)
            context = result_content
            inputs = tokenizer(query, context, return_tensors="pt")
            with torch.no_grad():
                outputs = llm(**inputs)

            answer_start_index = torch.argmax(outputs.start_logits)
            answer_end_index = torch.argmax(outputs.end_logits)
            if(answer_end_index<answer_start_index):
                # The start shouldn't have exceeded the end so we flip the values.
                tempStart = answer_start_index
                answer_start_index = answer_end_index
                answer_end_index = tempStart

            predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
            response = tokenizer.decode(predict_answer_tokens)

            return {"answer": response}

        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()     

        result = graph.invoke({"question": query})

        display(Image(graph.get_graph().draw_png()))

        # print(f"Context: {result['context']}\n\n")
        print(f"Answer: {result['answer']}")

def google_drive_analysis():
    import os.path

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    # Authenticate and create the service
    SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly"]

    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
            "./googleauth/credentials2.json", SCOPES
        )
        creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        service = build("drive", "v3", credentials=creds)

        # Call the Drive v3 API
        results = (
            service.files()
            .list(pageSize=10, fields="nextPageToken, files(id, name)")
            .execute()
        )
        items = results.get("files", [])

        # List files in Google Drive from a specific folder
        folder_id = '1zqwQZTskfus34lWXujYEDpU59H3yaH86'  # Replace with your folder ID
        query = f"'{folder_id}' in parents"
        results = service.files().list(q=query, pageSize=10, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            print('No files found.')
        else:
            print('Files:')
            for item in items:
                print(f"{item['name']} ({item['id']})")
    except HttpError as error:
        print(f"An error occurred: {error}")

def load_google_docs():
    import pandas as pd
    import pymongo
    import os

    # MongoDB connection details
    MONGO_URI = mongoCloudURI
    DATABASE_NAME = "playground"
    COLLECTION_NAME = "openmic_playlists"

    # Directory containing your XLS files
    XLS_FILES_DIR = "./data/Playlists/"

    def import_xls_to_mongodb(xls_file_path, collection):
        """Reads an XLS file and inserts its data into a MongoDB collection."""
        try:
            df = pd.read_excel(xls_file_path)
            df["source_file"] = xls_file_path
            data_to_insert = df.to_dict('records')
            if data_to_insert:
                collection.insert_many(data_to_insert)
                print(f"Successfully imported data from {xls_file_path}")
            else:
                print(f"No data found in {xls_file_path}")
        except Exception as e:
            print(f"Error importing {xls_file_path}: {e}")

    if __name__ == "__main__":
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]

        # Iterate through XLS files in the specified directory
        for filename in os.listdir(XLS_FILES_DIR):
            if filename.endswith(".xls") or filename.endswith(".xlsx"):
                xls_file_path = os.path.join(XLS_FILES_DIR, filename)
                import_xls_to_mongodb(xls_file_path, collection)

        client.close()
        print("Batch import process completed.")

if __name__ == "__main__":
    # sklearn_example2("Run sklearn example with Random Forest Classifier")
    # data_analysis()
    # sklearn_example()
    # pytorch_example()
    # tensorflow_example()
    # hf_example()
    # object_detection_example()
    # Google_Gemini_example()
    # RAG_Generate_Embeddings()
    RAG_Run_Query()
    # google_drive_analysis()
    # load_google_docs()
    # web_scraping_example()
    # Uncomment below if you have OpenAI API key setup
    #langchain_openai_example()
    print("All examples executed successfully.")