# Standard library imports
import os
import logging
from pathlib import Path

# Third-party imports
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from batch_embed import EmbeddingQueue
from doc_retrieval import doc_retriever
from utils import format_docs, get_metadata_from_docs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure paths
BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_PATH = os.path.join(BASE_DIR, "VECTOR_DB")


app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({
        "status": "success",
        "message": "Server is running"
    }), 200

@app.route('/add_vectors', methods=['POST'])
def add_vectors():
    try:
        data = request.get_json()
        
        # Get embeddings and document from request
        embeddings_data = data['embeddings']
        normalized_doc = data['normalized_doc']
        total_tokens_count = data['total_tokens_count']
        max_token_per_min = data['max_token_per_min']
        user_id = data['user_id']
        doc_id = data['doc_id']
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        split_docs = text_splitter.create_documents(normalized_doc)
        
        split_documents_with_metadata = [
            Document(page_content=document.page_content, metadata={"user_id": user_id, "doc_id": doc_id})
            for document in split_docs
        ]
        
        split_docs_length = len(split_documents_with_metadata)
        
        embeddings = {
            "model": embeddings_data['model'],
            "openai_api_key": embeddings_data['openai_api_key']
        }   
        
        # Create OpenAIEmbeddings instance
        embedding_function = OpenAIEmbeddings(
            model=embeddings['model'],
            openai_api_key=embeddings['openai_api_key']
        )

        # Calculate batch size based on token count
        if total_tokens_count > max_token_per_min:
            batch_size = split_docs_length // ((total_tokens_count // max_token_per_min) + 1)
        else:
            batch_size = split_docs_length
        
        # Initialize and start the queue
        embedding_queue = EmbeddingQueue(
            max_tokens_per_min=max_token_per_min,
            vectorstore_path=VECTORSTORE_PATH,
            embeddings=embedding_function,
            max_workers=3
        )

        try:
            processor_thread = embedding_queue.start_processing(split_documents_with_metadata, batch_size)
            processor_thread.join()  # Wait for all processing to complete
            
            if embedding_queue.processing_complete:
                logger.info("All documents have been processed and stored in the vector database!")
                return jsonify({
                    "status": "success",
                    "message": "All documents have been processed and stored in the vector database"
                }), 200
            return jsonify({
                "status": "error",
                "message": "Processing was not completed successfully"
            }), 500
            
        except KeyboardInterrupt:
            logger.warning("Processing was interrupted")
            embedding_queue.stop()
            processor_thread.join()
            return jsonify({
                "status": "error",
                "message": "Processing was interrupted"
            }), 500
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            embedding_queue.stop()
            processor_thread.join()
            return jsonify({
                "status": "error",
                "message": f"Error during processing: {str(e)}"
            }), 500
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route('/retrieve_documents', methods=['POST'])
def retrieve_documents():
    try:
        data = request.get_json()
        
        # Extract parameters from request
        query = data['query']
        user_id = data.get('user_id')
        document_id_list = data.get('document_id_list')
        score_threshold = data.get('score_threshold', 0.7)  # Default threshold
        k_value = data.get('k_value', 5)  # Default number of results
        openai_api_key = data['openai_api_key']
        embeddings_model = data['embeddings_model']
        chat_model = data['chat_model']
        
        # Call the document retriever
        results = doc_retriever(
            query=query,
            user_id=user_id,
            document_id_list=document_id_list,
            score_threshold=score_threshold,
            k_value=k_value,
            openai_api_key=openai_api_key,
            embeddings_model=embeddings_model,
            vectorstore_path=VECTORSTORE_PATH,
            chat_model=chat_model
        )
        
        retrieved_docs = results['results']
        
        try:
            context = format_docs(retrieved_docs)
            context_metadata = get_metadata_from_docs(retrieved_docs)
        except Exception as e:
            logger.error(f'No relevant docs were retrieved: {str(e)}')
            context = "No relevant docs were retrieved"
            context_metadata = "No data retrieved"
        
        return jsonify({
            "status": "success",
            "context": context,
            "context_metadata": context_metadata
        }), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error in retrieve_documents: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=2100, use_reloader=False)
