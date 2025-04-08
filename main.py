import os
import time 
import logging 
import traceback 
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path 
import streamlit as st 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler("app.log")
                    ])

logger = logging.getLogger(__name__)

VECTOR_STORE_PATH = Path("./vector_store/faiss")
DOCUMENTS_PATH = Path("./Documents")
DEFAULT_PDF_PATH = DOCUMENTS_PATH / "TBDY 2018.pdf"
EMBEDDINGS_MODEL = "text-embedding-3-small"
LLM_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
RETRIEVER_K = 5

class ConfigManager:
    @staticmethod
    def load_environment_variables():
        api_keys = {}
        required_keys = ["OPENAI_API_KEY", "GROQ_API_KEY"]
        missing_api_keys = []

        for key in required_keys:
            value = st.secrets.get(key)
            if not value:
                missing_api_keys.append(key)
            api_keys[key.lower()] = value
        
        if missing_api_keys:
            error_msg = f"Missing API key: {', '.join(missing_api_keys)}"
            raise ValueError(error_msg)
        
        logger.info("Environment variables loaded successfully")
        return api_keys
    

class DocumentProcessor:
    def __init__(self, file_path: str, openai_api_key: str):
        self.file_path = file_path
        self.openai_api_key = openai_api_key
        self.text_splitter = RecursiveCharacterTextSplitter(
            CHUNK_SIZE, CHUNK_OVERLAP
        )

    def load_documents(self) -> List[Document]:
        """Load documents from file path"""
        documents = []
        try:
            logger.info(f"Loading documents from {self.file_path}")

            if not os.path.exists(self.file_path):
                logger.error(f"File not found: {self.file_path}")
                raise FileNotFoundError(f"File not found: {self.file_path}")
            
            document_loader = PyPDFLoader(self.file_path)
            documents = document_loader.load()
            logger.info(f"Documents successfully loaded. {len(documents)} pages from, {self.file_path}")
        
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            logger.debug(traceback.format_exc())
            raise e
        
        return documents
    
    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            logger.info("Creating chunks from documents")
            processed_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Chunks created successfully. {len(processed_docs)} chunks")
            return processed_docs
        
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            logger.debug(traceback.format_exc())
            raise e
        
    def get_embeddings(self):
        """Get OpenAI Embeddings"""
        try:
            return OpenAIEmbeddings(
                api_key=self.openai_api_key,
                model=EMBEDDINGS_MODEL
            )
        
        except Exception as e:
            logging.error(f"Error creating embeddings: {str(e)}")
            logging.debug(traceback.format_exc())
            raise e
        
class VectorStoreManager:
    """Manages vector store operations"""

    def __init__(self, openai_api_key: str, file_path: str = str(DEFAULT_PDF_PATH)):
        self.openai_api_key = openai_api_key
        self.file_path = file_path
        self.document_processor = DocumentProcessor(file_path, openai_api_key)
        self.last_index_time = None
        self.last_doc_mod_time = None

    def create_vector_store(self):
        """Create vector store"""

        try:
            logger.info("Creating vector store")
            os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)

            documents = self.document_processor.load_documents()
            if not documents:
                error_msg = "No documents Loaded"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            processed_docs = self.document_processor.create_chunks(documents)
            if not processed_docs:
                error_msg = "No chunks created"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            embeddings = self.document_processor.get_embeddings()
            vector_store = FAISS.from_documents(
                processed_docs, embeddings
            )

            vector_store.save_local(str(VECTOR_STORE_PATH))
            logger.info(f"Vector store saved to {VECTOR_STORE_PATH}")
            return vector_store
        
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            logger.debug(traceback.format_exc())
            raise e
        
    @staticmethod
    def load_vector_store(openai_api_key:str):
        """Load vector store from file"""
        try:
            logger.info(f"Loading vector store from {VECTOR_STORE_PATH}")

            if not os.path.exists(VECTOR_STORE_PATH):
                logger.warning(f"Vector store not found at {VECTOR_STORE_PATH}")
                return None

            embeddings = OpenAIEmbeddings(
                api_key=openai_api_key,
                model=EMBEDDINGS_MODEL
            )

            vector_store = FAISS.load_local(str(VECTOR_STORE_PATH), embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Vector store loaded from {VECTOR_STORE_PATH}")
            return vector_store
        
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
        
    def get_or_create_vector_store(self) -> FAISS:
        """Get existing vector store or create a new one if needed"""
        # Check if documents have been modified
        current_mod_time = os.path.getmtime(self.file_path)
        vector_store = self.load_vector_store(self.openai_api_key)
        
        if vector_store is None:
            logger.info("Creating new vector store as existing one was not found")
            vector_store = self.create_vector_store()
            self.last_index_time = time.time()
            self.last_doc_mod_time = current_mod_time
        elif self.last_doc_mod_time is None or current_mod_time > self.last_doc_mod_time:
            logger.info("Documents have been modified - recreating vector store")
            vector_store = self.create_vector_store()
            self.last_index_time = time.time()
            self.last_doc_mod_time = current_mod_time
        else:
            logger.info("Using existing vector store")
            
        return vector_store    
    
class RetrievalChain:
    """Chain of retrieval models"""
    def __init__(self, vector_store: FAISS, groq_api_key: str):
        self.vector_store = vector_store
        self.groq_api_key = groq_api_key
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k":RETRIEVER_K}
        )

        try:
            self.llm = ChatGroq(
                model=LLM_MODEL,
                api_key=self.groq_api_key,
                temperature=0.1,
            )
            logger.info(f"LLM initialized with model: {LLM_MODEL}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            logger.debug(traceback.format_exc())
            raise 

    def create_prompt(self):
        """Create prompt for LLM"""
        prompt_text = """
                    İnşaat Mühendisliği alanında araştırmacılık tecrübesine sahip bir assistansın.
                    Kullanıcılar, inşaat yönetmelikleri ve inşaat yapımıyla ilgili teknik sorular sorabilir.
                    Yönetmeliklerden alıntılanan aşağıdaki metni kullanarak cevap ver.
                    Kullanıcı sorusu: {input}

                    <context>
                    {context}
                    <context>

                    - Sadece doğru ve güvenilir bilgiler ver. 
                    - Uydurma veya yanlış bilgiler verme.
                    - Eğer cevap veremiyorsan, lütfen bilmediğini açıkça belirt.
                    """
        return ChatPromptTemplate.from_template(prompt_text)
        
    def create_chain(self):
        """Create retrieval chain"""
        try:
            prompt = self.create_prompt()
            llm_chain = prompt | self.llm

            chain = create_retrieval_chain(
                retriever=self.retriever,
                combine_docs_chain=llm_chain
            )
            logger.info("Retrieval chain created")
            return chain

        except Exception as e:
            logger.error(f"Error creating prompt: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    def get_answer(self, question: str):
        """Get answer from LLM"""
        try:
            chain = self.create_chain()
            response = chain.invoke({"input":question})
            answer = response.get('answer', "Üzgünüm, bu soruya cevap veremiyorum.")
            source_docs = response.get('source_documents', [])
            logger.info(f"Answer generated successfully.")
            return answer, source_docs
        
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

class StreamlitApp:
    """Streamlit app interface"""
    def __init__(self, retrieval_chain: RetrievalChain):
        self.retrieval_chain = retrieval_chain

    def display_answer(self, answer: str, source_docs: list):
        st.markdown("### Cevap")
        st.write(answer.content)

        if source_docs and st.checkbox("Kaynak Dokümanları Göster"):
            st.markdown("### Kaynak Dokümanlar")
            for i, doc in enumerate(source_docs):
                with st.expander(f"Kaynak {i+1}"):
                    st.write(doc.content)
                    if hasattr(doc, 'metadata'):
                        st.write(f"Sayfa: {doc.metadata.get('page', 'Bilinmiyor')}")

    
    def run(self):
        st.title("CengAI")
        st.markdown("""
        Bu uygulama, Türk Bina Deprem Yönetmeliği 2018'e göre sorularınızı cevaplamaktadır.
        Sorularınızı Türkçe olarak yazabilirsiniz.
        """)

        with st.form("question_form"):
            user_question = st.text_input("Sorunuzu buraya yazın:",
                placeholder="Örnek: TBDY 2018'e göre deprem tasarım sınıfları nelerdir?"
            )
            submit_button = st.form_submit_button(label="Soru Sor")

        if submit_button and user_question:
            try:
                with st.spinner("Cevap hazırlanıyor..."):
                    answer, source_doc = self.retrieval_chain.get_answer(question=user_question)
                
                self.display_answer(answer, source_doc)
                logger.info(f"Successfully answered: {user_question}")
            
            except Exception as e:
                st.error("Bir hata oluştu lütfen daha sonra tekrar deneyiniz.")
                logger.error(f"Error answering question: {str(e)}")
                logger.debug(traceback.format_exc())

@st.cache_resource(show_spinner="Vektör veritabanı yükleniyor...")
def get_vector_store(openai_api_key: str, file_path: str):
    vector_store_manager = VectorStoreManager(openai_api_key, file_path)
    return vector_store_manager.get_or_create_vector_store()

def main():
    try:
        config = ConfigManager.load_environment_variables()
        openai_api_key = config["openai_api_key"]
        groq_api_key = config["groq_api_key"]

        file_path = str(DEFAULT_PDF_PATH)

        doc_mod_time = os.path.getmtime(file_path)
        vector_store = get_vector_store(openai_api_key, file_path)

        retrieval_chain = RetrievalChain(vector_store, groq_api_key)

        app = StreamlitApp(retrieval_chain)
        app.run()
    
    except Exception as e:
        st.error("Uygulama başlatılırken hata oluştu lütfen daha sonra tekrar deneyiniz.")
        logger.critical(f"Application startup error: {str(e)}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    main()