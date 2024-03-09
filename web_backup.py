import streamlit as st
import fitz  # PyMuPDF
import re
import logging
from utils import opensearch
import boto3
import json

from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.info("Streamlit 앱 실행 중...")

opensearch_username = 'admin'
opensearch_password = 'Admin12#$'
cluster_name = 'test'
index_name = 'llm'
region = 'ap-northeast-2'
bedrock_region = 'us-east-1'
early_stop_record_count = 100
early_stop = False

bedrock_model_id = "anthropic.claude-v2"
bedrock_embedding_model_id = "amazon.titan-embed-text-v1"

def get_bedrock_client(region):
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    return bedrock_client
    

def create_langchain_vector_embedding_using_bedrock(bedrock_client, bedrock_embedding_model_id):
    bedrock_embeddings_client = BedrockEmbeddings(
        client=bedrock_client,
        model_id=bedrock_embedding_model_id)
    return bedrock_embeddings_client
    

def create_opensearch_vector_search_client(index_name, opensearch_username, opensearch_password, bedrock_embeddings_client, opensearch_endpoint, _is_aoss=False):
    docsearch = OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=bedrock_embeddings_client,
        opensearch_url=f"https://{opensearch_endpoint}",
        http_auth=(opensearch_username, opensearch_password),
        is_aoss=_is_aoss
    )
    return docsearch


def create_bedrock_llm(bedrock_client, model_version_id):
    bedrock_llm = Bedrock(
        model_id=model_version_id, 
        client=bedrock_client,
        model_kwargs={'temperature': 0}
        )
    return bedrock_llm

def get_bedrock_client(region):
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    return bedrock_client

def create_vector_embedding_with_bedrock(text, name, bedrock_client):
    payload = {"inputText": f"{text}"}
    body = json.dumps(payload)
    modelId = "amazon.titan-embed-text-v1"
    # modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    # modelId = "anthropic.claude-v2"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_client.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    embedding = response_body.get("embedding")
    return {"_index": name, "text": text, "vector_field": embedding}



def extract_sentences_from_pdf(pdf_file, progress_bar):

    opensearch_client =  opensearch.get_opensearch_cluster_client(cluster_name, index_name, opensearch_username, opensearch_password, region)
    
    response = opensearch.delete_opensearch_index(opensearch_client, index_name)
    if response:
        logging.info("OpenSearch index successfully deleted")
    
    logging.info(f"Checking if index {index_name} exists in OpenSearch cluster")
    exists = opensearch.check_opensearch_index(opensearch_client, index_name)    

    if not exists:
        logging.info("Creating OpenSearch index")
        success = opensearch.create_index(opensearch_client, index_name)
        if success:
            logging.info("Creating OpenSearch index mapping")
            success = opensearch.create_index_mapping(opensearch_client, index_name)
            logging.info(f"OpenSearch Index mapping created")


    # doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    # all_text = ""
    # for page in doc:
    #     all_text += page.get_text()
    # doc.close()
    # all_records = re.split(r'(?<=[.!?])\s+', all_text)

    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    
    all_records = []
    for page in doc:
        all_records.append(page.get_text())



    logging.info(f"PDF LIST 개수 : {len(all_records)}")

    total_records = len(all_records)
    processed_records = 0
    
    bedrock_client = get_bedrock_client(bedrock_region)

    all_json_records = []
    logging.info(f"Creating embeddings for records")

    for record in all_records:
        if early_stop and processed_records > early_stop_record_count:
            # Bulk put all records to OpenSearch
            success, failed = opensearch.put_bulk_in_opensearch(all_json_records, opensearch_client)
            logging.info(f"Documents saved {success}, documents failed to save {failed}")
            break
        
        records_with_embedding = create_vector_embedding_with_bedrock(record, index_name, bedrock_client)
        all_json_records.append(records_with_embedding)
        
        # 프로그레스 바 업데이트
        processed_records += 1
        progress = int((processed_records / total_records) * 100)
        progress_bar.progress(progress)
        
        logging.info(f"Embedding for record {processed_records} created")

        if processed_records % 500 == 0 or processed_records == len(all_records):
            # Bulk put all records to OpenSearch
            success, failed = opensearch.put_bulk_in_opensearch(all_json_records, opensearch_client)
            all_json_records = []  # 초기화
            logging.info(f"Documents saved {success}, documents failed to save {failed}")
    
    logging.info("Finished creating records using Amazon Bedrock Titan text embedding")
    logging.info("Cleaning up")
    
    return total_records

def find_answer_in_sentences(question):
    # Creating all clients for chain
    bedrock_client = get_bedrock_client(bedrock_region)
    bedrock_llm = create_bedrock_llm(bedrock_client, bedrock_model_id)
    bedrock_embeddings_client = create_langchain_vector_embedding_using_bedrock(bedrock_client, bedrock_embedding_model_id)
    opensearch_endpoint = opensearch.get_opensearch_endpoint(cluster_name, region)
    # opensearch_password = secret.get_secret(index_name, region)
    opensearch_vector_search_client = create_opensearch_vector_search_client(index_name, opensearch_username, opensearch_password, bedrock_embeddings_client, opensearch_endpoint)
    
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. don't include harmful content

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    logging.info(f"Starting the chain with KNN similarity using OpenSearch, Bedrock FM {bedrock_model_id}, and Bedrock embeddings with {bedrock_embedding_model_id}")
    qa = RetrievalQA.from_chain_type(llm=bedrock_llm, 
                                     chain_type="stuff", 
                                     retriever=opensearch_vector_search_client.as_retriever(),
                                     return_source_documents=True,
                                     chain_type_kwargs={"prompt": PROMPT, "verbose": True},
                                     verbose=True)
    
    response = qa(question, return_only_outputs=False)
    
    # logging.info("This are the similar documents from OpenSearch based on the provided query")
    source_documents = response.get('source_documents')
    # for d in source_documents:
    #     logging.info(f"With the following similar content from OpenSearch:\n{d.page_content}\n")
    #     logging.info(f"Text: {d.metadata['text']}")
    
    logging.info(f"The answer from Bedrock {bedrock_model_id} is: {response.get('result')}")

    return f"{response.get('result')}"

def main():
    st.sidebar.title("PDF 파일 업로드")
    pdf_file = st.sidebar.file_uploader("PDF 파일을 여기에 업로드하세요.", type=["pdf"])
    
    # 파일이 업로드 되었고, 이전에 처리가 완료되지 않았다면 처리 시작
    if pdf_file is not None and not st.session_state.get('processed', False):
        if 'progress_bar' not in st.session_state:
            st.session_state['progress_bar'] = st.sidebar.progress(0)
        record_cnt = extract_sentences_from_pdf(pdf_file, st.session_state['progress_bar'])
        st.session_state['processed'] = True  # 파일 처리 완료 표시
        st.session_state['record_cnt'] = record_cnt  # 처리된 레코드 수 저장
        st.session_state['progress_bar'].progress(100)  # 진행 표시줄을 100%로 설정
        st.sidebar.success(f"{record_cnt} Vector 업로드 완료!")
    elif 'processed' in st.session_state and st.session_state['processed']:
        # 파일 처리가 이미 완료된 경우, 성공 메시지와 진행률 표시
        st.sidebar.progress(100)
        st.sidebar.success(f"{st.session_state.get('record_cnt', 0)} Vector 업로드 완료!")

    st.title("챗봇")

    question = st.text_input("질문을 입력하세요:", "")
    if question :
        answer = find_answer_in_sentences(question)
        # st.write(f"챗봇 응답: {answer}")
        st.success(f"{answer}")
    # elif user_input:
    #     st.write("PDF 파일을 먼저 업로드해주세요.")


if __name__ == "__main__":
    main()
ㅁ