import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk
import logging


def get_opensearch_cluster_client(cluster_name, index_name, username, password, region):
    opensearch_endpoint = get_opensearch_endpoint(cluster_name, region)
    print("#####", opensearch_endpoint)
    opensearch_client = OpenSearch(
        hosts=[{
            'host': opensearch_endpoint,
            'port': 443
            }],
        http_auth=(username, password),
        index_name = index_name,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30
        )
    return opensearch_client
    
    
def get_opensearch_endpoint(cluster_name, region):
    client = boto3.client('es', region_name=region)
    response = client.describe_elasticsearch_domain(
        DomainName=cluster_name
    )
    return response['DomainStatus']['Endpoints']['vpc']


def put_bulk_in_opensearch(list, client):
    logging.info(f"Putting {len(list)} documents in OpenSearch")
    success, failed = bulk(client, list)
    return success, failed

def check_opensearch_index(opensearch_client, index_name):
    return opensearch_client.indices.exists(index=index_name)


def create_index(opensearch_client, index_name):
    settings = {
        "settings": {
            "index": {
                "knn": True,
                "knn.space_type": "cosinesimil"
                }
            }
        }
    response = opensearch_client.indices.create(index=index_name, body=settings)
    return bool(response['acknowledged'])
    
    
def create_index_mapping(opensearch_client, index_name):
    response = opensearch_client.indices.put_mapping(
        index=index_name,
        body={
            "properties": {
                "vector_field": {
                    "type": "knn_vector",
                    "dimension": 1536
                },
                "text": {
                    "type": "keyword"
                }
            }
        }
    )
    return bool(response['acknowledged'])


def delete_opensearch_index(opensearch_client, index_name):
    logging.info(f"Trying to delete index {index_name}")
    try:
        response = opensearch_client.indices.delete(index=index_name)
        logging.info(f"Index {index_name} deleted")
        return response['acknowledged']
    except Exception as e:
        logging.info(f"Index {index_name} not found, nothing to delete")
        return True
        