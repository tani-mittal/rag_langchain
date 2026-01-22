import boto3
import time
from typing import List, Dict, Any, Optional
import logging
import config

class BedrockKnowledgeBase:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        try:
            # Use default credential chain (works with Okta)
            self.bedrock_agent = boto3.client(
                'bedrock-agent',
                region_name=config.AWS_REGION
            )
            
            self.bedrock_runtime = boto3.client(
                'bedrock-agent-runtime',
                region_name=config.AWS_REGION
            )
            
            # Test current AWS identity
            sts_client = boto3.client('sts', region_name=config.AWS_REGION)
            identity = sts_client.get_caller_identity()
            self.logger.info(f"BedrockKnowledgeBase AWS Identity: {identity.get('Arn', 'Unknown')}")
            
            self.logger.info("BedrockKnowledgeBase initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing BedrockKnowledgeBase: {e}")
            self.logger.error("💡 ")
            raise
    
    def retrieve(self, kb_id: str, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from knowledge base"""
        try:
            self.logger.info(f"Retrieving from KB {kb_id} with query: {query[:100]}...")
            
            response = self.bedrock_runtime.retrieve(
                knowledgeBaseId=kb_id,
                retrievalQuery={'text': query},
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': max_results
                    }
                }
            )
            
            results = response.get('retrievalResults', [])
            self.logger.info(f"Retrieved {len(results)} results")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving from knowledge base: {e}")
            raise
    
    def retrieve_and_generate(self, kb_id: str, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve and generate response using knowledge base"""
        try:
            self.logger.info(f"Retrieve and generate for KB {kb_id}")
            
            request_params = {
                'input': {'text': query},
                'retrieveAndGenerateConfiguration': {
                    'type': 'KNOWLEDGE_BASE',
                    'knowledgeBaseConfiguration': {
                        'knowledgeBaseId': kb_id,
                        'modelArn': config.model_arn
                    }
                }
            }
            
            # Add session ID if provided for conversation continuity
            if session_id:
                request_params['sessionId'] = session_id
            
            response = self.bedrock_runtime.retrieve_and_generate(**request_params)
            
            self.logger.info("Successfully generated response")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in retrieve and generate: {e}")
            raise
    
    def start_ingestion_job(self, kb_id: str, data_source_id: str) -> str:
        """Start an ingestion job for existing KB and data source"""
        try:
            self.logger.info(f"Starting ingestion job for KB {kb_id}, data source {data_source_id}")
            
            response = self.bedrock_agent.start_ingestion_job(
                knowledgeBaseId=kb_id,
                dataSourceId=data_source_id
            )
            
            job_id = response['ingestionJob']['ingestionJobId']
            self.logger.info(f"Started ingestion job: {job_id}")
            
            return job_id
            
        except Exception as e:
            self.logger.error(f"Error starting ingestion job: {e}")
            raise
    
    def get_ingestion_job_status(self, kb_id: str, data_source_id: str, job_id: str) -> Dict[str, Any]:
        """Get ingestion job status"""
        try:
            response = self.bedrock_agent.get_ingestion_job(
                knowledgeBaseId=kb_id,
                dataSourceId=data_source_id,
                ingestionJobId=job_id
            )
            
            return response['ingestionJob']
            
        except Exception as e:
            self.logger.error(f"Error getting ingestion job status: {e}")
            raise
    
    def wait_for_ingestion(self, kb_id: str, data_source_id: str, job_id: str, timeout: int = 1800) -> bool:
        """Wait for ingestion job to complete"""
        start_time = time.time()
        
        self.logger.info(f"Waiting for ingestion job {job_id} to complete...")
        
        while time.time() - start_time < timeout:
            try:
                job_info = self.get_ingestion_job_status(kb_id, data_source_id, job_id)
                status = job_info['status']
                
                self.logger.info(f"Ingestion job status: {status}")
                
                if status == 'COMPLETE':
                    self.logger.info("Ingestion job completed successfully")
                    return True
                elif status == 'FAILED':
                    failure_reasons = job_info.get('failureReasons', [])
                    self.logger.error(f"Ingestion job failed: {failure_reasons}")
                    return False
                
                time.sleep(30)  # Wait 30 seconds before checking again
                
            except Exception as e:
                self.logger.error(f"Error checking ingestion status: {e}")
                time.sleep(30)
        
        self.logger.error("Ingestion job timed out")
        return False
    
    def list_knowledge_bases(self) -> List[Dict[str, Any]]:
        """List available knowledge bases"""
        try:
            response = self.bedrock_agent.list_knowledge_bases()
            return response.get('knowledgeBaseSummaries', [])
        except Exception as e:
            self.logger.error(f"Error listing knowledge bases: {e}")
            return []
    
    def get_knowledge_base(self, kb_id: str) -> Dict[str, Any]:
        """Get knowledge base details"""
        try:
            response = self.bedrock_agent.get_knowledge_base(knowledgeBaseId=kb_id)
            return response['knowledgeBase']
        except Exception as e:
            self.logger.error(f"Error getting knowledge base {kb_id}: {e}")
            raise
    
    def list_data_sources(self, kb_id: str) -> List[Dict[str, Any]]:
        """List data sources for a knowledge base"""
        try:
            response = self.bedrock_agent.list_data_sources(knowledgeBaseId=kb_id)
            return response.get('dataSourceSummaries', [])
        except Exception as e:
            self.logger.error(f"Error listing data sources for KB {kb_id}: {e}")
            return []