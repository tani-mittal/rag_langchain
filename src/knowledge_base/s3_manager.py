import boto3
from pathlib import Path
from typing import List, Dict, Any
import json
import logging
from datetime import datetime
from src.document_processors.base_processor import DocumentChunk
import config

class S3Manager:
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
            # Debug: Print configuration
            self.logger.info("=== S3Manager Configuration ===")
            self.logger.info(f"AWS_REGION: {config.AWS_REGION}")
            self.logger.info(f"S3_BUCKET: {config.S3_BUCKET}")
            self.logger.info(f"S3_PREFIX: {config.S3_PREFIX}")
            
            # Use default credential chain (works with Okta)
            # Don't pass explicit credentials - let boto3 find them
            self.s3_client = boto3.client('s3', region_name=config.AWS_REGION)
            self.bucket = config.S3_BUCKET
            self.prefix = config.S3_PREFIX
            
            # Test current AWS identity
            sts_client = boto3.client('sts', region_name=config.AWS_REGION)
            identity = sts_client.get_caller_identity()
            self.logger.info(f"AWS Identity: {identity.get('Arn', 'Unknown')}")
            self.logger.info(f"Account: {identity.get('Account', 'Unknown')}")
            
            # Test bucket access immediately
            self.logger.info("Testing S3 bucket access...")
            self.s3_client.head_bucket(Bucket=self.bucket)
            self.logger.info(f"✅ Successfully connected to S3 bucket: {self.bucket}")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing S3Manager: {e}")
            self.logger.error("💡")
            raise
    
    def upload_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """Upload document chunks to S3 and return S3 URIs"""
        uploaded_uris = []
        
        self.logger.info("=== Starting S3 Upload Process ===")
        self.logger.info(f"📤 Uploading {len(chunks)} chunks to S3...")
        self.logger.info(f"Target: s3://{self.bucket}/{self.prefix}chunks/")
        
        for i, chunk in enumerate(chunks):
            try:
                # Create content with metadata
                content = self._create_chunk_content(chunk)
                
                # Create S3 key
                s3_key = f"{self.prefix}chunks/{chunk.chunk_id}.txt"
                
                self.logger.info(f"📄 Uploading chunk {i+1}/{len(chunks)}: {chunk.chunk_id}")
                self.logger.info(f"   S3 Key: {s3_key}")
                self.logger.info(f"   Content size: {len(content)} bytes")
                
                # Upload to S3
                response = self.s3_client.put_object(
                    Bucket=self.bucket,
                    Key=s3_key,
                    Body=content,
                    ContentType='text/plain',
                    Metadata={
                        'chunk_id': chunk.chunk_id,
                        'source_file': chunk.source_file,
                        'file_type': chunk.metadata.get('file_type', 'unknown'),
                        'upload_timestamp': datetime.now().isoformat()
                    }
                )
                
                s3_uri = f"s3://{self.bucket}/{s3_key}"
                uploaded_uris.append(s3_uri)
                
                self.logger.info(f"✅ Successfully uploaded: {s3_uri}")
                self.logger.info(f"   ETag: {response.get('ETag', 'N/A')}")
                
                if (i + 1) % 5 == 0:
                    self.logger.info(f"🔄 Progress: {i + 1}/{len(chunks)} chunks uploaded")
                
            except Exception as e:
                self.logger.error(f"❌ Error uploading chunk {chunk.chunk_id}: {e}")
                self.logger.error(f"   S3 Key: {s3_key}")
                self.logger.error(f"   Bucket: {self.bucket}")
                import traceback
                self.logger.error(f"   Traceback: {traceback.format_exc()}")
                continue
        
        self.logger.info("=== Upload Process Complete ===")
        self.logger.info(f"🎉 Successfully uploaded {len(uploaded_uris)}/{len(chunks)} chunks to S3")
        
        # Verify uploads
        if uploaded_uris:
            self.logger.info("🔍 Verifying uploads...")
            self._verify_recent_uploads(len(uploaded_uris))
        
        return uploaded_uris
    
    def _verify_recent_uploads(self, expected_count: int):
        """Verify recent uploads"""
        try:
            list_prefix = f"{self.prefix}chunks/"
            self.logger.info(f"Checking S3 path: s3://{self.bucket}/{list_prefix}")
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=list_prefix,
                MaxKeys=expected_count + 10
            )
            
            if 'Contents' in response:
                found_count = len(response['Contents'])
                self.logger.info(f"✅ Verification: Found {found_count} files in S3")
                
                # Show recent files
                sorted_objects = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
                self.logger.info("📋 Most recent uploads:")
                for i, obj in enumerate(sorted_objects[:5]):
                    self.logger.info(f"   {i+1}. {obj['Key']} ({obj['Size']} bytes, {obj['LastModified']})")
                    
                if found_count >= expected_count:
                    self.logger.info("🎉 All uploads verified successfully!")
                else:
                    self.logger.warning(f"⚠️ Expected {expected_count} files, but found {found_count}")
            else:
                self.logger.error("❌ No files found in S3 after upload!")
                self.logger.error(f"   Checked path: s3://{self.bucket}/{list_prefix}")
                
        except Exception as e:
            self.logger.error(f"❌ Error verifying uploads: {e}")
    
    def _create_chunk_content(self, chunk: DocumentChunk) -> str:
        """Create formatted content for the chunk"""
        content = f"# Document Chunk: {chunk.chunk_id}\n\n"
        content += f"**Upload Timestamp:** {datetime.now().isoformat()}\n"
        content += f"**Source:** {chunk.source_file}\n"
        content += f"**File Type:** {chunk.metadata.get('file_type', 'unknown')}\n"
        content += f"**File Name:** {chunk.metadata.get('file_name', 'unknown')}\n"
        
        if chunk.page_number:
            content += f"**Page:** {chunk.page_number}\n"
        
        if 'sheet' in chunk.metadata:
            content += f"**Sheet:** {chunk.metadata['sheet']}\n"
            content += f"**Rows:** {chunk.metadata.get('rows', 'N/A')}\n"
        
        if 'slide' in chunk.metadata:
            content += f"**Slide:** {chunk.metadata['slide']}\n"
        
        if 'section' in chunk.metadata:
            content += f"**Section:** {chunk.metadata['section']}\n"
        
        content += f"\n**Metadata:**\n```json\n{json.dumps(chunk.metadata, indent=2)}\n```\n\n"
        content += "## Content\n\n"
        content += chunk.content
        
        # Add OCR text from images if available
        if chunk.images:
            content += "\n\n## Images and OCR Text\n\n"
            for i, img_data in enumerate(chunk.images):
                if isinstance(img_data, dict) and img_data.get('ocr_text'):
                    content += f"### Image {i+1} OCR Text:\n{img_data['ocr_text']}\n\n"
                    content += f"**Image Size:** {img_data.get('size', 'Unknown')}\n"
                    content += f"**Has Text:** {img_data.get('has_text', False)}\n\n"
        
        return content
    
    def test_upload(self) -> bool:
        """Test S3 upload with a simple file"""
        try:
            test_key = f"{self.prefix}test/upload_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            test_content = f"Test upload at {datetime.now().isoformat()}\nBucket: {self.bucket}\nPrefix: {self.prefix}"
            
            self.logger.info(f"Testing upload to: s3://{self.bucket}/{test_key}")
            
            response = self.s3_client.put_object(
                Bucket=self.bucket,
                Key=test_key,
                Body=test_content,
                ContentType='text/plain'
            )
            
            self.logger.info(f"✅ Test upload successful! ETag: {response.get('ETag')}")
            
            # Verify the upload
            obj_response = self.s3_client.get_object(Bucket=self.bucket, Key=test_key)
            retrieved_content = obj_response['Body'].read().decode('utf-8')
            
            if retrieved_content == test_content:
                self.logger.info("✅ Test file verification successful!")
                
                # Clean up test file
                self.s3_client.delete_object(Bucket=self.bucket, Key=test_key)
                self.logger.info("✅ Test file cleaned up")
                
                return True
            else:
                self.logger.error("❌ Test file content mismatch!")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Test upload failed: {e}")
            return False
    
    def list_chunks(self, prefix: str = None) -> List[Dict[str, Any]]:
        """List chunks in S3"""
        try:
            list_prefix = f"{self.prefix}chunks/"
            if prefix:
                list_prefix += prefix
            
            self.logger.info(f"Listing objects with prefix: {list_prefix}")
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=list_prefix
            )
            
            chunks = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    chunks.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'],
                        'uri': f"s3://{self.bucket}/{obj['Key']}"
                    })
                self.logger.info(f"Found {len(chunks)} chunks in S3")
            else:
                self.logger.info("No chunks found in S3")
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error listing chunks: {e}")
            return []
    
    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """Delete specific chunks from S3"""
        try:
            objects_to_delete = []
            for chunk_id in chunk_ids:
                s3_key = f"{self.prefix}chunks/{chunk_id}.txt"
                objects_to_delete.append({'Key': s3_key})
            
            if objects_to_delete:
                response = self.s3_client.delete_objects(
                    Bucket=self.bucket,
                    Delete={'Objects': objects_to_delete}
                )
                
                deleted_count = len(response.get('Deleted', []))
                self.logger.info(f"Deleted {deleted_count} chunks from S3")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error deleting chunks: {e}")
            return False