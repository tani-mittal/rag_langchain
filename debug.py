#!/usr/bin/env python3
import sys
sys.path.append('.')

import logging
from pathlib import Path
from src.document_processors.pdf_processor import PDFProcessor
from src.knowledge_base.s3_manager import S3Manager
from src.document_processors.base_processor import DocumentChunk
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_document_processing():
    """Test document processing pipeline"""
    logger.info("=== Testing Document Processing Pipeline ===")
    
    # Create a test chunk manually first
    logger.info("1. Testing with manual chunk...")
    test_chunk = DocumentChunk(
        content="This is a test chunk to verify S3 upload functionality.",
        metadata={
            "file_type": "test",
            "file_name": "manual_test.txt",
            "created_at": datetime.now().isoformat(),
            "test": True
        },
        chunk_id="manual_test_001",
        source_file="manual_test.txt"
    )
    
    # Test S3 upload
    try:
        s3_manager = S3Manager()
        logger.info("S3Manager initialized successfully")
        
        # Test basic S3 functionality
        if s3_manager.test_upload():
            logger.info("✅ Basic S3 test passed")
        else:
            logger.error("❌ Basic S3 test failed")
            return False
        
        # Upload test chunk
        logger.info("Uploading test chunk...")
        uris = s3_manager.upload_chunks([test_chunk])
        
        if uris:
            logger.info(f"✅ Test chunk uploaded successfully: {uris[0]}")
            
            # Verify upload
            chunks = s3_manager.list_chunks()
            logger.info(f"Found {len(chunks)} chunks in S3")
            for chunk in chunks:
                logger.info(f"  - {chunk['key']} ({chunk['size']} bytes)")
            
            return True
        else:
            logger.error("❌ Test chunk upload failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in S3 testing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_pdf_processing():
    """Test PDF processing if you have a PDF file"""
    logger.info("=== Testing PDF Processing ===")
    
    # Look for any PDF files in documents folder
    docs_folder = Path("./documents")
    if not docs_folder.exists():
        docs_folder.mkdir(exist_ok=True)
        logger.info(f"Created documents folder: {docs_folder}")
    
    pdf_files = list(docs_folder.glob("*.pdf"))
    
    if not pdf_files:
        logger.info("No PDF files found in ./documents folder")
        logger.info("Please add a PDF file to ./documents/ to test PDF processing")
        return True
    
    # Process first PDF found
    pdf_file = pdf_files[0]
    logger.info(f"Processing PDF: {pdf_file}")
    
    try:
        processor = PDFProcessor(chunk_size=500, chunk_overlap=50)  # Smaller chunks for testing
        chunks = processor.extract_content(pdf_file)
        
        logger.info(f"✅ PDF processing successful: {len(chunks)} chunks created")
        
        if chunks:
            # Show first chunk details
            first_chunk = chunks[0]
            logger.info(f"First chunk ID: {first_chunk.chunk_id}")
            logger.info(f"First chunk content preview: {first_chunk.content[:100]}...")
            logger.info(f"First chunk metadata: {first_chunk.metadata}")
            
            # Test uploading PDF chunks
            logger.info("Uploading PDF chunks to S3...")
            s3_manager = S3Manager()
            uris = s3_manager.upload_chunks(chunks[:2])  # Upload first 2 chunks only for testing
            
            if uris:
                logger.info(f"✅ PDF chunks uploaded successfully: {len(uris)} chunks")
                return True
            else:
                logger.error("❌ PDF chunk upload failed")
                return False
        else:
            logger.warning("⚠️ No chunks created from PDF")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error processing PDF: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    logger.info("🚀 Starting RAG Pipeline Debug")
    
    # Test 1: Manual chunk upload
    if not test_document_processing():
        logger.error("❌ Manual chunk test failed - stopping")
        return
    
    # Test 2: PDF processing (if available)
    test_pdf_processing()
    
    logger.info("🎉 Debug complete!")

if __name__ == "__main__":
    main()