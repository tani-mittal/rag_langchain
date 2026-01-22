from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
import uuid

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_file: str
    page_number: Optional[int] = None
    images: Optional[List[Dict[str, Any]]] = None

class BaseDocumentProcessor(ABC):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    def extract_content(self, file_path: Path) -> List[DocumentChunk]:
        """Extract content from document and return chunks"""
        pass
    
    def process_images(self, images: List[Any]) -> List[Dict[str, Any]]:
        """Process images and return processed image data"""
        try:
            from src.utils.image_processor import ImageProcessor
            processor = ImageProcessor()
            return processor.process_images(images)
        except Exception as e:
            self.logger.error(f"Error processing images: {e}")
            return []
    
    def create_chunk_id(self, source_file: str, chunk_index: int) -> str:
        """Create unique chunk ID"""
        file_stem = Path(source_file).stem
        unique_id = str(uuid.uuid4())[:8]
        return f"{file_stem}_{chunk_index}_{unique_id}"
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if not text or len(text.strip()) == 0:
            return []
            
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + self.chunk_size - 100, start)
                sentence_end = -1
                
                for i in range(end, search_start, -1):
                    if text[i] in '.!?':
                        sentence_end = i + 1
                        break
                
                if sentence_end > start:
                    end = sentence_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= end:
                start = end
        
        return chunks
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove special characters that might cause issues
        text = text.replace('\x00', '')  # Remove null bytes
        text = text.replace('\ufffd', '')  # Remove replacement characters
        
        return text.strip()
