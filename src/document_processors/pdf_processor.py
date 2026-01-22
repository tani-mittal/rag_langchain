import fitz  # PyMuPDF
from typing import List
from pathlib import Path
from .base_processor import BaseDocumentProcessor, DocumentChunk

class PDFProcessor(BaseDocumentProcessor):
    def extract_content(self, file_path: Path) -> List[DocumentChunk]:
        chunks = []
        
        try:
            self.logger.info(f"Processing PDF: {file_path}")
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text
                text = page.get_text()
                text = self.clean_text(text)
                
                if not text:
                    continue
                
                # Extract images
                images = []
                try:
                    image_list = page.get_images()
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            images.append(img_data)
                        pix = None
                except Exception as e:
                    self.logger.warning(f"Error extracting images from page {page_num}: {e}")
                
                # Process images
                processed_images = self.process_images(images) if images else None
                
                # Split text into chunks
                text_chunks = self.split_text(text)
                
                for i, chunk_text in enumerate(text_chunks):
                    chunk_id = self.create_chunk_id(str(file_path), len(chunks))
                    
                    chunk = DocumentChunk(
                        content=chunk_text,
                        metadata={
                            "source": str(file_path),
                            "page": page_num + 1,
                            "chunk_index": i,
                            "file_type": "pdf",
                            "total_pages": len(doc),
                            "file_name": file_path.name
                        },
                        chunk_id=chunk_id,
                        source_file=str(file_path),
                        page_number=page_num + 1,
                        images=processed_images if i == 0 else None  # Attach images to first chunk of page
                    )
                    chunks.append(chunk)
            
            doc.close()
            self.logger.info(f"Successfully processed PDF: {len(chunks)} chunks created")
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
        
        return chunks