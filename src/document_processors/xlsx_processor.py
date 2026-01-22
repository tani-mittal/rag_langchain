import pandas as pd
from typing import List
from pathlib import Path
from .base_processor import BaseDocumentProcessor, DocumentChunk

class XLSXProcessor(BaseDocumentProcessor):
    def extract_content(self, file_path: Path) -> List[DocumentChunk]:
        chunks = []
        
        try:
            self.logger.info(f"Processing XLSX: {file_path}")
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    if df.empty:
                        continue
                    
                    # Convert dataframe to text representation
                    sheet_text = f"Sheet: {sheet_name}\n"
                    sheet_text += f"Columns: {', '.join(df.columns.astype(str))}\n\n"
                    
                    # Process in chunks of rows
                    chunk_size_rows = 50
                    
                    for start_row in range(0, len(df), chunk_size_rows):
                        end_row = min(start_row + chunk_size_rows, len(df))
                        chunk_df = df.iloc[start_row:end_row]
                        
                        # Convert chunk to text
                        chunk_text = sheet_text + chunk_df.to_string(index=False)
                        chunk_text = self.clean_text(chunk_text)
                        
                        if chunk_text:
                            chunk_id = self.create_chunk_id(str(file_path), len(chunks))
                            
                            chunk = DocumentChunk(
                                content=chunk_text,
                                metadata={
                                    "source": str(file_path),
                                    "sheet": sheet_name,
                                    "rows": f"{start_row + 1}-{end_row}",
                                    "chunk_index": len(chunks),
                                    "file_type": "xlsx",
                                    "total_rows": len(df),
                                    "columns": list(df.columns),
                                    "file_name": file_path.name
                                },
                                chunk_id=chunk_id,
                                source_file=str(file_path)
                            )
                            chunks.append(chunk)
                
                except Exception as e:
                    self.logger.warning(f"Error processing sheet {sheet_name}: {e}")
                    continue
            
            self.logger.info(f"Successfully processed XLSX: {len(chunks)} chunks created")
            
        except Exception as e:
            self.logger.error(f"Error processing XLSX {file_path}: {str(e)}")
            raise
        
        return chunks