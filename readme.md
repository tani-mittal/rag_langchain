# Multi-Format RAG Agent (PDF/PPTX/DOCX/XLSX) using AWS Bedrock KB + Titan Embeddings

This project builds a local RAG pipeline using **LangChain / LangGraph** and **AWS Bedrock Knowledge Bases** to support ingestion and question-answering across multiple document formats:

- PDF
- PowerPoint (PPT/PPTX)
- Word (DOCX)
- Excel (XLSX)

The pipeline converts all documents into normalized text chunks, uploads them to an S3-backed Bedrock Knowledge Base datasource, and triggers a KB ingestion job. Queries are answered using Bedrock Knowledge Base retrieval + an LLM (Claude).

---

## Architecture Overview

### Ingestion (Indexing)
1. File is provided locally (or can be downloaded from S3 raw uploads)
2. Format-aware extraction:
   - PDF → page wise
   - PPTX → slide wise
   - DOCX → section/heading wise
   - XLSX → sheet wise + row-range chunking
3. Chunks are written as `.txt` (or `.md`) files with metadata headers
4. Chunks are uploaded to S3 KB datasource prefix
5. Bedrock KB ingestion job is triggered
6. Titan embeddings are generated and stored in KB backend vector store

### Query (RAG)
1. User enters query
2. LangGraph agent calls Bedrock KB retrieve (and optionally generate)
3. Claude produces a grounded response with citations

---

