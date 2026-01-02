# PDF RAG System - Customization Guide

This guide explains how to customize the PDF RAG system as a template for different use cases and document types.

## 🎯 Quick Customization Checklist

### 1. **Document Type Changes**
- [ ] Update document processor for new file types
- [ ] Modify chunking strategy for document structure
- [ ] Adjust metadata extraction
- [ ] Update API file validation

### 2. **LLM Provider Changes**
- [ ] Replace Groq with your preferred provider
- [ ] Update configuration settings
- [ ] Modify query rewriter and answer generator
- [ ] Update environment variables

### 3. **Embedding Model Changes**
- [ ] Choose appropriate embedding model
- [ ] Update model configuration
- [ ] Adjust vector dimensions
- [ ] Test retrieval performance

### 4. **Domain-Specific Customization**
- [ ] Customize query enhancement prompts
- [ ] Adjust answer generation templates
- [ ] Modify retrieval parameters
- [ ] Add domain-specific validation

---

## 📋 Detailed Customization Areas

### 1. Document Processing (`src/document_processor.py`)

#### **For Different File Types:**

**Current:** PDF only
```python
# In document_processor.py
from langchain_community.document_loaders import PyPDFLoader
```

**Customize for:**
- **Word Documents:** Use `UnstructuredWordDocumentLoader`
- **Text Files:** Use `TextLoader`
- **Web Pages:** Use `WebBaseLoader`
- **Multiple Types:** Create a factory pattern

**Example for Word Documents:**
```python
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

class DocumentProcessor:
    def load_document(self, file_path: str):
        if file_path.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        return loader.load()
```

#### **For Different Metadata:**
```python
# Customize extract_metadata method
def extract_metadata(self, file_path: str) -> Dict[str, Any]:
    return {
        "filename": Path(file_path).name,
        "file_type": Path(file_path).suffix,
        "processed_at": datetime.now().isoformat(),
        # Add domain-specific metadata
        "department": "legal",  # Example
        "document_class": "contract"  # Example
    }
```

### 2. LLM Provider Changes

#### **Replace Groq with OpenAI:**

**Update `src/config.py`:**
```python
class Settings(BaseSettings):
    # Replace groq_api_key with openai_api_key
    openai_api_key: str = Field(..., description="OpenAI API key")
    llm_model: str = Field(default="gpt-4", description="OpenAI model")
```

**Update `src/query_rewriter.py` and `src/answer_generator.py`:**
```python
from langchain_openai import ChatOpenAI

# Replace ChatGroq with ChatOpenAI
self.llm = ChatOpenAI(
    model=settings.llm_model,
    temperature=0.1,
    api_key=settings.openai_api_key
)
```

#### **For Azure OpenAI:**
```python
from langchain_openai import AzureChatOpenAI

self.llm = AzureChatOpenAI(
    azure_endpoint=settings.azure_endpoint,
    api_key=settings.azure_api_key,
    api_version="2024-02-15-preview",
    deployment_name=settings.deployment_name
)
```

#### **For Local Models (Ollama):**
```python
from langchain_community.llms import Ollama

self.llm = Ollama(
    model=settings.llm_model,  # e.g., "llama2", "mistral"
    base_url=settings.ollama_base_url
)
```

### 3. Embedding Model Changes

#### **For Different Domains:**

**Current:** General-purpose `all-mpnet-base-v2`

**Domain-Specific Options:**
- **Legal:** `nlpaueb/legal-bert-base-uncased`
- **Medical:** `emilyalsentzer/Bio_ClinicalBERT`
- **Financial:** `ProsusAI/finbert`
- **Code:** `microsoft/codebert-base`

**Update `src/config.py`:**
```python
embedding_model: str = Field(
    default="nlpaueb/legal-bert-base-uncased",  # Legal domain
    description="Domain-specific embedding model"
)
```

#### **For Multilingual Support:**
```python
embedding_model: str = Field(
    default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    description="Multilingual embedding model"
)
```

### 4. Chunking Strategy (`src/chunking_engine.py`)

#### **For Structured Documents:**
```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

class ChunkingEngine:
    def __init__(self):
        # For markdown documents
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )
        
    def split_documents(self, documents):
        if self.is_markdown_document(documents[0]):
            return self.markdown_splitter.split_text(documents[0].page_content)
        # Fall back to recursive splitter
        return self.text_splitter.split_documents(documents)
```

#### **For Code Documents:**
```python
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

# For Python code
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=200
)
```

### 5. Query Enhancement (`src/query_rewriter.py`)

#### **Domain-Specific Prompts:**

**Legal Domain:**
```python
LEGAL_ENHANCEMENT_TEMPLATE = """
You are a legal document analysis expert. Enhance this query for better legal document retrieval:

Original Query: {original_query}

Enhanced Query Guidelines:
- Use precise legal terminology
- Include relevant legal concepts and synonyms
- Consider jurisdictional variations
- Focus on legal precedents and statutes

Enhanced Query:"""
```

**Medical Domain:**
```python
MEDICAL_ENHANCEMENT_TEMPLATE = """
You are a medical information specialist. Enhance this query for better medical document retrieval:

Original Query: {original_query}

Enhanced Query Guidelines:
- Use medical terminology and ICD codes
- Include anatomical and physiological terms
- Consider symptoms, diagnoses, and treatments
- Include drug names and medical procedures

Enhanced Query:"""
```

### 6. Answer Generation (`src/answer_generator.py`)

#### **Domain-Specific Answer Templates:**

**Legal Answers:**
```python
LEGAL_ANSWER_TEMPLATE = """
Based on the legal documents provided, answer the following question with legal precision:

Question: {question}

Legal Documents:
{context}

Instructions:
- Provide legally accurate information
- Cite specific sections and clauses
- Include relevant legal precedents if mentioned
- Clearly state any limitations or disclaimers
- Use proper legal citation format

Legal Analysis:"""
```

**Technical Documentation:**
```python
TECHNICAL_ANSWER_TEMPLATE = """
Based on the technical documentation provided, answer the following question:

Question: {question}

Technical Documentation:
{context}

Instructions:
- Provide step-by-step procedures when applicable
- Include code examples if mentioned in the documentation
- Reference specific sections and page numbers
- Highlight any warnings or prerequisites
- Use technical terminology appropriately

Technical Response:"""
```

### 7. API Customization (`src/api.py`)

#### **Add Domain-Specific Endpoints:**
```python
@app.post("/legal-analysis", response_model=LegalAnalysisResponse)
async def legal_analysis(request: LegalAnalysisRequest):
    """Specialized endpoint for legal document analysis."""
    # Custom legal processing logic
    pass

@app.post("/medical-query", response_model=MedicalQueryResponse)
async def medical_query(request: MedicalQueryRequest):
    """Specialized endpoint for medical information queries."""
    # Custom medical processing logic
    pass
```

#### **Add Custom Validation:**
```python
class LegalQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    jurisdiction: Optional[str] = Field(None, description="Legal jurisdiction")
    document_type: Optional[str] = Field(None, description="Type of legal document")
    
    @field_validator('query')
    @classmethod
    def validate_legal_query(cls, v):
        # Add legal-specific validation
        prohibited_terms = ["illegal advice", "legal advice"]
        if any(term in v.lower() for term in prohibited_terms):
            raise ValueError("Cannot provide legal advice")
        return v
```

### 8. Configuration Changes (`src/config.py`)

#### **Add Domain-Specific Settings:**
```python
class Settings(BaseSettings):
    # Domain-specific settings
    domain: str = Field(default="general", description="Application domain")
    
    # Legal domain settings
    legal_jurisdiction: Optional[str] = Field(None, description="Default legal jurisdiction")
    legal_citation_format: str = Field(default="bluebook", description="Legal citation format")
    
    # Medical domain settings
    medical_terminology_strict: bool = Field(True, description="Strict medical terminology")
    include_drug_interactions: bool = Field(False, description="Include drug interaction warnings")
    
    # Technical domain settings
    code_language: Optional[str] = Field(None, description="Primary programming language")
    include_code_examples: bool = Field(True, description="Include code examples in responses")
```

### 9. Environment Variables (`.env`)

#### **Update for New Providers:**
```bash
# Replace Groq with your provider
OPENAI_API_KEY=your_openai_api_key_here
# or
AZURE_OPENAI_API_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Domain-specific settings
DOMAIN=legal
LEGAL_JURISDICTION=US
EMBEDDING_MODEL=nlpaueb/legal-bert-base-uncased

# Custom model settings
LLM_MODEL=gpt-4
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### 10. Dependencies (`requirements.txt`)

#### **Add Domain-Specific Libraries:**
```txt
# For legal documents
spacy>=3.7.0
legal-nlp>=1.0.0

# For medical documents
scispacy>=0.5.0
medspacy>=1.0.0

# For code analysis
tree-sitter>=0.20.0
pygments>=2.16.0

# For different LLM providers
openai>=1.0.0  # Replace langchain-groq
# or
azure-openai>=1.0.0
# or
ollama>=0.1.0
```

---

## 🚀 Quick Start Templates

### Template 1: Legal Document System
```bash
# 1. Update configuration
sed -i 's/groq_api_key/openai_api_key/g' src/config.py
sed -i 's/llama-3.3-70b-versatile/gpt-4/g' src/config.py
sed -i 's/all-mpnet-base-v2/nlpaueb\/legal-bert-base-uncased/g' src/config.py

# 2. Update environment
echo "OPENAI_API_KEY=your_key_here" > .env
echo "DOMAIN=legal" >> .env
echo "LLM_MODEL=gpt-4" >> .env

# 3. Install legal dependencies
pip install spacy legal-nlp openai
```

### Template 2: Medical Document System
```bash
# 1. Update for medical domain
sed -i 's/all-mpnet-base-v2/emilyalsentzer\/Bio_ClinicalBERT/g' src/config.py

# 2. Update environment
echo "DOMAIN=medical" >> .env
echo "EMBEDDING_MODEL=emilyalsentzer/Bio_ClinicalBERT" >> .env

# 3. Install medical dependencies
pip install scispacy medspacy
```

### Template 3: Code Documentation System
```bash
# 1. Update for code domain
sed -i 's/all-mpnet-base-v2/microsoft\/codebert-base/g' src/config.py

# 2. Update environment
echo "DOMAIN=technical" >> .env
echo "CODE_LANGUAGE=python" >> .env

# 3. Install code analysis dependencies
pip install tree-sitter pygments
```

---

## 🔧 Testing Your Customizations

### 1. Update Tests
```bash
# Update test configurations
sed -i 's/groq/openai/g' tests/test_config.py
sed -i 's/llama-3.3-70b-versatile/gpt-4/g' tests/test_*.py
```

### 2. Run Validation Tests
```bash
# Test configuration
python -c "from src.config import settings; print(settings.model_dump())"

# Test document processing
python -c "from src.document_processor import DocumentProcessor; dp = DocumentProcessor(); print('OK')"

# Test full pipeline
pytest tests/ -v
```

### 3. Integration Testing
```bash
# Test with sample documents
python test_complete_rag_pipeline.py

# Test API endpoints
python api_client_example.py
```

---

## 📚 Common Use Cases

### 1. **Corporate Knowledge Base**
- **Documents:** Internal docs, policies, procedures
- **Customizations:** Employee authentication, department filtering
- **Models:** General-purpose embeddings, corporate LLM

### 2. **Legal Research System**
- **Documents:** Legal cases, statutes, regulations
- **Customizations:** Legal citation format, jurisdiction filtering
- **Models:** Legal-specific embeddings, legal-trained LLM

### 3. **Medical Information System**
- **Documents:** Medical literature, guidelines, protocols
- **Customizations:** Medical terminology, drug interaction warnings
- **Models:** Medical embeddings, healthcare-compliant LLM

### 4. **Technical Documentation**
- **Documents:** API docs, code documentation, manuals
- **Customizations:** Code syntax highlighting, version tracking
- **Models:** Code-aware embeddings, technical LLM

### 5. **Academic Research**
- **Documents:** Research papers, theses, journals
- **Customizations:** Citation management, field-specific terminology
- **Models:** Scientific embeddings, research-focused LLM

---

## ⚠️ Important Considerations

### 1. **Model Compatibility**
- Ensure embedding dimensions match between models
- Test retrieval quality with domain-specific models
- Validate LLM output format consistency

### 2. **Performance Impact**
- Larger models may require more memory/compute
- Domain-specific models might be slower
- Consider caching strategies for expensive operations

### 3. **Data Privacy**
- Review LLM provider data policies
- Consider local models for sensitive data
- Implement proper access controls

### 4. **Cost Management**
- Monitor API usage and costs
- Implement rate limiting
- Consider model size vs. performance trade-offs

### 5. **Compliance**
- Ensure compliance with domain regulations (HIPAA, GDPR, etc.)
- Implement audit logging
- Add appropriate disclaimers

---

This guide provides a comprehensive framework for customizing the PDF RAG system for your specific use case. Start with the quick customization checklist and gradually implement the detailed changes based on your requirements.