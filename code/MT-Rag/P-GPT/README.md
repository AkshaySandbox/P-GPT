# Pregnancy GPT: Canadian Pregnancy & Parenting Information Assistant

![Model](https://img.shields.io/badge/Model-GPT--4-blue)
![Embeddings](https://img.shields.io/badge/Embeddings-Custom%20MPNet-yellow)
![Framework](https://img.shields.io/badge/Framework-LangChain%20%7C%20LangGraph-green)
![Interface](https://img.shields.io/badge/Interface-Chainlit-purple)

## 📋 Description

Pregnancy GPT is a specialized AI assistant that provides reliable, Canadian-focused information about pregnancy, childbirth, and early parenthood. It combines a curated knowledge base with targeted web searches to deliver accurate and helpful information.


### Problem Statement
Expecting and new parents in Canada often face challenges accessing reliable, timely, and personalized information on pregnancy, postpartum care, and early parenthood. They may struggle with understanding healthcare options, newborn care, mental health support, and government benefits, leading to stress and uncertainty.

### Target Audience
- Expecting parents seeking guidance on pregnancy health and wellness
- New parents navigating postpartum recovery and infant care
- Caregivers and family members supporting early parenthood
- Individuals looking for Canadian-specific healthcare resources, benefits, and support programs
### How the Chat Assistant Helps
- Provides Reliable Information: Offers accurate, evidence-based guidance on pregnancy, childbirth, and early parenting
- Supports Mental Well-being: Shares resources on postpartum depression, parental mental health, and self-care
- Guides on Government Benefits: Assists in understanding maternity/parental leave, child benefits, and healthcare services available in Canada
- Answers Common Parenting Questions: Covers baby care, feeding, sleep routines, and developmental milestones
- Connects to Local Services: Helps users find relevant healthcare providers, lactation consultants, parenting support groups, and community programs


## 🌟 Key Features

- **Canadian-Focused Information**: Prioritizes Canadian healthcare guidelines and resources
- **Custom Embedding Model**: Uses domain-specific embeddings for better search relevance
- **Dual Information Sources**: Knowledge base + targeted web search
- **Source Citations**: All information includes references
- **Follow-up Suggestions**: Offers relevant follow-up questions

## 💬 Example Questions

- "What are the signs of early labor?"
- "How do I apply for maternity benefits in Canada?"
- "What foods should I avoid during pregnancy?"
- "What are normal newborn sleep patterns?"
- "What are the symptoms of postpartum depression?"
- "What prenatal vitamins are recommended in Canada?"

## 🔍 How It Works

1. Your question is processed by our custom embedding model
2. The system searches our knowledge base of verified information
3. If needed, it performs a targeted web search of official Canadian health resources
4. Information is presented with clear sections, key takeaways, and source citations
   
## Demo video
https://www.loom.com/share/eba0163d11514689bc54480137159bc8

# 🛠️ Technical Details

## 🤖 LLM Choice (GPT-3.5-Turbo)
**Reason**: The choice of GPT-3.5-Turbo was made primarily for cost-effectiveness in prototyping while still maintaining reliable domain-specific responses. The LLM is used to:
- Process and understand pregnancy and parenting-related queries
- Generate contextually relevant responses from the knowledge base
- Maintain conversation coherence with Canadian-specific healthcare context

## 📚 Vector Database (Qdrant)
**Reason**: Qdrant was selected as the vector database for several compelling reasons:
- Supports local storage capabilities, reducing deployment complexity
- Provides efficient semantic search operations
- Offers seamless integration with LangChain framework
- Handles custom domain-specific embeddings effectively
- Maintains good performance with the specialized pregnancy-focused knowledge base

## 📊 Monitoring and Observability
The system implements several monitoring features:
- **Debug Logging**: Comprehensive debug logging system with verbose options
- **State Tracking**: Conversation state and history tracking
- **Performance Metrics**: Tracks tool usage and response generation times
- **Error Handling**: Robust error catching and fallback mechanisms
- **Request/Response Logging**: Detailed logging of user interactions and system responses

## 🎯 Evaluation Framework
The system has a comprehensive evaluation setup with the following metrics:

| **Metric** | **Score** | **Description** |
|------------|----------|----------------|
| Faithfulness | 0.714 | Measures response accuracy and consistency with provided context, Highest performing metric, indicating strong reliability |
| Answer Relevancy | 0.581 | Evaluates response alignment with user queries, Shows moderate performance in addressing specific questions |
| Context Precision | 0.511 | Assesses retrieved context relevance,area identified for improvement |
| Context Recall | 0.580 | Measures completeness of information retrieval, Indicates moderate performance in capturing relevant information |


The evaluation results are stored in:
- `rag_evaluation_results.csv` for metrics
- `rag_evaluation_data.csv` for detailed evaluation data

## 💡 Tooling Choices Summary

1. **Custom MPNet Embeddings**: Chose a domain-specific fine-tuned model (`AkshaySandbox/pregnancy-mpnet-embeddings`) to better understand pregnancy and parenting terminology.

2. **LangGraph Framework**: Selected for its robust orchestration capabilities and ability to manage complex conversation flows.

3. **Chainlit UI**: Implemented for its user-friendly interface and real-time interaction capabilities, making it accessible for expecting parents.

4. **Tavily Search Integration**: Used for targeted web searches of official Canadian health websites, ensuring information accuracy and relevance.

------
-Describe all of your data sources and external APIs, and describe what you’ll use them for.
-Describe the default chunking strategy that you will use.  Why did you make this decision?

## 📚 Primary Data Sources

### 1. Local Knowledge Base
- **Type**: PDF Documents
- **Storage**: Qdrant Vector Database
- **Purpose**: Stores pre-processed Canadian pregnancy and parenting information
- **Content Types**:
  - Medical guidelines
  - Healthcare protocols
  - Parenting resources
  - Canadian-specific health information

- **Document list**
    **1. 2014-A-Health-Professionals-Guide-to-Using-the-Charts.pdf**
    - **Description:** Guide for health professionals on interpreting charts.
    - **How to Find:** Search for `"2014 A Health Professionals Guide to Using the Charts PDF"`.

    **2. 924154693X_eng.pdf**
    - **Description:** The number resembles an ISBN, possibly a WHO publication.
    - **How to Find:** Search for `"924154693X PDF WHO"`.

    **3. adoption_practice_standards_abbreviated.pdf**
    - **Description:** Abbreviated adoption practice standards.
    - **How to Find:** Search for `"Adoption Practice Standards Abbreviated PDF"`.

    **4. babys-best-chance.pdf**
    - **Description:** A guide for parents in British Columbia on pregnancy and baby care.
    - **How to Find:** Search for `"Baby's Best Chance PDF British Columbia"`.

    **5. Birth Certificates - Province of British Columbia.pdf**
    - **Description:** Information on obtaining birth certificates in British Columbia.
    - **How to Find:** Search for `"Birth Certificates British Columbia PDF"`.

    **6. Canada Country Background Report - Quality in Early Childhood Education and Care.pdf**
    - **Description:** Report on early childhood education and care quality in Canada.
    - **How to Find:** Search for `"Canada Country Background Report Early Childhood Education PDF"`.

    **7. Canada Education Savings Grant (CESG) - Canada.ca.pdf**
    - **Description:** Information on the Canada Education Savings Grant.
    - **How to Find:** Search for `"Canada Education Savings Grant PDF site:canada.ca"`.

    **8. Canada Learning Bond - Canada.ca.pdf**
    - **Description:** Details about the Canada Learning Bond.
    - **How to Find:** Search for `"Canada Learning Bond PDF site:canada.ca"`.

    **9. ccb23byclc-e.pdf**
    - **Description:** Possibly related to the Canada Child Benefit (CCB).
    - **How to Find:** Search for `"ccb23byclc-e PDF Canada Child Benefit"`.

    **10.Early_Childhood_Development_in_Canada_EN_20200106.pdf**
    - **Description:** Report on early childhood development in Canada.
    - **How to Find:** Search for `"Early Childhood Development in Canada PDF"`.

    **11. EI maternity and parental benefits_ Eligibility - Canada.ca.pdf**
    - **Description:** Eligibility for EI maternity and parental benefits.
    - **How to Find:** Search for `"EI maternity and parental benefits eligibility PDF site:canada.ca"`.

    **12. EI maternity and parental benefits_ How much you could receive - Canada.ca.pdf**
    - **Description:** Information on how much can be received under EI maternity benefits.
    - **How to Find:** Search for `"EI maternity parental benefits amount PDF site:canada.ca"`.

    **13. EI maternity and parental benefits_ What these benefits offer - Canada.ca.pdf**
    - **Description:** Overview of EI maternity and parental benefits.
    - **How to Find:** Search for `"EI maternity parental benefits overview PDF site:canada.ca"`.

    **14. Employment Insurance Benefits Estimator.pdf**
    - **Description:** Guide to estimating Employment Insurance benefits.
    - **How to Find:** Search for `"Employment Insurance Benefits Estimator PDF Canada"`.

    **15. having-your-baby-your-hospital-stay-4648.pdf**
    - **Description:** Information on hospital stays during childbirth.
    - **How to Find:** Search for `"Having Your Baby Your Hospital Stay PDF"`.

    **16. maternity-guidelines-chapter-5-en.pdf**
    - **Description:** Chapter 5 of maternity guidelines.
    - **How to Find:** Search for `"Maternity Guidelines Chapter 5 PDF"`.

    **17. maternity-newborn-care-guidelines-chapter-4-eng.pdf**
    - **Description:** Chapter 4 of maternity and newborn care guidelines.
    - **How to Find:** Search for `"Maternity Newborn Care Guidelines Chapter 4 PDF"`.

    **18. parents_guide_to_selecting_and_monitoring_child_care_in_bc_dec_2019.pdf**
    - **Description:** Guide for parents on selecting and monitoring child care in British Columbia.
    - **How to Find:** Search for `"Parents Guide to Selecting Child Care BC PDF"`.

    **19. Postpartum and Newborn Care Summary Checklist for Primary Care Providers_0.pdf**
    - **Description:** Checklist for primary care providers on postpartum and newborn care.
    - **How to Find:** Search for `"Postpartum Newborn Care Summary Checklist PDF"`.

    **20. postpartum-health-guide.pdf**
    - **Description:** Guide on health during the postpartum period.
    - **How to Find:** Search for `"Postpartum Health Guide PDF"`.

    **21. PregnancyPassport.pdf**
    - **Description:** Likely a guide for pregnant individuals, possibly from a health authority.
    - **How to Find:** Search for `"Pregnancy Passport PDF Canada"`.

### 2. External APIs

#### a. Tavily Search API
- **Purpose**: Real-time web search for Canadian pregnancy information
- **Configuration**:
  - Max Results: 5
  - Search Depth: Advanced
  - Domain Restrictions:
    ```python
    [
        "canada.ca",
        "healthycanadians.gc.ca",
        "pregnancyinfo.ca",
        "caringforkids.cps.ca",
        "sogc.org",
        "cmaj.ca",
        "phac-aspc.gc.ca"
    ]
    ```
- **Usage**: Supplements knowledge base with current information from official Canadian health sources

#### b. OpenAI GPT API
- **Purpose**: Natural language processing and response generation
- **Model**: GPT-3.5-Turbo
- **Usage**: 
  - Processes user queries
  - Generates coherent responses
  - Summarizes search results
  - Formats information for user presentation

#### c. HuggingFace Hub API
- **Purpose**: Custom embeddings generation
- **Model**: `AkshaySandbox/pregnancy-mpnet-embeddings`
- **Usage**: Creates semantic embeddings for knowledge base documents and user queries, this fine tuned model for all the information related about pregnency. 

## 🔄 Chunking Strategy

### Default Chunking Implementation
The project uses a specialized document splitting strategy implemented in `pregnancy_kb/document_processor.py`:

The system uses a RecursiveCharacterTextSplitter with the following configuration:

```
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # Characters per chunk
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    length_function=tiktoken_len,
)
```

1. **Chunk Size**: 300
   - Documents are split into manageable chunks
   - Each chunk maintains context coherence
   - Chunks are processed in batches for efficient vector store insertion

2. **Chunking Rationale**:
   - **Context Preservation**: Ensures medical information isn't fragmented
   - **Semantic Coherence**: Maintains the integrity of related information
   - **Retrieval Optimization**: Balances chunk size for effective similarity search

3. **Processing Flow**:
   ```python
   documents = load_pdf_documents()
   splits = split_documents(documents)
   vector_store.add_documents(splits)
   ```

### Why This Chunking Decision?

#### **Chunk Size (300 characters):**:
   - Small enough to maintain context specificity
   - Large enough to capture meaningful information
   - Optimized for pregnancy-related medical content where precise information needs to be preserved

#### **Recursive Splitting**
- Uses a hierarchy of separators (`\n\n`, `\n`, `.`, etc.)
- Preserves natural text boundaries
- Prevents splitting in the middle of important medical terms or instructions

#### **Custom Length Function**
- Uses `tiktoken_len` for accurate token counting
- Ensures consistency with LLM token limits
- Better handles medical terminology and specialized vocabulary

#### **Metadata Preservation**
- Each chunk maintains original document metadata
- Adds section identification based on the first line
- Includes unique chunk IDs for traceability

#### **Category Classification** 
- Each chunk is categorized into specific pregnancy-related topics
- Categories include:
  - `prenatal`
  - `labor_delivery`
  - `postpartum`
  - `newborn_care`
  - `maternal_health`
  - `complications`
  - `nutrition`
  - `mental_health`
  - `canadian_benefits`
- Helps in targeted information retrieval and context maintenance



# Conclusions on Pipeline Performance and Effectiveness

## Strengths

### a) High Faithfulness (0.714)
- This is the strongest metric, indicating the system provides accurate and consistent information.
- Particularly important for medical/pregnancy advice where accuracy is crucial.
- Suggests the combination of custom embeddings and GPT-3.5 Turbo is effective for maintaining factual accuracy.

### b) Domain-Specific Optimization
- Custom-trained MPNet embeddings for pregnancy content.
- Well-structured categorization system (9 specific categories).
- Targeted web search restricted to official Canadian health websites.
- These optimizations help maintain domain relevance.

## Areas of Concern

### a) Context Precision (0.511)
- The lowest scoring metric.
- Suggests the retrieval system isn't always pulling the most relevant context.
- **Possible causes:**
  - Chunk size (300 characters) might be too small for some complex medical concepts.
  - Vector similarity might not always capture medical relevance effectively.

### b) Answer Relevancy (0.581)
- Moderate performance indicates responses sometimes drift from the question.
- Could be improved with better prompt engineering or context filtering.
- Might benefit from larger context windows.

### c) Context Recall (0.580)
- Moderate recall suggests some relevant information is being missed.
- **Possible causes:**
  - Conservative chunk overlap settings.
  - Limited context window size.
  - Potential gaps in the knowledge base.

## System Architecture Implications

### a) RAG Pipeline Effectiveness
- The system balances between knowledge base retrieval and web search.
- Hybrid approach helps compensate for gaps in either source.
- But metrics suggest room for optimization in retrieval strategy.

### b) Chunking Strategy Impact
- The 300-character chunk size might be contributing to the lower context precision.
- Small chunks maintain specificity but might fragment important medical concepts.
- Could benefit from dynamic chunk sizing based on content type.

## Recommendations for Improvement

### a) Short-term Optimizations:
- Increase chunk size for complex medical concepts.
- Implement better context filtering before LLM processing.
- Add medical-specific relevance scoring to retrieval.

### b) Structural Changes:
- Consider implementing a hierarchical retrieval system.
- Add medical concept linking across chunks.
- Implement fact-checking against medical databases.

### c) Evaluation Framework:
- Add medical accuracy as a specific metric.
- Implement user feedback loops.
- Track performance by category.

## Risk Assessment

### a) Medical Information Reliability
- High faithfulness (0.714) is good but not perfect for medical advice.
- System appropriately includes disclaimers.
- Conservative approach with official source citations.

### b) Information Coverage
- Moderate recall (0.580) suggests potential information gaps.
- **Mitigated by:**
  - Multiple information sources.
  - Clear source attribution.
  - Fallback to official resources.

## 📂 Additional Resources
- **FineTuned Model**: [Pregnancy MPNet Embeddings](https://huggingface.co/AkshaySandbox/pregnancy-mpnet-embeddings)
- **Evaluation Results**: [GitHub Repository](https://github.com/AkshaySandbox/P-GPT/tree/main/code/MT-Rag/P-GPT/evaluation)


Recommendations:
--------------
- Enhance answer focus by improving question understanding
- Refine context retrieval by adjusting similarity search parameters
- Expand knowledge coverage by adding more comprehensive documentation

## ⚠️ Disclaimer

This tool provides information but does not replace professional medical advice. Always consult healthcare providers for medical decisions. 
