# Pregnancy GPT: Canadian Pregnancy & Parenting Information Assistant

![Model](https://img.shields.io/badge/Model-GPT--4-blue)
![Embeddings](https://img.shields.io/badge/Embeddings-Custom%20MPNet-yellow)
![Framework](https://img.shields.io/badge/Framework-LangChain%20%7C%20LangGraph-green)
![Interface](https://img.shields.io/badge/Interface-Chainlit-purple)

## 📋 Description

Pregnancy GPT is a specialized AI assistant that provides reliable, Canadian-focused information about pregnancy, childbirth, and early parenthood. It combines a curated knowledge base with targeted web searches to deliver accurate and helpful information.

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

## 🛠️ Technical Details

- **Vector Database**: Qdrant for semantic search
- **Embedding Model**: Custom fine-tuned MPNet (`AkshaySandbox/pregnancy-mpnet-embeddings`)
- **Agent Framework**: LangGraph for orchestration
- **Web Search**: Targeted search of official Canadian health websites
- **UI**: Chainlit for interactive chat experience

## FineTuned Model 

- https://huggingface.co/AkshaySandbox/pregnancy-mpnet-embeddings

## Evaluation Result 

- https://github.com/AkshaySandbox/P-GPT/tree/main/code/MT-Rag/P-GPT/evaluation

faithfulness: 0.714
Description: Measures how accurate and consistent the responses are with the provided context

answer_relevancy: 0.581
Description: Evaluates how well the responses address the specific questions asked

context_precision: 0.511
Description: Assesses the relevance of the retrieved context to the question

context_recall: 0.580
Description: Measures how well the system captures all relevant information

Results saved to:
- rag_evaluation_results.csv (metrics)
- rag_evaluation_data.csv (full evaluation data)

Summary Statistics:
------------------
Average Score: 0.597
Best Metric: faithfulness (0.714)
Areas for Improvement: context_precision (0.511)

Recommendations:
--------------
- Enhance answer focus by improving question understanding
- Refine context retrieval by adjusting similarity search parameters
- Expand knowledge coverage by adding more comprehensive documentation

## Demo Video
- https://www.loom.com/share/b167f2b002ac4477bd3bcef9434b3aac


## ⚠️ Disclaimer

This tool provides information but does not replace professional medical advice. Always consult healthcare providers for medical decisions. 
