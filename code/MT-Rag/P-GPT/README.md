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

## ⚠️ Disclaimer

This tool provides information but does not replace professional medical advice. Always consult healthcare providers for medical decisions. 