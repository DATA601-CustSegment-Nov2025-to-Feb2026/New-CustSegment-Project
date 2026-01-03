import os
import sys
from typing import List, Dict, Optional
from dotenv import load_dotenv

# LangChain imports - latest versions
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
import chromadb

# Load environment variables
load_dotenv()

class SegmentationAgent:
    """
    AI Agent for Customer Segmentation Analysis
    Features:
    - Multiple LLM support (OpenAI, Google Gemini, X.AI Grok)
    - Few-shot prompting for consistent analysis
    - RAG with semantic search for cluster data
    - Conversational memory
    """
    
    def __init__(self, model_choice: str = "openai"):
        """Initialize the agent with specified model"""
        self.model_choice = model_choice
        self.llm = self._initialize_llm(model_choice)
        
        # Use OpenAI embeddings (no PyTorch dependency)
        print("🔧 Using OpenAI embeddings...")
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small",  # Use smaller, faster model
            timeout=30,  # 30 second timeout
            max_retries=2
        )
        
        self.vectorstore = None
        self.chat_history = []  # Simple list for conversation history
        self._initialize_knowledge_base()
        
    def _initialize_llm(self, model_choice: str) -> ChatOpenAI:
        """Initialize the chosen LLM"""
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        
        models = {
            "openai": {
                "model": "gpt-4o-mini",
                "use_base_url": False
            },
            "gemini": {
                "model": "google/gemini-2.0-flash-exp",
                "use_base_url": True
            },
            "grok": {
                "model": "x-ai/grok-2-1212",
                "use_base_url": True
            }
        }
        
        config = models.get(model_choice, models["openai"])
        
        # Build kwargs
        kwargs = {
            "model": config["model"],
            "api_key": api_key,
            "temperature": 0.7
        }
        
        # Add base_url only for non-OpenAI models
        if config["use_base_url"] and base_url:
            kwargs["base_url"] = base_url
        
        return ChatOpenAI(**kwargs)
    
    def _initialize_knowledge_base(self):
        """Initialize in-memory vector database with cluster analysis knowledge"""
        
        print("📚 Loading cluster analysis knowledge base...")
        
        # Sample cluster data
        cluster_documents = [
            Document(
                page_content="""
                RFM Model 1 - K-Means with k=4:
                - Cluster 1: High frequency (28 transactions), medium recency (143 days), medium monetary ($412)
                - Cluster 2: Low frequency (9 transactions), high recency (210 days), low monetary ($115)
                - Cluster 3: Very high frequency (52 transactions), low recency (85 days), high monetary ($687)
                - Cluster 4: Medium frequency (18 transactions), medium recency (165 days), medium monetary ($298)
                
                Key findings: Monetary and frequency explain 54% and 45% of variance respectively.
                Clusters are well-balanced (23-28% each).
                """,
                metadata={"model": "rfm_kmeans", "type": "cluster_summary"}
            ),
            Document(
                page_content="""
                Demographic Model 7 - K-Means with k=5:
                - Age is the strongest differentiator (Cramer's V: 0.41)
                - Kid category shows high cluster purity (75.3%)
                - Income has weak association with clusters
                - Clusters show moderate imbalance
                
                Interpretation: Age and family structure (kids) are key demographic segments.
                """,
                metadata={"model": "demographic_kmeans", "type": "cluster_summary"}
            ),
            Document(
                page_content="""
                Behavioral Model 13 - K-Means with k=2:
                - Time of day is the primary differentiator
                - Morning, evening, and night proportions show medium effects
                - Campaign exposure and discount usage show negligible differences
                
                Clusters: 
                1. Morning/afternoon shoppers (82% of customers)
                2. Evening/night shoppers (18% of customers)
                """,
                metadata={"model": "behavioral_kmeans", "type": "cluster_summary"}
            ),
            Document(
                page_content="""
                Product Preference Model 19 - K-Means with k=2:
                - Seafood is the main differentiator (12% variance explained)
                - All clusters prefer National brands over Private brands
                - Very small differences in category proportions
                
                Clusters:
                1. General grocery focused (74% of customers)
                2. Fresh food focused (26% of customers)
                """,
                metadata={"model": "product_kmeans", "type": "cluster_summary"}
            ),
            Document(
                page_content="""
                Marketing Strategy Recommendations:
                
                For High-Value Customers (RFM Cluster 3):
                - VIP loyalty programs with exclusive benefits
                - Early access to new products
                - Personalized recommendations
                - Premium customer service
                
                For At-Risk Customers (RFM Cluster 2):
                - Re-engagement campaigns
                - Win-back offers and discounts
                - Survey to understand churn reasons
                - Targeted email campaigns
                """,
                metadata={"type": "marketing_strategy"}
            ),
            Document(
                page_content="""
                Validation Metrics Summary:
                
                Best performing models:
                1. RFM K-Means (k=4): Silhouette 0.30, Davies-Bouldin 1.03
                2. Product HDBSCAN: DBCV 0.68, Silhouette 0.48
                3. Behavioral Hierarchical Complete: Silhouette 0.51
                
                Key insight: RFM and product preference show strongest natural clustering.
                Demographic variables alone produce weaker clusters.
                """,
                metadata={"type": "validation_metrics"}
            )
        ]
        
        # Create vector store with error handling
        try:
            print("🔄 Creating vector embeddings (this may take 30-60 seconds)...")
            print("   Calling OpenAI API to create embeddings...")
            
            # Store documents in-memory with their embeddings
            self.knowledge_documents = cluster_documents
            self.document_embeddings = []
            
            # Embed documents incrementally
            for i, doc in enumerate(cluster_documents, 1):
                print(f"   Embedding document {i}/{len(cluster_documents)}...", end="\r")
                embedding = self.embeddings.embed_query(doc.page_content)
                self.document_embeddings.append(embedding)
            
            print("✅ Knowledge base initialized with 6 cluster analysis documents")
            
        except Exception as e:
            print(f"\n❌ Error creating embeddings: {e}")
            print("\nPossible issues:")
            print("  1. API key invalid or expired")
            print("  2. No credits in your OpenAI account")
            print("  3. Network connection problem")
            print("  4. Rate limit exceeded")
            print("\n💡 Try running: python test_api.py")
            raise
    
    def _create_few_shot_prompt(self) -> FewShotPromptTemplate:
        """Create few-shot prompt template for consistent responses"""
        
        examples = [
            {
                "query": "What are the main customer segments?",
                "answer": """Based on the RFM analysis, we have identified 4 main customer segments:

1. **Champions** (Cluster 3): Very high frequency shoppers (52 transactions) with high spending ($687). They shop frequently and recently (85 days ago).

2. **Loyal Customers** (Cluster 1): Regular shoppers (28 transactions) with moderate spending ($412). Good engagement level.

3. **At-Risk** (Cluster 2): Low frequency (9 transactions) with long time since last purchase (210 days). Need re-engagement.

4. **Potential Loyalists** (Cluster 4): Medium frequency (18 transactions) who could be nurtured into loyal customers.

These segments explain 54% (monetary) and 45% (frequency) of customer behavior variance."""
            },
            {
                "query": "How should we market to high-value customers?",
                "answer": """For high-value customers (Champions - Cluster 3), I recommend:

**Retention Strategies:**
- Launch a VIP loyalty program with exclusive perks
- Provide early access to new products and sales
- Offer personalized product recommendations based on purchase history
- Assign dedicated customer service representatives

**Communication:**
- Send personalized thank-you messages
- Request feedback and product reviews
- Invite to exclusive events or webinars
- Share insider tips and content

**Incentives:**
- Points-based rewards program
- Free shipping and priority delivery
- Special birthday/anniversary offers
- Referral bonuses

Focus on making them feel valued and maintaining their engagement."""
            },
            {
                "query": "What time do most customers shop?",
                "answer": """Based on the behavioral analysis:

**Primary Shopping Pattern:**
- 82% of customers are **morning/afternoon shoppers**
- 18% are **evening/night shoppers**

**Implications:**
- Schedule email campaigns for morning hours (8-10 AM)
- Stock inventory and staff accordingly
- Consider extended evening hours for the night shopper segment
- Run flash sales during peak afternoon times (2-4 PM)

Time of day shows medium effect in differentiating customer behavior, making it a useful targeting dimension."""
            }
        ]
        
        example_template = PromptTemplate(
            template="Question: {query}\nExpert Answer: {answer}",
            input_variables=["query", "answer"]
        )
        
        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_template,
            prefix="""You are an expert customer segmentation analyst. You provide clear, actionable insights based on clustering analysis.
Use the retrieved context to answer questions accurately. Structure your responses with bullet points and sections when appropriate.""",
            suffix="""
Retrieved Context:
{context}

Question: {query}
Expert Answer:""",
            input_variables=["context", "query"]
        )
        
        return few_shot_prompt
    
    def query(self, question: str, use_rag: bool = True) -> str:
        """
        Query the agent with optional RAG
        
        Args:
            question: User's question
            use_rag: Whether to use RAG (Retrieval Augmented Generation)
        
        Returns:
            Agent's response
        """
        
        if use_rag and self.vectorstore:
            # Retrieve relevant documents
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
            docs = retriever.invoke(question)
            
            # Combine retrieved context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Use few-shot prompt with context
            few_shot_prompt = self._create_few_shot_prompt()
            formatted_prompt = few_shot_prompt.format(
                context=context,
                query=question
            )
            
            # Invoke with message format
            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            
            # Add to chat history
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": response.content})
            
            return response.content
        else:
            # Direct query without RAG
            response = self.llm.invoke([HumanMessage(content=question)])
            
            # Add to chat history
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": response.content})
            
            return response.content
    
    def semantic_search(self, query: str, k: int = 3) -> List[Document]:
        """Perform semantic search on cluster knowledge base using in-memory embeddings"""
        if not hasattr(self, 'knowledge_documents') or not self.document_embeddings:
            return []
        
        import math
        
        # Embed the query
        query_embedding = self.embeddings.embed_query(query)
        
        # Calculate cosine similarity manually
        def cosine_similarity(a, b):
            """Calculate cosine similarity between two vectors"""
            dot_product = sum(x * y for x, y in zip(a, b))
            mag_a = math.sqrt(sum(x * x for x in a))
            mag_b = math.sqrt(sum(x * x for x in b))
            if mag_a == 0 or mag_b == 0:
                return 0
            return dot_product / (mag_a * mag_b)
        
        # Calculate similarity with all documents
        similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in self.document_embeddings]
        
        # Get top k indices
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]
        
        # Return top documents
        results = [self.knowledge_documents[i] for i in top_indices]
        return results
    
    def switch_model(self, model_choice: str):
        """Switch to a different LLM"""
        if model_choice in ["openai", "gemini", "grok"]:
            self.llm = self._initialize_llm(model_choice)
            self.model_choice = model_choice
            print(f"✅ Switched to {model_choice}")
        else:
            print(f"❌ Invalid model choice. Use: openai, gemini, or grok")
    
    def get_conversation_history(self) -> str:
        """Get the conversation history"""
        if not self.chat_history:
            return ""
        
        history = []
        for msg in self.chat_history:
            prefix = "You: " if msg["role"] == "user" else "Agent: "
            history.append(f"{prefix}{msg['content']}")
        
        return "\n\n".join(history)


def main():
    """Main interactive loop"""
    
    print("=" * 60)
    print("🤖 Customer Segmentation AI Agent")
    print("=" * 60)
    print("\nFeatures:")
    print("  ✓ Multiple LLMs (OpenAI, Gemini, Grok)")
    print("  ✓ Few-shot prompting for consistent analysis")
    print("  ✓ RAG with semantic search")
    print("  ✓ Conversation memory")
    print("\nAvailable Commands:")
    print("  /model <openai|gemini|grok> - Switch LLM")
    print("  /search <query> - Semantic search")
    print("  /history - View conversation history")
    print("  /help - Show this help")
    print("  /quit - Exit")
    print("=" * 60)
    
    # Initialize agent
    model = input("\nChoose model (openai/gemini/grok) [openai]: ").strip().lower()
    if not model:
        model = "openai"
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY not found in .env file")
        print("\n📝 Create a .env file with:")
        print("OPENAI_API_KEY=your-api-key-here")
        print("\nGet your API key from:")
        print("  • OpenAI: https://platform.openai.com/api-keys")
        print("  • OpenRouter: https://openrouter.ai/keys")
        return
    
    try:
        agent = SegmentationAgent(model_choice=model)
        print(f"\n✅ Agent initialized with {model}")
    except Exception as e:
        print(f"\n❌ Error initializing agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n💬 Start chatting! (type /help for commands)\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                
                if command == "/quit":
                    print("\n👋 Goodbye!")
                    break
                
                elif command == "/help":
                    print("\nCommands:")
                    print("  /model <openai|gemini|grok> - Switch LLM")
                    print("  /search <query> - Semantic search")
                    print("  /history - View conversation history")
                    print("  /help - Show this help")
                    print("  /quit - Exit\n")
                
                elif command == "/model":
                    if len(parts) > 1:
                        agent.switch_model(parts[1])
                    else:
                        print(f"Current model: {agent.model_choice}")
                
                elif command == "/search":
                    if len(parts) > 1:
                        query = parts[1]
                        results = agent.semantic_search(query)
                        print(f"\n🔍 Search results for: '{query}'")
                        for i, doc in enumerate(results, 1):
                            print(f"\n--- Result {i} ---")
                            print(doc.page_content[:300] + "...")
                    else:
                        print("Usage: /search <query>")
                
                elif command == "/history":
                    history = agent.get_conversation_history()
                    if history:
                        print("\n📜 Conversation History:")
                        print("-" * 60)
                        print(history)
                        print("-" * 60)
                    else:
                        print("\nNo conversation history yet.")
                
                else:
                    print(f"Unknown command: {command}")
                
                continue
            
            # Regular query with RAG
            print("\n🤔 Thinking...")
            response = agent.query(user_input, use_rag=True)
            print(f"\n🤖 Agent: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()