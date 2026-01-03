"""
Simple Customer Segmentation Agent - No Embeddings Required
This version works without vector database to avoid API/setup issues
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

class SimpleSegmentationAgent:
    """Lightweight agent without RAG - just LLM with context"""
    
    def __init__(self, model_choice: str = "openai"):
        """Initialize with LLM only"""
        self.model_choice = model_choice
        self.llm = self._initialize_llm(model_choice)
        self.chat_history = []
        
        # Static knowledge base (no embeddings needed)
        self.knowledge = """
        CUSTOMER SEGMENTATION KNOWLEDGE BASE:
        
        RFM ANALYSIS (4 Clusters):
        1. Champions (Cluster 3): Very high frequency (52 transactions), high spending ($687), recent (85 days)
        2. Loyal Customers (Cluster 1): Regular shoppers (28 transactions), moderate spending ($412)
        3. At-Risk (Cluster 2): Low frequency (9 transactions), long absence (210 days), low spending ($115)
        4. Potential Loyalists (Cluster 4): Medium frequency (18 transactions), moderate spending ($298)
        
        DEMOGRAPHIC INSIGHTS:
        - Age is strongest differentiator (Cramer's V: 0.41)
        - Family structure (kids) shows high cluster purity (75.3%)
        - 5 distinct demographic segments identified
        
        BEHAVIORAL PATTERNS:
        - 82% shop morning/afternoon
        - 18% shop evening/night
        - Time of day is primary differentiator
        
        PRODUCT PREFERENCES:
        - 74% general grocery focused
        - 26% fresh food focused
        - Seafood is main differentiator (12% variance)
        
        MARKETING STRATEGIES:
        
        For High-Value Customers (Champions):
        - VIP loyalty programs with exclusive benefits
        - Early access to new products
        - Personalized recommendations
        - Premium customer service
        
        For At-Risk Customers:
        - Re-engagement campaigns
        - Win-back offers and discounts
        - Survey to understand churn reasons
        - Targeted email campaigns
        
        For Potential Loyalists:
        - Incentive programs to increase frequency
        - Educational content about products
        - Special promotions to boost spending
        """
        
        print("✅ Simple agent initialized (no embeddings required)")
    
    def _initialize_llm(self, model_choice: str) -> ChatOpenAI:
        """Initialize LLM"""
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        
        models = {
            "openai": {"model": "gpt-4o-mini", "use_base": False},
            "gemini": {"model": "google/gemini-2.0-flash-exp", "use_base": True},
            "grok": {"model": "x-ai/grok-2-1212", "use_base": True}
        }
        
        config = models.get(model_choice, models["openai"])
        
        kwargs = {
            "model": config["model"],
            "api_key": api_key,
            "temperature": 0.7
        }
        
        if config["use_base"] and base_url:
            kwargs["base_url"] = base_url
        
        return ChatOpenAI(**kwargs)
    
    def query(self, question: str) -> str:
        """Query the agent"""
        
        # Create system message with knowledge
        system_msg = SystemMessage(content=f"""You are an expert customer segmentation analyst.
        
Use this knowledge base to answer questions:

{self.knowledge}

Provide clear, actionable insights. Structure responses with sections and bullet points when appropriate.""")
        
        # Create user message
        user_msg = HumanMessage(content=question)
        
        # Get response
        response = self.llm.invoke([system_msg, user_msg])
        
        # Save to history
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": response.content})
        
        return response.content
    
    def get_history(self) -> str:
        """Get conversation history"""
        if not self.chat_history:
            return "No conversation history yet."
        
        history = []
        for msg in self.chat_history:
            prefix = "You: " if msg["role"] == "user" else "Agent: "
            history.append(f"{prefix}{msg['content']}")
        
        return "\n\n".join(history)


def main():
    """Main loop"""
    
    print("=" * 60)
    print("🤖 Simple Customer Segmentation Agent")
    print("=" * 60)
    print("\nFeatures:")
    print("  ✓ Fast startup (no embeddings)")
    print("  ✓ Built-in cluster knowledge")
    print("  ✓ Conversation history")
    print("\nCommands:")
    print("  /history - View conversation")
    print("  /help - Show this help")
    print("  /quit - Exit")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY not found in .env file")
        print("Run: python test_api.py")
        return
    
    # Initialize agent
    try:
        agent = SimpleSegmentationAgent(model_choice="openai")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Run: python test_api.py")
        return
    
    print("\n💬 Start chatting! (type /help for commands)\n")
    
    # Example questions
    print("💡 Try asking:")
    print("  • What are the main customer segments?")
    print("  • How should I market to high-value customers?")
    print("  • When do most customers shop?")
    print("  • What products do customers prefer?\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower()
                
                if command == "/quit":
                    print("\n👋 Goodbye!")
                    break
                
                elif command == "/help":
                    print("\nCommands:")
                    print("  /history - View conversation")
                    print("  /help - Show this help")
                    print("  /quit - Exit\n")
                
                elif command == "/history":
                    print("\n📜 Conversation History:")
                    print("-" * 60)
                    print(agent.get_history())
                    print("-" * 60 + "\n")
                
                else:
                    print(f"Unknown command: {command}\n")
                
                continue
            
            # Regular query
            print("\n🤔 Thinking...")
            response = agent.query(user_input)
            print(f"\n🤖 Agent:\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
