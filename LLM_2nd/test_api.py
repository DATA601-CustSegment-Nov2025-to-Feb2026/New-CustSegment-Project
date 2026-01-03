"""
API Configuration Test Script
Run this to verify your .env setup is correct
"""

import os
from dotenv import load_dotenv

print("=" * 60)
print("🔍 API Configuration Checker")
print("=" * 60)

# Load .env file
print("\n1️⃣ Loading .env file...")
load_dotenv()

# Check if .env exists
if not os.path.exists('.env'):
    print("❌ ERROR: .env file not found!")
    print("\n📝 Create a .env file with:")
    print("OPENAI_API_KEY=your-api-key-here")
    print("\nGet API key from:")
    print("  • OpenAI: https://platform.openai.com/api-keys")
    print("  • OpenRouter: https://openrouter.ai/keys")
    exit(1)

print("✅ .env file found")

# Check API key
print("\n2️⃣ Checking API key...")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ ERROR: OPENAI_API_KEY not set in .env")
    print("\nYour .env file should look like:")
    print("OPENAI_API_KEY=sk-proj-...")
    print("\nMake sure:")
    print("  • No quotes around the key")
    print("  • No spaces")
    print("  • File is named exactly .env")
    exit(1)

print(f"✅ OPENAI_API_KEY found: {api_key[:15]}...{api_key[-4:]}")

# Check API base (optional)
print("\n3️⃣ Checking API base...")
api_base = os.getenv("OPENAI_API_BASE")
if api_base:
    print(f"✅ OPENAI_API_BASE set: {api_base}")
    if "openrouter" in api_base.lower():
        print("   → Using OpenRouter (multi-model access)")
else:
    print("ℹ️  OPENAI_API_BASE not set")
    print("   → Will use OpenAI default endpoint")

# Detect provider
print("\n4️⃣ Detecting provider...")
if api_key.startswith("sk-proj-"):
    provider = "OpenAI Direct"
    models_available = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
elif api_key.startswith("sk-or-"):
    provider = "OpenRouter"
    models_available = ["gpt-4o-mini", "google/gemini-2.0-flash-exp", "x-ai/grok-2-1212"]
else:
    provider = "Unknown (check your key format)"
    models_available = []

print(f"✅ Detected Provider: {provider}")

# Test the API
print("\n" + "=" * 60)
print("🧪 Testing API Connection...")
print("=" * 60)

try:
    from langchain_openai import ChatOpenAI
    
    # Configure based on provider
    print("\n📡 Sending test request...")
    
    if api_base:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            base_url=api_base,
            temperature=0.7
        )
    else:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.7
        )
    
    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content="Reply with exactly: 'API connection successful!'")])
    
    print("\n✅ SUCCESS! API is working!")
    print(f"📨 Response: {response.content}")
    
    print("\n" + "=" * 60)
    print("🎉 Configuration Complete!")
    print("=" * 60)
    
    if models_available:
        print("\n📋 Available models:")
        for model in models_available:
            print(f"   • {model}")
    
    print("\n🚀 You can now run:")
    print("   python segmentation_agent.py")
    
except ImportError as e:
    print("\n❌ Missing package!")
    print(f"Error: {e}")
    print("\n📦 Install with:")
    print("   pip install langchain-openai python-dotenv")
    
except Exception as e:
    print(f"\n❌ API Test Failed!")
    print(f"Error: {e}")
    
    print("\n🔍 Common Issues:")
    print("1. Invalid API key - check your key is correct")
    print("2. No credits - add funds to your account")
    print("3. Wrong API base - check OPENAI_API_BASE URL")
    
    if "401" in str(e) or "authentication" in str(e).lower():
        print("\n💡 This looks like an authentication error")
        print("   → Verify your API key is correct")
        print("   → Check key has not expired")
        print("   → Make sure there are no quotes around the key in .env")
        
    if "404" in str(e) or "not found" in str(e).lower():
        print("\n💡 This looks like a model/endpoint error")
        print("   → Check OPENAI_API_BASE is correct")
        print("   → For OpenRouter use: https://openrouter.ai/api/v1")
    
    exit(1)

print("\n✅ Setup complete! Ready to build AI agents! 🤖")
"""
API Configuration Test Script
Run this to verify your .env setup is correct
"""

import os
from dotenv import load_dotenv

print("=" * 60)
print("🔍 API Configuration Checker")
print("=" * 60)

# Load .env file
load_dotenv()

# Check if .env exists
if not os.path.exists('.env'):
    print("\n❌ ERROR: .env file not found!")
    print("\n📝 Create a .env file with:")
    print("OPENAI_API_KEY=your-api-key-here")
    print("\nGet API key from:")
    print("  • OpenAI: https://platform.openai.com/api-keys")
    print("  • OpenRouter: https://openrouter.ai/keys")
    exit(1)

print("\n✅ .env file found")

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("\n❌ ERROR: OPENAI_API_KEY not set in .env")
    print("\nYour .env file should contain:")
    print("OPENAI_API_KEY=sk-...")
    exit(1)

print(f"✅ OPENAI_API_KEY found: {api_key[:20]}...{api_key[-4:]}")

# Check API base (optional)
api_base = os.getenv("OPENAI_API_BASE")
if api_base:
    print(f"✅ OPENAI_API_BASE set: {api_base}")
    if "openrouter" in api_base.lower():
        print("   → Using OpenRouter (multi-model access)")
    else:
        print(f"   → Using custom base: {api_base}")
else:
    print("ℹ️  OPENAI_API_BASE not set (using OpenAI default)")

# Detect provider
if api_key.startswith("sk-proj-"):
    provider = "OpenAI Direct"
    models_available = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
elif api_key.startswith("sk-or-"):
    provider = "OpenRouter"
    models_available = ["gpt-4o-mini", "google/gemini-2.0-flash-exp", "x-ai/grok-2-1212"]
else:
    provider = "Unknown"
    models_available = []

print(f"\n🔧 Detected Provider: {provider}")

# Test the API
print("\n" + "=" * 60)
print("🧪 Testing API Connection...")
print("=" * 60)

try:
    from langchain_openai import ChatOpenAI
    
    # Configure based on provider
    if api_base:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            base_url=api_base,
            temperature=0.7
        )
    else:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.7
        )
    
    print("📡 Sending test request...")
    response = llm.invoke("Reply with exactly: 'API connection successful!'")
    
    print("\n✅ SUCCESS! API is working!")
    print(f"📨 Response: {response.content}")
    
    print("\n" + "=" * 60)
    print("🎉 Configuration Complete!")
    print("=" * 60)
    
    if models_available:
        print("\n📋 Available models:")
        for model in models_available:
            print(f"   • {model}")
    
    print("\n🚀 You can now run:")
    print("   python segmentation_agent.py")
    
except ImportError as e:
    print("\n❌ Missing package!")
    print(f"Error: {e}")
    print("\n📦 Install with:")
    print("   pip install langchain-openai python-dotenv")
    
except Exception as e:
    print(f"\n❌ API Test Failed!")
    print(f"Error: {e}")
    
    print("\n🔍 Common Issues:")
    print("1. Invalid API key - check your key is correct")
    print("2. No credits - add funds to your account")
    print("3. Wrong API base - check OPENAI_API_BASE URL")
    
    if "401" in str(e) or "authentication" in str(e).lower():
        print("\n💡 This looks like an authentication error")
        print("   → Verify your API key is correct")
        print("   → Check key has not expired")
        
    if "404" in str(e) or "not found" in str(e).lower():
        print("\n💡 This looks like a model/endpoint error")
        print("   → Check OPENAI_API_BASE is correct")
        print("   → For OpenRouter use: https://openrouter.ai/api/v1")
        
    print("\n📚 See API_SETUP_GUIDE.md for detailed help")
    exit(1)

print("\n✅ Setup complete! Ready to build AI agents! 🤖")