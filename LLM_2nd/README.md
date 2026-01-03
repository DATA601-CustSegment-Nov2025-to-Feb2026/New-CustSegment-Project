# Customer Segmentation AI Agents

AI-powered customer segmentation analysis using LangChain, OpenAI/Gemini/Grok, and RAG (Retrieval Augmented Generation).

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the `LLM_2nd` directory:

```env
OPENAI_API_KEY=your-api-key-here
OPENAI_API_BASE=https://api.openai.com/v1  # Optional: for custom endpoints
```

**Get API keys from:**
- **OpenAI**: https://platform.openai.com/api-keys (requires paid account)
- **OpenRouter**: https://openrouter.ai/keys (supports multiple models)

### 3. Verify Setup

Run the test scripts to ensure everything is configured correctly:

```bash
# Test API connection
python test_api.py

# Test embeddings
python test_embedding.py

# Test Chroma vector database
python test_chroma.py
```

### 4. Run the Agents

**Full-featured agent with RAG:**
```bash
python segmentation_agent.py
```

**Lightweight agent (no embeddings):**
```bash
python simple_agent.py
```

---

## 📁 Files Overview

### Main Agent Files

#### `segmentation_agent.py` - Full-Featured RAG Agent
Advanced customer segmentation agent with semantic search and vector database.

**Features:**
- ✅ Retrieval Augmented Generation (RAG) with Chroma
- ✅ Semantic search on cluster knowledge base
- ✅ Few-shot prompting for consistent responses
- ✅ Multi-LLM support (OpenAI, Gemini, Grok)
- ✅ Conversational memory
- ✅ 6 built-in cluster analysis documents

**Commands:**
```
/model <openai|gemini|grok>  - Switch to different LLM
/search <query>              - Semantic search knowledge base
/history                     - View conversation history
/help                        - Show help menu
/quit                        - Exit
```

**Example Usage:**
```bash
$ python segmentation_agent.py
Choose model (openai/gemini/grok) [openai]: openai

You: What are the main customer segments?
🤖 Agent: Based on the RFM analysis, we have identified 4 main customer segments...

You: /search high-value customers
🔍 Search results for: 'high-value customers'
```

**Requirements:**
- OpenAI API key (costs money for embeddings)
- 30-60 seconds for initialization (creates embeddings)

---

#### `simple_agent.py` - Lightweight Agent
Lightweight agent without vector database for quick testing.

**Features:**
- ✅ Fast startup (no embedding API calls)
- ✅ Static knowledge base (hardcoded)
- ✅ Multi-LLM support
- ✅ Conversational memory
- ✅ No additional dependencies

**Commands:**
```
/history  - View conversation history
/help     - Show help menu
/quit     - Exit
```

**Example Usage:**
```bash
$ python simple_agent.py
💬 Start chatting! (type /help for commands)

You: How should I market to high-value customers?
🤖 Agent: For high-value customers (Champions), I recommend...
```

**Best For:**
- Quick testing without API costs
- Demonstrating agent functionality
- Low-latency responses

---

### Test & Setup Files

#### `test_api.py` - API Configuration Checker
**Purpose:** Verify API keys and LLM connection.

**What it tests:**
- ✅ `.env` file exists and is readable
- ✅ `OPENAI_API_KEY` is set correctly
- ✅ API provider detection (OpenAI Direct vs OpenRouter)
- ✅ Actual API connection with test request

**Run it:**
```bash
python test_api.py
```

**Output Example:**
```
🔍 API Configuration Checker
✅ .env file found
✅ OPENAI_API_KEY found: sk-proj-...abc123
✅ Detected Provider: OpenAI Direct
✅ SUCCESS! API is working!
📨 Response: API connection successful!
```

**Troubleshooting:**
If this fails, check:
1. `.env` file exists in the same directory
2. API key format: `sk-proj-...` for OpenAI or `sk-or-...` for OpenRouter
3. No quotes around the key in `.env`
4. API key has sufficient credits

---

#### `test_setup.py` - Alternative Setup Verification
**Purpose:** Same as `test_api.py` with slightly different output format.

**Run it:**
```bash
python test_setup.py
```

Use this as an alternative if `test_api.py` has issues.

---

#### `test_embedding.py` - Embeddings Test
**Purpose:** Test OpenAI embeddings initialization and vector creation.

**What it tests:**
- ✅ `OpenAIEmbeddings` initialization
- ✅ Creating a sample embedding vector
- ✅ Embedding timeout handling

**Run it:**
```bash
python test_embedding.py
```

**Expected Output:**
```
Starting embedding test...
Initializing OpenAI embeddings...
✅ Embeddings initialized successfully!
Testing embedding a simple text...
✅ Embedding successful! Vector size: 1536
✅ Test completed!
```

**Note:** This test may take 10-20 seconds due to API calls.

---

#### `test_chroma.py` - Vector Database Test
**Purpose:** Test Chroma vector store creation and document embedding.

**What it tests:**
- ✅ Creating a Chroma in-memory database
- ✅ Embedding multiple documents
- ✅ Storing documents in vector format
- ✅ Timeout handling (60-second limit)

**Run it:**
```bash
python test_chroma.py
```

**Expected Output:**
```
Starting Chroma test...
Initializing embeddings...
✅ Embeddings initialized
Creating Chroma vector store...
This will embed 3 documents...
✅ Vector store created successfully!
Number of documents in store: 3
✅ Test completed!
```

**Note:** This is the most comprehensive test - it combines embeddings + vector database.

---

#### `test_input.txt` - Sample Input File
**Purpose:** Automated test input for the agents.

**Contents:**
- Model choice: `openai`
- Sample question: `What are the main customer segments?`
- Exit command: `/quit`

**Usage:**
```bash
python segmentation_agent.py < test_input.txt
```

---

### Configuration Files

#### `requirements.txt`
**Purpose:** Python package dependencies.

**Packages:**
- `langchain` - AI agent framework
- `langchain-openai` - OpenAI integration
- `langchain-community` - Community integrations
- `chromadb` - Vector database
- `python-dotenv` - Environment variable management
- `openai` - OpenAI API client

**Install all:**
```bash
pip install -r requirements.txt
```

---

## 🔧 Environment Setup

### Creating `.env` File

1. Create a file named `.env` in the `LLM_2nd` directory
2. Add your API keys:

```env
# Required
OPENAI_API_KEY=sk-proj-your-key-here

# Optional (for OpenRouter or custom endpoints)
OPENAI_API_BASE=https://api.openai.com/v1
```

### Important Notes

⚠️ **Security:**
- **Never commit `.env` to GitHub** - it contains sensitive keys
- `.env` should be in `.gitignore` (already configured)
- If `.env` is accidentally committed, regenerate all keys immediately

✅ **Verification:**
- Run `test_api.py` to verify setup
- Check API key doesn't have quotes: `OPENAI_API_KEY=sk-...` (NOT `"sk-..."`)

---

## 📋 Recommended Setup Order

For new team members:

1. **Clone the repository**
   ```bash
   git clone https://github.com/...
   cd LLM_2nd
   ```

2. **Create `.env` file**
   - Add your API key

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run tests in this order**
   ```bash
   python test_api.py          # 1. Basic API check (5 sec)
   python test_embedding.py    # 2. Embeddings test (15 sec)
   python test_chroma.py       # 3. Vector DB test (45 sec)
   ```

5. **Try the agents**
   ```bash
   python simple_agent.py      # Start here (lightweight)
   python segmentation_agent.py # Then try RAG version
   ```

---

## 🎯 Use Cases

### For Quick Testing
- Use `simple_agent.py` - no API costs, instant startup
- Good for demos and development

### For Production
- Use `segmentation_agent.py` with RAG
- Better responses using real data
- Semantic search capabilities

### For Multiple LLMs
Both agents support:
- OpenAI (gpt-4o-mini, gpt-4o)
- Google Gemini (requires OpenRouter)
- X.AI Grok (requires OpenRouter)

Switch models with `/model <name>` command.

---

## 🚨 Troubleshooting

### "OPENAI_API_KEY not found"
- ✅ Check `.env` file exists in `LLM_2nd` directory
- ✅ Verify key format (no quotes, no spaces)
- ✅ Run `test_api.py` for detailed diagnostics

### "API connection failed / 401 error"
- ✅ Verify API key is active (not expired/revoked)
- ✅ Check account has credits (if using OpenAI)
- ✅ Try running `test_api.py`

### "Timeout during initialization"
- ✅ This is normal for `segmentation_agent.py` (30-60 sec)
- ✅ For faster startup, use `simple_agent.py`
- ✅ Run `test_embedding.py` and `test_chroma.py` separately

### "Module not found errors"
- ✅ Install requirements: `pip install -r requirements.txt`
- ✅ Verify Python 3.8+ installed
- ✅ Check virtual environment is activated

### "ModuleNotFoundError: No module named 'chromadb'"
- ✅ Run: `pip install chromadb`
- ✅ Or reinstall all: `pip install -r requirements.txt`

---

## 📊 Knowledge Base Contents

The agents include knowledge about:

**RFM Segmentation** (4 clusters):
- Champions: High frequency, high value, recent
- Loyal Customers: Regular shoppers, moderate spend
- At-Risk: Low frequency, long inactivity
- Potential Loyalists: Medium frequency, growth potential

**Demographics:**
- Age (strong differentiator)
- Family structure
- Income levels

**Behavioral Patterns:**
- Shopping times (morning/afternoon vs evening/night)
- Product preferences (fresh vs general grocery)

**Marketing Strategies:**
- VIP programs for Champions
- Re-engagement for At-Risk
- Incentives for Potential Loyalists

---

## 🤝 Contributing

When adding new files:
1. Update this README
2. Add appropriate `.gitignore` entries
3. Include test scripts for new features
4. Document API costs if applicable

---

## 📄 License

See LICENSE file in repository root.

---

## 📞 Support

For issues:
1. Run the test scripts to diagnose
2. Check this README's Troubleshooting section
3. Review error messages carefully
4. Contact team lead with test output

---

## 🎓 Example Queries

Try these questions with the agents:

1. **"What are the main customer segments?"**
   - Returns RFM cluster breakdown

2. **"How should I market to high-value customers?"**
   - Provides VIP marketing strategies

3. **"When do most customers shop?"**
   - Shows behavioral patterns

4. **"What products do customers prefer?"**
   - Details product preferences by segment

5. **"How can I re-engage at-risk customers?"**
   - Win-back campaign recommendations

---

**Last Updated:** January 2026
**Version:** 1.0
