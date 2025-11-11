# Environment Setup Guide

## Groq API Key Configuration

The application requires a Groq API key for AI-powered video summarization and Q&A features.

### Step 1: Get Your Groq API Key

1. Visit [Groq Console](https://console.groq.com/keys)
2. Sign up or log in
3. Create a new API key
4. Copy the key (it starts with `gsk_`)

### Step 2: Set Up Environment Variables

Create a `.env` file in the root directory of the project:

```bash
# Copy the example file
cp .env.example .env
```

Then edit `.env` and add your API key:

```
GROQ_API_KEY=your_actual_groq_api_key_here
```

### Step 3: Verify Setup

The backend will automatically load the `.env` file when it starts. If the API key is missing, you'll see an error:

```
ValueError: GROQ_API_KEY environment variable is not set.
```

### Security Notes

- **Never commit `.env` to git** - it's already in `.gitignore`
- The `.env.example` file shows the structure without actual secrets
- Each developer/deployment needs their own `.env` file

### For Production

Set the environment variable in your hosting platform:

**Railway/Render/Heroku:**
```
GROQ_API_KEY=your_key_here
```

**Docker:**
```bash
docker run -e GROQ_API_KEY=your_key_here ...
```

**Systemd service:**
```ini
[Service]
Environment="GROQ_API_KEY=your_key_here"
```
