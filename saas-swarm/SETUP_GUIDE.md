# Enhanced SaaS-Swarm Setup Guide

This guide will help you set up the enhanced SaaS-Swarm platform with OpenAI research capabilities and email sending functionality.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create Environment File
Copy the example environment file:
```bash
cp env.example .env
```

### 3. Configure Your Environment

Edit the `.env` file with your credentials:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Email Configuration
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password_here
EMAIL_FROM_ADDRESS=your_email@gmail.com
EMAIL_USE_TLS=true
```

### 4. Test Your Setup
```bash
python3 test_enhanced_swarm.py
```

## 🔧 Detailed Configuration

### OpenAI Setup

1. **Get an OpenAI API Key:**
   - Go to [OpenAI Platform](https://platform.openai.com/)
   - Sign up or log in
   - Navigate to API Keys
   - Create a new API key
   - Copy the key to your `.env` file

2. **API Key Format:**
   ```
   OPENAI_API_KEY=sk-...your-key-here...
   ```

### Email Setup (Gmail Example)

1. **Enable 2-Factor Authentication:**
   - Go to your Google Account settings
   - Enable 2-Step Verification

2. **Generate App Password:**
   - Go to [Google App Passwords](https://myaccount.google.com/apppasswords)
   - Select "Mail" and your device
   - Generate the password
   - Use this password in your `.env` file

3. **Email Configuration:**
   ```env
   EMAIL_SMTP_SERVER=smtp.gmail.com
   EMAIL_SMTP_PORT=587
   EMAIL_USERNAME=your_email@gmail.com
   EMAIL_PASSWORD=your_16_character_app_password
   EMAIL_FROM_ADDRESS=your_email@gmail.com
   EMAIL_USE_TLS=true
   ```

### Other Email Providers

#### Outlook/Hotmail:
```env
EMAIL_SMTP_SERVER=smtp-mail.outlook.com
EMAIL_SMTP_PORT=587
EMAIL_USE_TLS=true
```

#### Yahoo:
```env
EMAIL_SMTP_SERVER=smtp.mail.yahoo.com
EMAIL_SMTP_PORT=587
EMAIL_USE_TLS=true
```

#### Custom SMTP:
```env
EMAIL_SMTP_SERVER=your.smtp.server.com
EMAIL_SMTP_PORT=587
EMAIL_USE_TLS=true
```

## 🧪 Testing Your Setup

### 1. Configuration Test
```bash
python3 test_enhanced_swarm.py
```

Expected output:
```
✅ Configuration is valid!
✅ All advanced tools are available!
✅ Swarm created successfully!
✅ Swarm execution completed!
```

### 2. Run Enhanced Email Writer
```bash
python3 saas_swarm/examples/enhanced_email_writer.py
```

### 3. Test Individual Components

#### Test OpenAI Research:
```python
from saas_swarm.tools.registry import create_default_tool_registry

registry = create_default_tool_registry()
result = await registry.call_tool("openai_research", "artificial intelligence trends")
print(result)
```

#### Test Email Sending:
```python
result = await registry.call_tool(
    "send_email",
    to_address="test@example.com",
    subject="Test Email",
    body="This is a test email from SaaS-Swarm!"
)
print(result)
```

## 🔄 Feedback Loops

The enhanced system includes feedback loops that:

1. **Evaluate Research Quality:** Rate the quality of OpenAI research
2. **Assess Email Effectiveness:** Track email delivery and response rates
3. **Adapt Agent Behavior:** Adjust agent parameters based on feedback
4. **Learn from Interactions:** Improve future research and email generation

### Using Feedback Loops

```python
from saas_swarm.core.feedback_loop import FeedbackLoop

# Create feedback loop
feedback_loop = FeedbackLoop()

# Evaluate research quality
evaluation = await feedback_loop.evaluate_swarm_output({
    'research_quality': 0.8,
    'email_sent': True,
    'response_rate': 0.6
})

# Propagate feedback to agents
await feedback_loop.propagate_feedback(evaluation)
```

## 🛠️ Customization

### Adding Custom Tools

1. **Create a new tool function:**
```python
async def my_custom_tool(input_data: str) -> dict:
    # Your custom logic here
    return {"result": "custom_output"}
```

2. **Register the tool:**
```python
registry.register_tool(
    "my_tool",
    my_custom_tool,
    "Description of my tool",
    is_async=True
)
```

### Custom Agent Inference

```python
async def my_agent_inference(input_data, tool_registry):
    # Use OpenAI for research
    research = await tool_registry.call_tool("openai_research", input_data)
    
    # Use custom tool
    custom_result = await tool_registry.call_tool("my_tool", research['research'])
    
    # Send email
    email_result = await tool_registry.call_tool("send_email", 
        to_address="recipient@example.com",
        subject="Custom Research",
        body=custom_result['result']
    )
    
    return email_result
```

## 🚨 Troubleshooting

### Common Issues

1. **"Missing required environment variables"**
   - Check your `.env` file exists
   - Ensure all required variables are set
   - Verify no extra spaces or quotes

2. **"OpenAI API key invalid"**
   - Verify your API key is correct
   - Check your OpenAI account has credits
   - Ensure the key has proper permissions

3. **"Email authentication failed"**
   - Verify your email credentials
   - Check if you're using an App Password (Gmail)
   - Ensure SMTP settings are correct

4. **"Tool not found"**
   - Check if the tool is registered
   - Verify the tool name is correct
   - Ensure the tool registry is properly initialized

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Testing Individual Components

```bash
# Test configuration only
python3 -c "from saas_swarm.config import Config; print(Config.validate())"

# Test tool registry only
python3 -c "from saas_swarm.tools.registry import create_default_tool_registry; registry = create_default_tool_registry(); print(registry.list_tools())"
```

## 📚 Next Steps

1. **Explore the API:** Start the FastAPI server
   ```bash
   python -m saas_swarm.api.main
   ```

2. **Use the CLI:** Manage swarms via command line
   ```bash
   swarm --help
   ```

3. **Create Custom Swarms:** Build your own agent swarms
   ```python
   from saas_swarm.examples.enhanced_email_writer import create_enhanced_email_writer_swarm
   swarm = create_enhanced_email_writer_swarm()
   ```

4. **Extend Functionality:** Add more tools and agents
   - Database integration
   - Web scraping
   - File processing
   - API integrations

## 🎯 Production Deployment

For production use:

1. **Use Environment Variables:** Don't commit `.env` files
2. **Secure Credentials:** Use a secrets manager
3. **Monitor Usage:** Track API calls and costs
4. **Error Handling:** Implement proper error recovery
5. **Rate Limiting:** Respect API rate limits
6. **Logging:** Add comprehensive logging

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your configuration is correct
3. Test individual components
4. Check the logs for detailed error messages
5. Ensure all dependencies are installed correctly 