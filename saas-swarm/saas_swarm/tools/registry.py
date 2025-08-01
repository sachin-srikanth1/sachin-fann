"""
ToolRegistry module for SaaS-Swarm platform.

Provides dynamic tool registration and dispatch capabilities
for agent tool integration.
"""

import asyncio
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass
import json
import time
import smtplib
from email.message import EmailMessage
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from ..config import Config


@dataclass
class ToolConfig:
    """Configuration for a tool."""
    name: str
    description: str
    function: Callable
    is_async: bool = False
    input_schema: Optional[Dict] = None
    output_schema: Optional[Dict] = None


class ToolRegistry:
    """
    Dynamic tool registry for agent tool integration.
    
    Supports:
    - Dynamic tool registration
    - Async and sync tool execution
    - Tool metadata and schemas
    - Tool usage tracking
    """
    
    def __init__(self):
        self.tools: Dict[str, ToolConfig] = {}
        self.usage_history: List[Dict] = []
        self.max_history_size = 1000
        
    def register_tool(self, name: str, function: Callable, description: str = "",
                     is_async: bool = False, input_schema: Optional[Dict] = None,
                     output_schema: Optional[Dict] = None) -> bool:
        """
        Register a new tool.
        
        Args:
            name: Tool name
            function: Tool function
            description: Tool description
            is_async: Whether the function is async
            input_schema: Input schema for validation
            output_schema: Output schema for validation
            
        Returns:
            True if registration successful
        """
        if name in self.tools:
            print(f"Warning: Tool '{name}' already registered, overwriting")
            
        tool_config = ToolConfig(
            name=name,
            description=description,
            function=function,
            is_async=is_async,
            input_schema=input_schema,
            output_schema=output_schema
        )
        
        self.tools[name] = tool_config
        print(f"Registered tool: {name}")
        return True
        
    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self.tools:
            del self.tools[name]
            print(f"Unregistered tool: {name}")
            return True
        return False
        
    async def call_tool(self, name: str, *args, **kwargs) -> Any:
        """
        Call a registered tool.
        
        Args:
            name: Tool name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Tool execution result
        """
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
            
        tool_config = self.tools[name]
        
        # Record usage
        usage_record = {
            'tool_name': name,
            'timestamp': time.time(),
            'args': args,
            'kwargs': kwargs
        }
        
        try:
            # Execute tool
            if tool_config.is_async:
                result = await tool_config.function(*args, **kwargs)
            else:
                result = tool_config.function(*args, **kwargs)
                
            usage_record['success'] = True
            usage_record['result'] = result
            
        except Exception as e:
            usage_record['success'] = False
            usage_record['error'] = str(e)
            raise
            
        finally:
            # Add to history
            self.usage_history.append(usage_record)
            
            # Trim history if too large
            if len(self.usage_history) > self.max_history_size:
                self.usage_history = self.usage_history[-self.max_history_size:]
                
        return result
        
    def __getitem__(self, name: str) -> Callable:
        """Get a tool function for direct calling."""
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found")
        return self.tools[name].function
        
    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self.tools
        
    def get_tool_info(self, name: str) -> Optional[Dict]:
        """Get information about a specific tool."""
        if name not in self.tools:
            return None
            
        tool_config = self.tools[name]
        return {
            'name': tool_config.name,
            'description': tool_config.description,
            'is_async': tool_config.is_async,
            'input_schema': tool_config.input_schema,
            'output_schema': tool_config.output_schema
        }
        
    def list_tools(self) -> List[str]:
        """Get list of registered tool names."""
        return list(self.tools.keys())
        
    def get_tool_usage_stats(self, name: str) -> Dict[str, Any]:
        """Get usage statistics for a specific tool."""
        tool_usage = [record for record in self.usage_history if record['tool_name'] == name]
        
        if not tool_usage:
            return {'tool_name': name, 'no_usage': True}
            
        success_count = sum(1 for record in tool_usage if record.get('success', False))
        total_count = len(tool_usage)
        
        return {
            'tool_name': name,
            'total_calls': total_count,
            'successful_calls': success_count,
            'success_rate': success_count / total_count if total_count > 0 else 0.0,
            'last_used': max(record['timestamp'] for record in tool_usage) if tool_usage else None
        }
        
    def clear_history(self):
        """Clear usage history."""
        self.usage_history.clear()
        
    def export_tools(self) -> Dict[str, Any]:
        """Export tool registry state."""
        return {
            'tools': {
                name: {
                    'description': config.description,
                    'is_async': config.is_async,
                    'input_schema': config.input_schema,
                    'output_schema': config.output_schema
                }
                for name, config in self.tools.items()
            },
            'usage_history_size': len(self.usage_history)
        }


# Built-in tools
class BuiltInTools:
    """Collection of built-in tools for common operations."""
    
    @staticmethod
    def text_summarizer(text: str, max_length: int = 100) -> str:
        """Summarize text to specified length."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
        
    @staticmethod
    def text_classifier(text: str, categories: List[str]) -> str:
        """Simple text classification based on keywords."""
        text_lower = text.lower()
        
        for category in categories:
            if category.lower() in text_lower:
                return category
                
        return categories[0] if categories else "unknown"
        
    @staticmethod
    def data_transformer(data: Any, transform_type: str) -> Any:
        """Transform data based on type."""
        if transform_type == "uppercase" and isinstance(data, str):
            return data.upper()
        elif transform_type == "lowercase" and isinstance(data, str):
            return data.lower()
        elif transform_type == "length" and isinstance(data, str):
            return len(data)
        elif transform_type == "reverse" and isinstance(data, str):
            return data[::-1]
        else:
            return data
            
    @staticmethod
    async def async_web_search(query: str) -> Dict[str, Any]:
        """Mock async web search tool."""
        # Simulate async operation
        await asyncio.sleep(0.1)
        
        return {
            'query': query,
            'results': [
                {'title': f'Result 1 for {query}', 'url': 'https://example1.com'},
                {'title': f'Result 2 for {query}', 'url': 'https://example2.com'}
            ],
            'total_results': 2
        }
        
    @staticmethod
    async def async_data_processor(data: List[Any], operation: str) -> List[Any]:
        """Mock async data processing tool."""
        await asyncio.sleep(0.05)
        
        if operation == "sort":
            return sorted(data)
        elif operation == "reverse":
            return list(reversed(data))
        elif operation == "unique":
            return list(set(data))
        else:
            return data


# Advanced tools with OpenAI and email
class AdvancedTools:
    """Advanced tools using OpenAI and email functionality."""
    
    @staticmethod
    async def openai_research(query: str, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """Research a topic using OpenAI."""
        try:
            # Initialize OpenAI client
            llm = ChatOpenAI(
                model=model,
                api_key=Config.OPENAI_API_KEY,
                temperature=0.7
            )
            
            # Create research prompt
            research_prompt = f"""
            Please research the following topic and provide a comprehensive analysis:
            
            Topic: {query}
            
            Please provide:
            1. Key findings and insights
            2. Recent developments
            3. Important trends
            4. Relevant statistics or data
            5. Summary of main points
            
            Format your response as a structured analysis.
            """
            
            # Get response from OpenAI
            messages = [HumanMessage(content=research_prompt)]
            response = await llm.ainvoke(messages)
            
            return {
                'query': query,
                'research': response.content,
                'model': model,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'query': query,
                'error': str(e),
                'status': 'error'
            }
    
    @staticmethod
    async def send_email(
        to_address: str,
        subject: str,
        body: str,
        from_address: Optional[str] = None,
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,
        use_tls: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Send an email using SMTP."""
        
        # Use config defaults if not provided
        email_config = Config.get_email_config()
        from_address = from_address or email_config['from_address']
        smtp_server = smtp_server or email_config['smtp_server']
        smtp_port = smtp_port or email_config['smtp_port']
        smtp_username = smtp_username or email_config['username']
        smtp_password = smtp_password or email_config['password']
        use_tls = use_tls if use_tls is not None else email_config['use_tls']
        
        def send():
            msg = EmailMessage()
            msg["From"] = from_address
            msg["To"] = to_address
            msg["Subject"] = subject
            msg.set_content(body)
            
            try:
                if use_tls:
                    server = smtplib.SMTP(smtp_server, smtp_port)
                    server.starttls()
                else:
                    server = smtplib.SMTP(smtp_server, smtp_port)
                
                if smtp_username and smtp_password:
                    server.login(smtp_username, smtp_password)
                
                server.send_message(msg)
                server.quit()
                return {"status": "sent", "to": to_address, "subject": subject}
                
            except Exception as e:
                return {"status": "error", "error": str(e)}
        
        # Run the blocking send in a thread
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, send)
    
    @staticmethod
    async def openai_generate_content(
        prompt: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """Generate content using OpenAI."""
        try:
            llm = ChatOpenAI(
                model=model,
                api_key=Config.OPENAI_API_KEY,
                temperature=0.7,
                max_tokens=max_tokens
            )
            
            messages = [HumanMessage(content=prompt)]
            response = await llm.ainvoke(messages)
            
            return {
                'prompt': prompt,
                'content': response.content,
                'model': model,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'prompt': prompt,
                'error': str(e),
                'status': 'error'
            }


def create_default_tool_registry() -> ToolRegistry:
    """Create a tool registry with built-in tools."""
    registry = ToolRegistry()
    
    # Register built-in tools
    registry.register_tool(
        "summarize",
        BuiltInTools.text_summarizer,
        "Summarize text to specified length",
        is_async=False
    )
    
    registry.register_tool(
        "classify",
        BuiltInTools.text_classifier,
        "Classify text into categories",
        is_async=False
    )
    
    registry.register_tool(
        "transform",
        BuiltInTools.data_transformer,
        "Transform data based on type",
        is_async=False
    )
    
    registry.register_tool(
        "web_search",
        BuiltInTools.async_web_search,
        "Search the web for information",
        is_async=True
    )
    
    registry.register_tool(
        "process_data",
        BuiltInTools.async_data_processor,
        "Process data with various operations",
        is_async=True
    )
    
    # Register advanced tools (if configuration is valid)
    if Config.validate():
        registry.register_tool(
            "openai_research",
            AdvancedTools.openai_research,
            "Research topics using OpenAI",
            is_async=True
        )
        
        registry.register_tool(
            "send_email",
            AdvancedTools.send_email,
            "Send emails via SMTP",
            is_async=True
        )
        
        registry.register_tool(
            "openai_generate",
            AdvancedTools.openai_generate_content,
            "Generate content using OpenAI",
            is_async=True
        )
    else:
        print("Warning: Advanced tools (OpenAI, email) not available due to missing configuration.")
    
    return registry 