from ollama import Client
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Ollama for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.client = Client(host=host)
        self.model = model

        # Pre-build base options
        self.base_options = {
            "temperature": 0,
            "num_predict": 800
        }

    def _convert_tools_to_ollama(self, tools: List) -> List[Dict]:
        """Convert Anthropic-style tool definitions to Ollama format"""
        ollama_tools = []
        for tool in tools:
            ollama_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {})
                }
            }
            ollama_tools.append(ollama_tool)
        return ollama_tools

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Build messages list
        messages = [{"role": "user", "content": query}]

        # Prepare API call parameters
        kwargs = {
            "model": self.model,
            "messages": messages,
            "system": system_content,
            "options": self.base_options
        }

        # Add tools if available
        if tools:
            kwargs["tools"] = self._convert_tools_to_ollama(tools)

        # Get response from Ollama
        response = self.client.chat(**kwargs)

        # Handle tool execution if needed
        if response.message.tool_calls and tool_manager:
            return self._handle_tool_execution(response, messages, system_content, tool_manager)

        # Return direct response
        return response.message.content or ""

    def _handle_tool_execution(self, initial_response, messages: List, system_content: str, tool_manager) -> str:
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            messages: Current message list
            system_content: System prompt content
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Add AI's tool use response to messages
        messages.append({
            "role": "assistant",
            "content": initial_response.message.content or "",
            "tool_calls": initial_response.message.tool_calls
        })

        # Execute all tool calls and collect results
        for tool_call in initial_response.message.tool_calls:
            tool_result = tool_manager.execute_tool(
                tool_call.function.name,
                **tool_call.function.arguments
            )

            messages.append({
                "role": "tool",
                "content": tool_result
            })

        # Get final response
        final_response = self.client.chat(
            model=self.model,
            messages=messages,
            system=system_content,
            options=self.base_options
        )

        return final_response.message.content or ""