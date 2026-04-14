from ollama import Client, RequestError, ResponseError
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Ollama for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **Sequential tool calling supported** — you have a maximum of 2 sequential reasoning rounds. Each round may include one or more tool calls. Plan your strategy within this limit.
- Within a single round, all tool calls are independent — do not place dependent calls (where one needs the result of another) in the same round
- When making a second-round tool call, use information from the first round's results to refine your search (e.g., use a discovered topic name as the query)
- Only use a second round when the first round's results are insufficient to answer the question
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Chained search example:
  Round 1: search_course_content(query="topic X", course_name="Course A") → learn that lesson 5 covers X
  Round 2: search_course_content(query="advanced X", course_name="Course B") → find related content

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
  - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
  - Do not mention "based on the search results"
  - When chaining tool calls, you may briefly state what you're looking for (e.g., "Let me find the lesson title first")

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

    def _call_api(self, messages: List[Dict], tools: Optional[List] = None):
        """Make an API call to Ollama with error handling.

        Args:
            messages: Complete message list including system message
            tools: Optional raw tool definitions (will be converted to Ollama format)

        Returns:
            Raw Ollama response object
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "options": self.base_options
        }
        if tools:
            kwargs["tools"] = self._convert_tools_to_ollama(tools)

        try:
            return self.client.chat(**kwargs)
        except (RequestError, ResponseError) as e:
            raise RuntimeError(f"Ollama client error: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Ollama client error: {e}") from e

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

        Supports sequential tool calling with up to 2 rounds per query.
        Each tool call round allows the AI to reason about previous results
        before making additional tool calls.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Build messages list — includes system message so it's self-contained
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]

        # Get initial response from Ollama
        response = self._call_api(messages, tools=tools)

        # Handle sequential tool execution if needed (up to 2 rounds)
        if getattr(response.message, "tool_calls", None) and tool_manager:
            return self._handle_sequential_tool_execution(
                initial_response=response,
                messages=messages,
                tool_manager=tool_manager,
                tools=tools,
                max_rounds=2
            )

        # Return direct response
        return response.message.content or ""

    def _handle_sequential_tool_execution(
        self,
        initial_response,
        messages: List,
        tool_manager,
        tools: Optional[List],
        max_rounds: int = 2
    ) -> str:
        """
        Handle sequential tool execution with up to max_rounds.

        Flow per round:
        1. Execute tool calls from current response, append results to messages
        2. Make next API call (with tools if not last round, without if last)
        3. If next response has no tool_calls, return content directly

        Args:
            initial_response: The initial response containing tool use requests
            messages: Current message list (including system message; will be mutated)
            tool_manager: Manager to execute tools
            tools: Available tools for subsequent API calls
            max_rounds: Maximum number of tool call rounds (default 2)

        Returns:
            Final response text after all tool execution
        """
        current_response = initial_response

        for round_num in range(max_rounds):
            tool_calls = getattr(current_response.message, "tool_calls", None)

            if not tool_calls:
                # Model responded directly — no further tool calls needed
                return current_response.message.content or ""

            # Append assistant message once (contains all tool calls)
            messages.append(current_response.message)

            # Execute each tool call and append results
            for tool_call in tool_calls:
                try:
                    tool_result = tool_manager.execute_tool(
                        tool_call.function.name,
                        **(tool_call.function.arguments or {})
                    )
                except Exception as e:
                    tool_result = f"Error executing tool '{tool_call.function.name}': {e}"

                messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_name": tool_call.function.name
                })

            # Last round: synthesize final response (no tools — model must answer)
            if round_num == max_rounds - 1:
                final_response = self._call_api(messages)
                return final_response.message.content or ""

            # Not last round: call API with tools so model can chain
            current_response = self._call_api(messages, tools=tools)

            # If model responded directly (no tool_calls), return content
            if not getattr(current_response.message, "tool_calls", None):
                return current_response.message.content or ""

        return current_response.message.content or ""