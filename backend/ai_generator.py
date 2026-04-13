from ollama import Client, RequestError, ResponseError
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Ollama for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **Sequential tool calling supported** — you may make up to 2 tool calls across separate reasoning rounds to chain operations (e.g. first retrieve a lesson title, then search for courses on that topic)
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

        # Build system content efficiently
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Build messages list with a system prompt for Ollama
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]

        # Prepare API call parameters
        kwargs = {
            "model": self.model,
            "messages": messages,
            "options": self.base_options
        }

        # Add tools if available
        if tools:
            kwargs["tools"] = self._convert_tools_to_ollama(tools)

        # Get response from Ollama
        try:
            response = self.client.chat(**kwargs)
        except (RequestError, ResponseError) as e:
            raise RuntimeError(f"Ollama client error: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Ollama client error: {e}") from e

        # Handle sequential tool execution if needed (up to 2 rounds)
        if getattr(response.message, "tool_calls", None) and tool_manager:
            return self._handle_sequential_tool_execution(
                initial_response=response,
                messages=messages,
                system_content=system_content,
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
        system_content: str,
        tool_manager,
        tools: Optional[List],
        max_rounds: int = 2
    ) -> str:
        """
        Handle sequential tool execution with up to max_rounds.

        Each round:
        1. Execute tool calls and add results to messages
        2. If not last round and response has tool_calls, make another API call
        3. On last round or when no more tool_calls, synthesize final response

        Args:
            initial_response: The initial response containing tool use requests
            messages: Current message list (will be mutated)
            system_content: System prompt content
            tool_manager: Manager to execute tools
            tools: Available tools for subsequent API calls
            max_rounds: Maximum number of tool call rounds (default 2)

        Returns:
            Final response text after all tool execution
        """
        current_response = initial_response

        for round_num in range(max_rounds):
            # Add AI's tool use response to messages (append message object directly)
            messages.append(current_response.message)

            # Execute all tool calls and collect results
            for tool_call in current_response.message.tool_calls:
                tool_result = tool_manager.execute_tool(
                    tool_call.function.name,
                    **tool_call.function.arguments
                )

                # Tool result message requires tool_name field for Ollama
                messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_name": tool_call.function.name
                })

            # Check if we should continue to another round
            has_more_tool_calls = getattr(current_response.message, "tool_calls", None)

            if not has_more_tool_calls or round_num == max_rounds - 1:
                # No more tool calls or max rounds reached - synthesize final response
                api_messages = [{"role": "system", "content": system_content}] + messages
                try:
                    final_response = self.client.chat(
                        model=self.model,
                        messages=api_messages,
                        options=self.base_options
                    )
                except (RequestError, ResponseError) as e:
                    raise RuntimeError(f"Ollama client error: {e}") from e
                except Exception as e:
                    raise RuntimeError(f"Ollama client error: {e}") from e
                return final_response.message.content or ""

            # Not last round and has more tool calls - make API call for next round
            api_messages = [{"role": "system", "content": system_content}] + messages

            # Pass tools for potential next round
            kwargs = {
                "model": self.model,
                "messages": api_messages,
                "options": self.base_options
            }
            if tools:
                kwargs["tools"] = self._convert_tools_to_ollama(tools)

            # Get next response
            try:
                current_response = self.client.chat(**kwargs)
            except (RequestError, ResponseError) as e:
                raise RuntimeError(f"Ollama client error: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Ollama client error: {e}") from e

        # Safety net (should not reach here)
        return current_response.message.content or ""