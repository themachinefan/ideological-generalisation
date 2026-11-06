"""
Context-augmented model wrapper for InspectAI.

This wrapper prepends system messages and/or conversation history to all prompts
before they are sent to the underlying model. This allows for easy context
augmentation across different experiments.

Usage:
    model_name = "context_augmented/path/to/context.jsonl:gpt-4o-mini"
    model_name = "context_augmented/path/to/context.yaml:gpt-4o-mini"
    model = get_model(model_name)
    
Supports both JSONL and YAML formats (auto-detected by file extension).

JSONL format: Each line has {"messages": [...]} with user/assistant pairs.
All messages from all lines are combined into one conversation history.

YAML format: See load_context_config_yaml docstring for supported formats.

Note: Uses : as separator between file path and base model name. The separator
is parsed from the right to handle file paths that may contain colons.
"""

from typing import Any, List
from inspect_ai.model._chat_message import ChatMessage, ChatMessageSystem, ChatMessageUser, ChatMessageAssistant
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model_output import ModelOutput
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_ai.model import get_model as inspect_get_model, modelapi
import yaml
import json
import warnings
import os
from pathlib import Path


def _resolve_model_specific_content(content: Any, model_name: str | None) -> str:
    """
    Resolve model-specific content from various formats.
    
    Supports:
    1. Simple string: "content" -> used for all models
    2. Model-specific dict: {gpt-4o: "content1", gpt-4o-mini: "content2", default: "content3"}
    
    Args:
        content: Content value (string or dict)
        model_name: Model name to select content for
    
    Returns:
        Resolved content string
    
    Raises:
        ValueError: If model-specific content not found and no default
    """
    # Simple string case - use for all models
    if isinstance(content, str):
        return content
    
    # Dictionary case - model-specific content
    if isinstance(content, dict):
        # Extract just the model name without provider prefix
        model_key = model_name.split('/')[-1] if model_name and '/' in model_name else model_name
        
        # Try exact match first
        if model_key and model_key in content:
            return content[model_key]
        
        # Try partial matching (e.g., "gpt-4o" matches "gpt-4o-2024-05-13")
        if model_key:
            for key in content.keys():
                if key != 'default' and (model_key.startswith(key) or key in model_key):
                    return content[key]
        
        # Fall back to default
        if 'default' in content:
            return content['default']
        
        # No match found
        available_models = [k for k in content.keys() if k != 'default']
        has_default = 'default' in content
        error_msg = f"No content found for model '{model_name}'. "
        if available_models:
            error_msg += f"Available models: {available_models}. "
        if not has_default:
            error_msg += "No default content available."
        raise ValueError(error_msg)
    
    # Unknown format
    raise ValueError(f"Invalid content format: {type(content)}")


def _resolve_file_path(file_path: str) -> str:
    """
    Resolve a file path, trying relative to cwd first, then relative to package root.
    
    Args:
        file_path: Path to file (can be relative or absolute)
    
    Returns:
        Resolved full path as string
    
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.isabs(file_path):
        # Try relative to current working directory
        if os.path.exists(file_path):
            return file_path
        else:
            # Try relative to this file's directory (package root)
            base_dir = Path(__file__).parent.parent.parent
            full_path = base_dir / file_path
            if not full_path.exists():
                raise FileNotFoundError(f"Context file not found: {file_path}")
            return str(full_path)
    else:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Context file not found: {file_path}")
        return file_path


def load_context_from_jsonl(jsonl_path: str, model_name: str | None = None) -> dict:
    """
    Load context configuration from JSONL file.
    
    Each line in the JSONL should have format:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    
    Optionally, lines can include a system message:
        {"system": "System prompt", "messages": [...]}
    
    All messages from all lines are combined into one conversation history.
    System messages: only one is used. If multiple different system messages are found,
    a warning is printed and the first one is used.
    
    Args:
        jsonl_path: Path to JSONL file (can be relative or absolute)
        model_name: Not used for JSONL, kept for API compatibility
    
    Returns:
        Dictionary with 'system_message' (optional) and 'messages' (list)
    
    Raises:
        FileNotFoundError: If JSONL file doesn't exist
    """
    full_path = _resolve_file_path(jsonl_path)
    
    all_messages = []
    system_message = None
    
    with open(full_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {jsonl_path}: {e}")
            
            # Handle system message
            if 'system' in entry:
                new_system = entry['system']
                if system_message is None:
                    system_message = new_system
                elif system_message != new_system:
                    warnings.warn(
                        f"Multiple different system messages found in {jsonl_path}. "
                        f"Using the first one. Conflicting message on line {line_num}: {new_system[:50]}..."
                    )
            
            # Handle messages
            if 'messages' in entry:
                for msg in entry['messages']:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    
                    # Handle system messages within the messages array
                    if role == 'system':
                        if system_message is None:
                            system_message = content
                        elif system_message != content:
                            warnings.warn(
                                f"Multiple different system messages found in {jsonl_path}. "
                                f"Using the first one. Conflicting message: {content[:50]}..."
                            )
                    else:
                        all_messages.append({
                            'role': role,
                            'content': content
                        })
    
    result = {'messages': all_messages}
    if system_message is not None:
        result['system_message'] = system_message
    
    return result


def load_context_config_yaml(yaml_path: str, model_name: str | None = None) -> dict:
    """
    Load context configuration from YAML file.
    
    Supports multiple formats:
    
    Format 1 - Simple shared config (works for all models):
        system_message: "Optional system message"
        messages:
          - role: user
            content: "User message"
          - role: assistant
            content: "Assistant response"
    
    Format 2 - Model-specific assistant responses (user message shared):
        messages:
          - role: user
            content: "User message"
          - role: assistant
            content:
              gpt-4o: "Response specific to GPT-4o"
              gpt-4o-mini: "Response specific to GPT-4o-mini"
              default: "Default response for other models"
    
    Format 3 - Fully model-specific configs (everything separate):
        models:
          gpt-4o:
            messages:
              - role: user
                content: "User message for GPT-4o"
              - role: assistant
                content: "Assistant response for GPT-4o"
          gpt-4o-mini:
            messages:
              - role: user
                content: "User message for GPT-4o-mini"
              - role: assistant
                content: "Assistant response for GPT-4o-mini"
        default:  # optional fallback
          messages:
            - role: user
              content: "Default message"
    
    Args:
        yaml_path: Path to YAML file (can be relative or absolute)
        model_name: Base model name to select appropriate config (e.g., "gpt-4o", "gpt-4o-mini")
                    If None, uses default or raises error for model-specific content
    
    Returns:
        Dictionary with 'system_message' (optional) and 'messages' (optional list)
    
    Raises:
        FileNotFoundError: If YAML file doesn't exist
        ValueError: If model-specific config is not found and no default is available
    """
    full_path = _resolve_file_path(yaml_path)
    
    with open(full_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if this is Format 3 (fully model-specific with 'models' key)
    if 'models' in config or ('default' in config and 'messages' not in config):
        # Format 3: Fully model-specific configs
        selected_config = None
        
        # Try to find model-specific config
        if model_name and 'models' in config:
            # Extract just the model name without provider prefix
            model_key = model_name.split('/')[-1] if '/' in model_name else model_name
            
            # Try exact match first
            if model_key in config['models']:
                selected_config = config['models'][model_key]
            else:
                # Try matching partial names (e.g., "gpt-4o" matches "gpt-4o-2024-05-13")
                for key in config['models'].keys():
                    if model_key.startswith(key) or key in model_key:
                        selected_config = config['models'][key]
                        break
        
        # Fall back to default if no model-specific config found
        if selected_config is None and 'default' in config:
            selected_config = config['default']
        
        # If still no config found, raise error
        if selected_config is None:
            available_models = list(config.get('models', {}).keys())
            has_default = 'default' in config
            error_msg = f"No configuration found for model '{model_name}' in {yaml_path}. "
            if available_models:
                error_msg += f"Available models: {available_models}. "
            if not has_default:
                error_msg += "No default configuration available."
            raise ValueError(error_msg)
        
        config = selected_config
    
    # Process the config (works for Format 1, Format 2, and processed Format 3)
    result = {}
    
    # Handle system_message
    if 'system_message' in config:
        system_msg = config['system_message']
        if isinstance(system_msg, dict):
            # Model-specific system message
            system_msg = _resolve_model_specific_content(system_msg, model_name)
        result['system_message'] = system_msg
    
    # Handle messages
    if 'messages' in config:
        processed_messages = []
        for msg in config['messages']:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            # Resolve model-specific content if it's a dict
            if isinstance(content, dict):
                content = _resolve_model_specific_content(content, model_name)
            
            processed_messages.append({
                'role': role,
                'content': content
            })
        
        result['messages'] = processed_messages
    
    return result


def load_context_config(file_path: str, model_name: str | None = None) -> dict:
    """
    Load context configuration from file (auto-detects YAML vs JSONL by extension).
    
    Args:
        file_path: Path to config file (.yaml, .yml, or .jsonl)
        model_name: Base model name (used for YAML model-specific configs)
    
    Returns:
        Dictionary with 'system_message' (optional) and 'messages' (list)
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file extension is not recognized
    """
    ext = Path(file_path).suffix.lower()
    
    if ext in ('.yaml', '.yml'):
        return load_context_config_yaml(file_path, model_name)
    elif ext == '.jsonl':
        return load_context_from_jsonl(file_path, model_name)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}' for context config. "
            f"Use .yaml, .yml, or .jsonl"
        )


def _normalize_base_model_name(model_name: str) -> str:
    """
    Normalize base model name by adding provider prefix if needed.
    
    This is a simplified version that doesn't require importing from eval.py.
    """
    if "/" in model_name:
        return model_name
    
    # Add provider prefix based on common patterns
    if model_name.startswith("gpt-") or model_name.startswith("ft:gpt"):
        return f"openai/{model_name}"
    elif model_name.startswith("claude-"):
        return f"anthropic/{model_name}"
    elif "llama" in model_name.lower():
        return f"together/{model_name}"
    else:
        # Default to openai if unclear
        return f"openai/{model_name}"


def parse_context_augmented_model_name(model_name: str) -> tuple[str, str]:
    """
    Parse a context-augmented model name.
    
    Format: "context_augmented/file_path:base_model_name"
    (Uses : as separator, parsed from the right to handle paths with colons)
    
    Args:
        model_name: Model name in format "context_augmented/path/to/config.jsonl:gpt-4o-mini"
                    or "context_augmented/path/to/config.yaml:gpt-4o-mini"
    
    Returns:
        Tuple of (file_path, base_model_name)
    """
    if not model_name.startswith("context_augmented"):
        raise ValueError(f"Model name must start with 'context_augmented': {model_name}")
    
    # Remove the provider prefix (context_augmented/ or context_augmented:)
    if model_name.startswith("context_augmented/"):
        model_id = model_name[len("context_augmented/"):]
    elif model_name.startswith("context_augmented:"):
        model_id = model_name[len("context_augmented:"):]
    else:
        raise ValueError(f"Invalid format. Expected 'context_augmented/yaml_path:base_model', got: {model_name}")
    
    # Split from the right to handle file paths that may contain colons
    if ":" not in model_id:
        raise ValueError(f"Invalid format. Expected 'context_augmented/yaml_path:base_model', got: {model_name}")
    
    parts = model_id.rsplit(":", 1)  # Split from right, max 1 split
    if len(parts) != 2:
        raise ValueError(f"Invalid format. Expected 'context_augmented/yaml_path:base_model', got: {model_name}")
    
    yaml_path, base_model = parts
    return yaml_path, base_model


class ContextAugmentedModelAPI:
    """
    Model API class that wraps any InspectAI model with context augmentation.
    
    This class is instantiated by InspectAI when using "context_augmented/file:base_model"
    format. It parses the model name, loads the context config, and wraps the base model.
    
    Supports both JSONL and YAML config files (auto-detected by extension).
    """
    
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: Any = None,
        **model_args: Any,
    ):
        """
        Initialize context-augmented model API.
        
        Args:
            model_name: Model name in format "context_augmented/yaml_path:base_model"
                       or just "yaml_path:base_model" (model_id part)
            base_url: Not used, kept for API compatibility
            api_key: Not used, kept for API compatibility
            config: Not used, kept for API compatibility
            **model_args: Additional arguments to pass to base model
        """
        # InspectAI may pass the full model name or just the model_id part
        # Handle both cases
        if model_name.startswith("context_augmented"):
            yaml_path, base_model_name = parse_context_augmented_model_name(model_name)
        else:
            # Just the model_id part (after the /), parse it directly
            # Split from the right to handle file paths that may contain colons
            if ":" not in model_name:
                raise ValueError(f"Invalid format. Expected 'yaml_path:base_model', got: {model_name}")
            parts = model_name.rsplit(":", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid format. Expected 'yaml_path:base_model', got: {model_name}")
            yaml_path, base_model_name = parts
        
        # Normalize the base model name (add provider prefix if needed)
        normalized_base_model = _normalize_base_model_name(base_model_name)
        
        # Load context configuration with model name for model-specific configs
        context_config = load_context_config(yaml_path, base_model_name)
        
        # Get the base model
        base_model = inspect_get_model(normalized_base_model, **model_args)
        
        # Store the base model and its API directly to avoid repeated attribute access
        self.base_model = base_model
        self._base_api = base_model.api if hasattr(base_model, 'api') else base_model
        
        # Store context configuration
        self.system_message = context_config.get('system_message')
        self.context_messages = context_config.get('messages', [])
        
        # Store model_name for compatibility with InspectAI
        self.model_name = model_name
        
        # Store normalized base model name for canonical_name()
        self._normalized_base_model = normalized_base_model
        
        # Store API attributes (InspectAI may access these)
        self.base_url = base_url or (getattr(self._base_api, 'base_url', None) if hasattr(self._base_api, 'base_url') else None)
        self.api_key = api_key or (getattr(self._base_api, 'api_key', None) if hasattr(self._base_api, 'api_key') else None)
        
        # Expose self as api attribute (InspectAI expects model.api.canonical_name())
        self.api = self
    
    def canonical_name(self) -> str:
        """Return the canonical name for this model API."""
        # Return the base model name for canonical name
        return self._normalized_base_model
    
    def _augment_messages(self, input_messages: List[ChatMessage]) -> List[ChatMessage]:
        """
        Augment input messages with context.
        
        Args:
            input_messages: Original message list from the task
        
        Returns:
            Augmented message list with context prepended
        """
        augmented = []
        
        # Add system message if provided
        if self.system_message:
            # Check if there's already a system message in the input
            has_system = any(isinstance(msg, ChatMessageSystem) for msg in input_messages)
            if has_system:
                # Replace the first system message with ours
                for i, msg in enumerate(input_messages):
                    if isinstance(msg, ChatMessageSystem):
                        augmented.append(ChatMessageSystem(content=self.system_message))
                        # Skip the original system message
                        augmented.extend(input_messages[i+1:])
                        break
            else:
                # Prepend our system message
                augmented.append(ChatMessageSystem(content=self.system_message))
                augmented.extend(input_messages)
        else:
            augmented = list(input_messages)
        
        # Convert context messages to ChatMessage objects and prepend
        context_chat_messages = []
        for msg_dict in self.context_messages:
            role = msg_dict.get('role', 'user')
            content = msg_dict.get('content', '')
            
            if role == 'system':
                context_chat_messages.append(ChatMessageSystem(content=content))
            elif role == 'user':
                context_chat_messages.append(ChatMessageUser(content=content))
            elif role == 'assistant':
                context_chat_messages.append(ChatMessageAssistant(content=content))
            else:
                raise ValueError(f"Unknown message role: {role}")
        
        # Prepend context messages (but after system message)
        if context_chat_messages:
            # Find where to insert (after system message if any)
            system_count = sum(1 for msg in augmented if isinstance(msg, ChatMessageSystem))
            if system_count > 0:
                # Insert after the last system message
                insert_idx = 0
                for i, msg in enumerate(augmented):
                    if isinstance(msg, ChatMessageSystem):
                        insert_idx = i + 1
                augmented = augmented[:insert_idx] + context_chat_messages + augmented[insert_idx:]
            else:
                # No system message, prepend at the beginning
                augmented = context_chat_messages + augmented
        
        return augmented
    
    async def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """
        Generate a response with context augmentation.
        
        This method prepends the configured context messages to the input
        before calling the underlying model's generate() method.
        """
        augmented_input = self._augment_messages(input)
        #with open("test.log", "a") as f:
        #    f.write(f"Messages sent to API: {augmented_input}\n")
        
        # Call generate on the base model's API directly to avoid any wrapper overhead
        # Ensure tools is a list (not None) so the OpenAI provider's len() check works
        return await self._base_api.generate(
            input=augmented_input,
            tools=tools if tools is not None else [],
            tool_choice=tool_choice,
            config=config,
        )
    
    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to base model or its API.
        
        This is a fallback for attributes not explicitly set. We try to avoid
        using this during parallel execution by explicitly setting common attributes.
        """
        # Avoid recursive calls - if we're looking for something we already have, return it
        if name in ('base_model', 'system_message', 'context_messages', 'model_name', 
                   '_normalized_base_model', 'base_url', 'api_key', 'api'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # First try the base model directly
        if hasattr(self.base_model, name):
            attr = getattr(self.base_model, name)
            # Cache it to avoid repeated lookups during parallel execution
            try:
                setattr(self, name, attr)
            except (TypeError, AttributeError):
                # Can't cache some attributes (like methods), that's fine
                pass
            return attr
        
        # If not found, try the base model's API
        if hasattr(self.base_model, 'api'):
            base_api = self.base_model.api
            if hasattr(base_api, name):
                attr = getattr(base_api, name)
                # Cache it
                try:
                    setattr(self, name, attr)
                except (TypeError, AttributeError):
                    pass
                return attr
        
        # Re-raise the original AttributeError if not found
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


@modelapi(name="context_augmented")
def context_augmented_model_factory():
    """Factory function for context-augmented models."""
    return ContextAugmentedModelAPI

