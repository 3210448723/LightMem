from typing import Dict, Optional, List, Any
from lightmem.configs.pre_compressor.llmlingua_2 import LlmLingua2Config


class LlmLingua2Compressor:
    def __init__(self, config: Optional[LlmLingua2Config] = None):
        self.config = config

        try:
            import importlib
            importlib.import_module('llmlingua')
        except ImportError:
            raise ImportError(
                "Required package 'llmlingua' not found. "
                "Please install it with: pip install llmlingua\n"
                "Or for the latest version: pip install git+https://github.com/microsoft/LLMLingua.git"
            )

        try:
            from llmlingua import PromptCompressor
            if config.llmlingua_config['use_llmlingua2'] is True:
                self._compressor = PromptCompressor(
                    model_name=config.llmlingua_config['model_name'],
                    device_map=config.llmlingua_config['device_map'],
                    use_llmlingua2=config.llmlingua_config['use_llmlingua2'],
                    llmlingua2_config=config.llmlingua2_config
                )
            else:
                self._compressor = PromptCompressor(
                    model_name=config.llmlingua_config['model_name'],
                    device_map=config.llmlingua_config['device_map']
                )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LlmLingua2Compressor: {str(e)}")

    def compress(
        self,
        messages: List[Dict[str, str]],
        tokenizer: Optional[Any]
    ):
        # TODO : Consider adding an extra field in the message, compressed_content, and put the compressed content in this field while keeping content unchanged.
        for mes in messages:
            compress_config = {
                'context': [mes['content']],
                **self.config.compress_config  # compress_config['rate']=0.8 compress_config['target_token']=-1
            }
            # Normalize compressor output to a string
            result = self._compressor.compress_prompt(**compress_config)
            if isinstance(result, dict) and 'compressed_prompt' in result:
                comp_content = result['compressed_prompt']
            else:
                # Fallback: if the compressor returns a string directly
                comp_content = result if isinstance(result, str) else str(result)

            # Iteratively compress until within token budget (when tokenizer is provided)
            while tokenizer is not None and isinstance(comp_content, str) and len(tokenizer.encode(comp_content)) >= 512:
                new_compress_config = {
                    # LLMLingua expects a list for context
                    'context': [comp_content],
                    **self.config.compress_config
                }
                result = self._compressor.compress_prompt(**new_compress_config)
                if isinstance(result, dict) and 'compressed_prompt' in result:
                    comp_content = result['compressed_prompt']
                else:
                    comp_content = result if isinstance(result, str) else str(result)
            if comp_content != "":
                mes['content'] = comp_content
            mes['content'] = mes['content'].strip()
            
        return messages

    @property
    def inner_compressor(self):
        """
        Access the underlying PromptCompressor instance directly.
        """
        return self._compressor
