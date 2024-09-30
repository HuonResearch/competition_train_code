import openai

class GPT:
    def __init__(self, api_key, version):
        self.api_key = api_key
        self.version = version

    def submit(self, prompt, temperature=1.0, num_responses=3, max_tokens=1024) -> list:
        """
        Submit a prompt to GPT.
        :param prompt: Prompt for GPT.
        :return: List of length `num_responses` containing the LLM's output in response to the prompt.
        """
        openai.api_key = self.api_key # TODO: Where else could this be assigned within the class?

        client = openai.OpenAI()
        llm_response = client.chat.completions.create(model=self.version,
                                                               messages=[{"role": "user", "content": prompt}],
                                                               temperature=temperature,
                                                               max_tokens=max_tokens,
                                                               n=num_responses)

        # Extract LLM responses
        instruction_responses = [choice.message.content for choice in llm_response.choices]

        return instruction_responses
