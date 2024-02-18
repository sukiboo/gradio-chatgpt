import gradio as gr
from openai import OpenAI


class GPT:
    """Instantiate GPT model for a multi-step conversation."""

    def __init__(self, **params):
        """Setup model parameters and system prompt."""
        self.client = OpenAI()
        self.params = {'model': 'gpt-3.5-turbo', **params}
        self.messages = []

    def chat(self, user_prompt, chat_history, system_prompt, *params):
        """Generate a response with a given parameters."""

        # update GPT parameters
        temperature, top_p, frequency_penalty, presence_penalty = params
        self.params['temperature'] = temperature
        self.params['top_p'] = top_p
        self.params['frequency_penalty'] = frequency_penalty
        self.params['presence_penalty'] = presence_penalty

        # update the message buffer
        self.messages = [{'role': 'system', 'content': f'{system_prompt}'}]
        for prompt, response in chat_history:
            self.messages.append({'role': 'user', 'content': f'{prompt}'})
            self.messages.append({'role': 'assistant', 'content': f'{response}'})
        self.messages.append({'role': 'user', 'content': f'{user_prompt}'})

        # generate a response
        completion = self.client.chat.completions.create(messages=self.messages, **self.params)
        response = completion.choices[0].message.content
        self.messages.append({'role': 'assistant', 'content': response})
        chat_history.append((user_prompt, response))

        return '', chat_history


def run_chatbot(gpt):
    """Configure and launch the chatbot interface."""
    with gr.Blocks(title='ChatGPT') as chatbot:

        # chatbot interface
        chat_history = gr.Chatbot(height=500, layout='bubble', label='ChatGPT')
        with gr.Row():
            user_prompt = gr.Textbox(placeholder='Message ChatGPT...', container=False, min_width=500, scale=9)
            submit_button = gr.Button('Submit')

        # parameters accordion
        with gr.Accordion(label='GPT Parameters', open=False):
            info = gr.Markdown('For parameter documentation see [OpenAI Chat API Reference]' \
                               + '(https://platform.openai.com/docs/api-reference/chat)')
            system_prompt = gr.Textbox('You are a helpful assistant.', label='system prompt')
            with gr.Row():
                temperature = gr.Slider(0., 2., value=1., step=.1, min_width=200, label='temperature')
                top_p = gr.Slider(0., 1., value=1., step=.01, min_width=200, label='top_p')
                frequency_penalty = gr.Slider(-2., 2., value=0, step=.1, min_width=200, label='frequency_penalty')
                presence_penalty = gr.Slider(-2., 2., value=0, step=.1, min_width=200, label='presence_penalty')

        # submit user prompt
        inputs = [user_prompt, chat_history, system_prompt,
                  temperature, top_p, frequency_penalty, presence_penalty]
        outputs = [user_prompt, chat_history]
        submit_button.click(gpt.chat, inputs=inputs, outputs=outputs)
        user_prompt.submit(gpt.chat, inputs=inputs, outputs=outputs)

    # instantiate the chatbot
    gr.close_all()
    chatbot.queue(default_concurrency_limit=None)
    chatbot.launch()


if __name__ == '__main__':
    gpt = GPT()
    run_chatbot(gpt)
