import gradio as gr
from openai import OpenAI


class GPT:
    """Instantiate GPT model for a multi-step conversation."""

    def __init__(self, **params):
        """Setup model parameters and system prompt."""
        self.client = OpenAI()
        self.params = {'model': 'gpt-3.5-turbo', **params}
        self.messages = []

    def chat(self, user_prompt, history=[], *params):
        """Update model parameters and generate a response."""
        _, system_prompt, temperature, top_p, frequency_penalty, presence_penalty = params

        # update the parameters
        self.params['temperature'] = temperature
        self.params['top_p'] = top_p
        self.params['frequency_penalty'] = frequency_penalty
        self.params['presence_penalty'] = presence_penalty

        # update the message buffer
        self.messages = [{'role': 'system', 'content': f'{system_prompt}'}]
        for prompt, response in history:
            self.messages.append({'role': 'user', 'content': f'{prompt}'})
            self.messages.append({'role': 'assistant', 'content': f'{response}'})
        self.messages.append({'role': 'user', 'content': f'{user_prompt}'})

        # generate a response
        completion = self.client.chat.completions.create(messages=self.messages, **self.params)
        response = completion.choices[0].message.content
        self.messages.append({'role': 'assistant', 'content': response})

        return response


def run_chatbot(gpt):
    """Configure and launch the chatbot interface."""
    gr.close_all()
    with gr.Blocks() as chatbot:

        # parameters accordion
        with gr.Accordion(label='GPT Parameters', open=False) as parameters:
            info = gr.Markdown('For parameter documentation see [OpenAI Chat API Reference]' \
                               + '(https://platform.openai.com/docs/api-reference/chat)')
            system_prompt = gr.Textbox('You are a helpful assistant.', label='system prompt')
            with gr.Row():
                temperature = gr.Slider(0., 2., value=1., step=.1, label='temperature')
                top_p = gr.Slider(0., 1., value=1., step=.01, label='top_p')
            # with gr.Row():
                frequency_penalty = gr.Slider(-2., 2., value=0, step=.1, label='frequency_penalty')
                presence_penalty = gr.Slider(-2., 2., value=0, step=.1, label='presence_penalty')

        # chatbot interface
        gr.ChatInterface(
            fn=gpt.chat,
            chatbot=gr.Chatbot(height=600, layout='bubble', label='ChatGPT', render=False),
            textbox=gr.Textbox(placeholder='Message ChatGPT...', scale=9, render=False),
            additional_inputs_accordion=parameters,
            additional_inputs=[info, system_prompt, temperature, top_p, frequency_penalty, presence_penalty],
            retry_btn=None,
            undo_btn=None,
            clear_btn=None,
            concurrency_limit=None,
            theme=None,
        )

    chatbot.launch()


if __name__ == '__main__':
    gpt = GPT()
    run_chatbot(gpt)
