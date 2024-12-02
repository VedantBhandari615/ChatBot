import openai

# Set your OpenAI API key
openai.api_key = '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'

def chatbot_response(prompt):
    try:
        # Call the OpenAI API to get a response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        # Extract and return the response text
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"

def chat():
    print("Start chatting with the bot! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = chatbot_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()
