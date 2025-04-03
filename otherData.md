# <!-- Sellmate Company Agent old SI -->
        # final_system_instruction = f"""You are the AI Support Agent for {company.name}.
        #     Role and Context:
        #     - You represent {company.name} and must respond as an official company representative
        #     - Use the following company information to inform your responses:
        #     {company.system_instruction}

        #     Response Guidelines:
        #     1. Provide accurate, helpful answers based on the company information provided
        #     2. Keep responses concise and easy to understand
        #     3. Stay within the scope of the company information provided
        #     4. If unsure about any specific details, acknowledge the limitation and suggest contacting the company directly
        #     5. Maintain a professional yet friendly tone
        #     6. Focus on addressing the customer's needs while representing the company's values and sell the company service to visitors that we our company can also do this services but in too much short.
        #     7. If visitor is also asking some question looking like personal but also related to company services then answer on behalf of company in short.
        #     8. If someone is asking the question related services like what is inside this services then answer from your knowledge based and answer on behalf of comapny.

        #     If you cannot confidently answer a question based on the provided information, respond with basic answer and say to connect with company contact info.
        #     """
##

      function formatBotResponse(text) {
        // Handle bold text
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Handle bullet points/lists
        const lines = text.split('\n');
        let formattedText = '';
        let inList = false;
        
        lines.forEach(line => {
            if (line.trim().startsWith('*')) {
                if (!inList) {
                    formattedText += '<ul class="bot-list">';
                    inList = true;
                }
                const listItem = line.trim().substring(1).trim();
                formattedText += `<li>${listItem}</li>`;
            } else {
                if (inList) {
                    formattedText += '</ul>';
                    inList = false;
                }
                formattedText += `<p>${line}</p>`;
            }
        });
        
        if (inList) {
            formattedText += '</ul>';
        }
        
        return formattedText;



system_instruction = """
    You are an AI assistant helping to gather information about companies to set up their chatbots.
    Follow these steps:
    1. Ask for the company name
    2. Ask detailed questions about:
    - Company's main products/services
    - Target audience
    - Key features/benefits
    - Company values and mission
    3. Organize the gathered information into a structured format
    4. Confirm the information before saving

    Keep the conversation professional but friendly.
    Only proceed to save when you have gathered comprehensive information.
    """


@csrf_exempt
def ai_chat(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_message = data.get("message", "")
            chat_history = data.get("history", [])
            current_step = data.get("currentStep", "name")

            # Create a new config with the company onboarding system instruction
            onboarding_config = types.GenerateContentConfig(
                temperature=0.7,  # Slightly higher temperature for more creative responses
                top_p=0.8,
                max_output_tokens=1024,
                response_modalities=["TEXT"],
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT", threshold="OFF"
                    ),
                ],
                system_instruction=system_instruction,  # Use the company onboarding instruction
            )

            # Prepare the content for the model
            contents = [
                types.Content(
                    role="user", parts=[types.Part.from_text(text=user_message)]
                )
            ]

            # Add chat history to contents
            for msg in chat_history:
                contents.append(
                    types.Content(
                        role=msg["role"],
                        parts=[types.Part.from_text(text=msg["content"])],
                    )
                )

            # Generate the response using the onboarding config
            response_text = ""
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=onboarding_config,  # Use the onboarding config
            ):
                response_text += chunk.text if chunk.text else ""

            # Process response based on current step
            save_ready = False
            if current_step == "name":
                if len(user_message.strip()) < 2:
                    response_text = (
                        "Please provide a valid company name (at least 2 characters)."
                    )
                else:
                    save_ready = True
                    response_text = "Great! Now, please tell me about your company's main products/services, target audience, key features/benefits, and company values/mission."
            elif current_step == "information":
                if len(user_message.strip()) < 50:
                    response_text = "Please provide more detailed information about the company. Include products/services, target audience, and company values."
                else:
                    save_ready = True

            return JsonResponse({"message": response_text, "save": save_ready})

        except Exception as e:
            import traceback

            print(traceback.format_exc())
            return JsonResponse(
                {"message": f"An error occurred: {str(e)}", "save": False}, status=200
            )

    return JsonResponse({"error": "Invalid request method"}, status=405)
