from django.http import JsonResponse, Http404
from django.shortcuts import render, redirect, get_object_or_404
from google import genai
from google.genai import types
from .models import ChatHistory, Company, SellMateAgentChatHistory
import uuid
from django.db.models import Count
from django.db.models import Max
from django.views.decorators.csrf import csrf_exempt
import json
import logging

logger = logging.getLogger(__name__)
# from django.middleware.csrf import get_token


# Initialize the Vertex AI client
client = genai.Client(
    vertexai=True,
    project="metricvibes-1718777660306",
    location="us-central1",
)

model = "gemini-2.0-flash-lite-001"  # LLM Model ID for the chatbot


# Render the chatbot UI for a specific company
def home(request, company_name):
    try:
        company = Company.objects.get(name=company_name)
    except Company.DoesNotExist:
        raise Http404("Company not found")

    # Generate a unique session ID for the user if not already present
    if not request.session.get("session_id"):
        request.session["session_id"] = str(uuid.uuid4())

    return render(request, "chatbot/index.html", {"company_name": company.name})


# Handle chat requests for a specific company
@csrf_exempt
def chat(request, company_name):
    try:
        company = Company.objects.get(name=company_name)
    except Company.DoesNotExist:
        return JsonResponse({"error": "Company not found"}, status=404)

    if request.method == "POST":
        session_id = request.session.get("session_id")
        user_input = request.POST.get("message")
        if not user_input:
            return JsonResponse({"error": "No input provided"}, status=400)
        # Get recent chat history for context (last 5 messages)
        recent_chats = ChatHistory.objects.filter(
            session_id=session_id, company=company
        ).order_by("-timestamp")[:5][::-1]  # Reverse to get chronological order

        # Prepare conversation history for the model
        contents = []

        final_system_instruction = f"""You are the AI Support Agent for {company.name}, acting as the company's official representative. Your goal is to assist visitors, provide clear answers, and drive conversions while keeping responses **short, natural, and human-like**.      
        - Use the following company information to inform your responses:{company.system_instruction}

        ### **Response Style Guidelines**  
        1. **Be concise & to the point** – Provide brief, direct answers. If the visitor asks for more details, then elaborate.  
        2. **Sound human-like & engaging** – Respond naturally as a support rep or salesperson would. Avoid robotic or overly formal replies.  
        3. **Use bullet points for clarity** – When explaining multiple reasons or steps, list them in short bullet points.  
        4. **Prioritize sales conversion** – Subtly highlight company services whenever relevant. Example:  
        - ❌ *"We offer website design, marketing, and hosting services."*  
        - ✅ *"We can help with this! Our team handles website design, marketing & hosting—want details?"*  
        5. **Acknowledge issues before suggesting solutions** – If a visitor reports a problem, briefly recognize it first, then provide possible solutions.  

        ### **Response Handling**  
        - **General Inquiries:** Answer briefly based on the knowledge base. If more details are needed, ask, *"Would you like a deeper explanation?"*  
        - **Service-Related Queries:** If asked about a service, summarize it in one line. If the visitor asks for more, provide additional details.  
        - **Personal Queries Related to Services:** If a visitor’s question is semi-personal but connected to services, respond in a short, relevant manner.  
        - **Uncertain or Out-of-Scope Questions:** If unsure, give a general response and suggest contacting the company directly.  

        ### **Examples of Human-Like Responses**  
        - **Visitor:** “My order hasn’t arrived. What could be the reason?”  
        - **Agent:** “Sorry about that! It could be due to:  
            - 📦 Delayed shipment  
            - ⏳ Processing issues  
            - 📍 Incorrect address  
            Want me to check the status for you?”  

        - **Visitor:** “What’s included in your SEO service?”  
        - **Agent:** “We handle keyword research, optimization, and ranking strategies. Want a full breakdown?”  

        - **Visitor:** “Can you do social media ads?”  
        - **Agent:** “Yes! We run targeted ad campaigns. Need more details?”  

        - **Visitor:** “I need help with website design.”  
        - **Agent:** “Got it! We create modern, high-converting websites. What are you looking for?”  

        - **Visitor:** “Tell me about pricing.”  
        - **Agent:** “Pricing depends on the service. Want me to guide you based on your needs?”  

        If you cannot confidently answer a question based on the provided knowledge base, **acknowledge the limitation and suggest contacting the company directly.**  
        """

        # Create a company-specific config
        company_config = types.GenerateContentConfig(
            temperature=0.2,
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
            system_instruction=final_system_instruction,  # Use company-specific instruction
        )

        # Add conversation history
        for chat in recent_chats:
            contents.extend(
                [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=chat.user_message)],
                    ),
                    types.Content(
                        role="assistant",
                        parts=[types.Part.from_text(text=chat.bot_response)],
                    ),
                ]
            )

        # Add current user message
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
        )

        # Create company-specific config
        company_config = types.GenerateContentConfig(
            temperature=0.2,
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
            system_instruction=final_system_instruction,
        )

        # Generate response with conversation context
        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=company_config,
        ):
            response_text += chunk.text if chunk.text else ""

        # Save the chat to the database
        ChatHistory.objects.create(
            session_id=session_id,
            company=company,
            user_message=user_input,
            bot_response=response_text,
        )

        return JsonResponse({"response": response_text})

    elif request.method == "GET":
        # Retrieve chat history for the current session and company
        session_id = request.session.get("session_id")
        chat_history = ChatHistory.objects.filter(
            session_id=session_id, company=company
        ).order_by("timestamp")
        history = [
            {"user_message": chat.user_message, "bot_response": chat.bot_response}
            for chat in chat_history
        ]
        return JsonResponse({"chat_history": history})

    return JsonResponse({"error": "Invalid request method"}, status=405)


# View to handle adding a new company
def add_company(request):
    if request.method == "POST":
        company_name = request.POST.get("company_name")
        company_info = request.POST.get("company_info")
        if company_name and company_info:
            # Save the new company to the database
            Company.objects.create(name=company_name, system_instruction=company_info)
            return redirect("/dashboard")  # Redirect to the dashboard after saving
    return render(request, "chatbot/add-company.html")


def dashboard(request):
    # Get all companies
    companies = Company.objects.all()

    # Calculate total unique sessions
    total_sessions = ChatHistory.objects.values("session_id").distinct().count()

    # Calculate total messages (sum of all chat entries)
    total_messages = ChatHistory.objects.count()

    # Prepare context data
    context = {
        "companies": companies,
        "total_sessions": total_sessions,
        "total_messages": total_messages,
    }

    return render(request, "chatbot/dashboard.html", context)


# View to list all companies with chatbot statistics
def companies(request):
    companies = Company.objects.annotate(
        total_sessions=Count(
            "chathistory__session_id", distinct=True
        ),  # Total unique sessions
        total_messages=Count("chathistory"),  # Total messages
    )
    return render(request, "chatbot/companies.html", {"companies": companies})


# View to list all chat sessions for a specific company
def company_chats(request, company_id):
    try:
        company = Company.objects.get(id=company_id)
    except Company.DoesNotExist:
        raise Http404("Company not found")

    chats = ChatHistory.objects.filter(company=company).order_by("timestamp")
    return render(request, "chatbot/chats.html", {"company": company, "chats": chats})


# View to list all chat sessions for a specific company
def company_sessions(request, company_id):
    company = get_object_or_404(Company, id=company_id)
    sessions = (
        ChatHistory.objects.filter(company=company)
        .values("session_id")
        .annotate(
            last_interaction=Max(
                "timestamp"
            ),  # Get the latest timestamp for each session
            chat_status=Max(
                "status"
            ),  # Assuming you have a 'status' field in ChatHistory
        )
    )
    return render(
        request, "chatbot/chats.html", {"company": company, "sessions": sessions}
    )


# View to display full chat history for a specific session
def session_chat(request, session_id):
    chats = ChatHistory.objects.filter(session_id=session_id).order_by("timestamp")
    if not chats.exists():
        raise Http404("Session not found")
    company = chats.first().company
    return render(
        request,
        "chatbot/user-chat.html",
        {"company": company, "chats": chats, "session_id": session_id},
    )

@csrf_exempt
def save_company(request):
    if request.method == "POST":
        try:
            # Handle both JSON and Form submissions
            if request.content_type == "application/json":
                data = json.loads(request.body)
            else:
                data = request.POST

            name = data.get("name")
            information = data.get("information")

            # Validation: Ensure required fields are provided
            if not name or not information:
                return JsonResponse(
                    {
                        "status": "error",
                        "message": "Company name and information are required",
                    },
                    status=400,
                )

            # Check if company already exists
            if Company.objects.filter(name=name).exists():
                return JsonResponse(
                    {
                        "status": "error",
                        "message": f"A company named '{name}' already exists.",
                    },
                    status=400,
                )

            # Create new company
            company = Company.objects.create(name=name, system_instruction=information)

            return JsonResponse(
                {
                    "status": "success",
                    "message": f"Company '{company.name}' has been successfully created.",
                    "company_id": company.id,
                    "company_name": company.name,
                },
                status=201,
            )

        except json.JSONDecodeError:
            return JsonResponse(
                {"status": "error", "message": "Invalid JSON data"}, status=400
            )
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    return JsonResponse(
        {"status": "error", "message": "Invalid request method"}, status=405
    )

@csrf_exempt
def sellmate_agent_chat(request):
    if request.method == "GET":
        # Return chat history for the current session
        session_id = request.session.get("session_id")
        if session_id:
            chat_history = SellMateAgentChatHistory.objects.filter(
                session_id=session_id
            ).order_by("timestamp")
            return JsonResponse(
                {
                    "chat_history": [
                        {
                            "user_message": chat.user_message,
                            "bot_response": chat.bot_response,
                        }
                        for chat in chat_history
                    ]
                }
            )
        return JsonResponse({"chat_history": []})

    elif request.method == "POST":
        try:
            data = json.loads(request.body)
            user_message = data.get("message", "").strip()
            if not user_message:
                return JsonResponse({"error": "Message cannot be empty"}, status=400)

            # Create a new session ID if it doesn't exist
            session_id = request.session.get("session_id", str(uuid.uuid4()))
            request.session["session_id"] = session_id

            # Get recent chat history for context
            recent_chats = SellMateAgentChatHistory.objects.filter(
                session_id=session_id
            ).order_by("-timestamp")[:3][::-1]  # Limit to the last 3 messages

            # System instruction for SellMate Agent (Admin AI - Business Owner's Assistant)
            system_instruction = """
            You are **SellMate Agent**, an advanced AI assistant that helps business owners create AI-powered personalized chatbots for their companies. 

            ## **Your Responsibilities:**
            1. **Helping business owners create a personalized chatbot** that serves as an AI support agent for their company.  
            2. **Ensuring chatbots accurately respond to visitor inquiries based on company details and FAQs.**  
            3. **Monitoring chatbot performance and providing insights to business owners on user interactions.**  
            
            ---  
            ## **1️⃣ Creating & Managing Company Chatbots**
            If a user requests to **create a chatbot for a new company**, collect the following details:

            - **Company Name**  
            - **Company Description** (What does the company do?)  
            - **Company Contact Information** (Email, Phone, etc.)  
            - **Company Website** (if available, else optional)  
            - **Common customer queries and their answers** (if available, else optional)  

            - **Ask only for missing details.**  
                - If the user **already provided** some information, do not ask again.  
                - Acknowledge what is provided and only ask for what is missing.  

            - **Before finalizing, confirm the collected details with the user**:  
                - "Here’s what I have collected so far:  
                   - **Company Name**: [Value]  
                   - **Description**: [Value]  
                   - **Contact Info**: [Value]  
                   - **Website**: [Value] (if available)  
                   - **FAQs for Chatbot**: [Value] (if available)  
                   Would you like to proceed with chatbot creation?"  

            - **Only after confirmation, generate a structured JSON response**:  

            ```json
            {
              "company_name": "<Company Name>",
              "company_info": "<Company Description>. Contact: <Contact Info>. Website: <Website>."
            }
            ```

            ---  
            ## **2️⃣ Response Guidelines**
            - **Never re-ask for details that have already been provided.**  
            - **Always confirm before generating the final JSON.**  
            - **If the user doesn’t have a website, proceed without it.**  
            - **Keep the conversation smooth, structured, and user-friendly.**  
            - **If a user asks unrelated personal questions, politely decline and steer the conversation back to business-related topics.**  

            ---  
            ## **3️⃣ Example Scenarios**
            ✅ **Correct Flow (User provides all info at once)**  
            **User:** "I want to create a company named Lysum. We provide health supplements offline. Contact: info@lyesum.com. Website: lysum.com."  
            **AI:** "Great! Here’s what I’ve collected:  
                   - **Company Name**: Lysum  
                   - **Description**: Lysum provides health supplements offline.  
                   - **Contact Info**: info@lyesum.com  
                   - **Website**: lysum.com  
                   Would you like to proceed with chatbot creation?"  
            **User:** "Yes."  
            ✅ **AI (Only Now Returns JSON):**  
            ```json
            {
              "company_name": "Lysum",
              "company_info": "Lysum provides health supplements offline. Contact: info@lyesum.com. Website: lysum.com."
            }
            ```

            ✅ **Correct Flow (User provides partial info)**  
            **User:** "I want to create a company named Lysum. We provide health supplements offline."  
            **AI:** "Got it! Please provide your contact details (email, phone, etc.). Also, do you have a website?"  
            **User:** "Contact: info@lyesum.com."  
            **AI:** "Do you have a website? If not, that's okay."  
            **User:** "We don’t have a website."  
            **AI:** "Great! Here’s what I’ve collected:  
                   - **Company Name**: Lysum  
                   - **Description**: Lysum provides health supplements offline.  
                   - **Contact Info**: info@lyesum.com  
                   - **Website**: Not Available  
                   Would you like to proceed with chatbot creation?"  
            **User:** "Yes."  
            ✅ **AI (Only Now Returns JSON):**  
            ```json
            {
              "company_name": "Lysum",
              "company_info": "Lysum provides health supplements offline. Contact: info@lyesum.com."
            }
            ```

            🚀 **Your goal is to ensure a smooth, structured, and efficient experience for the user while creating AI-powered chatbots.**
            """



            # Prepare conversation history
            contents = []

            # Add system instruction as part of the first user message
            if not recent_chats:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=system_instruction)],
                    )
                )

            # Add chat history
            for chat in recent_chats:
                contents.extend(
                    [
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=chat.user_message)],
                        ),
                        types.Content(
                            role="assistant",
                            parts=[types.Part.from_text(text=chat.bot_response)],
                        ),
                    ]
                )

            # Add current user message
            contents.append(
                types.Content(
                    role="user", parts=[types.Part.from_text(text=user_message)]
                )
            )

            # Create configuration
            config = types.GenerateContentConfig(
                temperature=0.7,
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

            # Generate response
            response = client.models.generate_content(
                model=model, contents=contents, config=config
            )

            response_text = response.text

            # Save to database
            chat = SellMateAgentChatHistory.objects.create(
                session_id=session_id,
                user_message=user_message,
                bot_response=response_text,
            )

            return JsonResponse({"response": response_text})

        except Exception as e:
            logger.error("Error in sellmate_agent_chat: %s", str(e))
            import traceback

            logger.error(traceback.format_exc())
            return JsonResponse({"error": "Failed to process request"}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)
