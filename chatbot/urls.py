from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('add-company/', views.add_company, name='add_company'),
    path('companies/', views.companies, name='companies'),
    # path('ai-chat/', views.ai_chat, name='ai_chat'),
    path('companies/<int:company_id>/chats/', views.company_chats, name='company_chats'),
    path('companies/<int:company_id>/sessions/', views.company_sessions, name='company_sessions'),
    path('sessions/<str:session_id>/', views.session_chat, name='session_chat'),
    path('save-company/', views.save_company, name='save_company'),
    path('sellmate-agent-chat/', views.sellmate_agent_chat, name='sellmate_agent_chat'),
    path('<str:company_name>/', views.home, name='home'),
    path('<str:company_name>/chat/', views.chat, name='chat'),
    path('<str:company_name>/embed.js', views.dynamic_embed_js, name='dynamic_embed_js'),
]