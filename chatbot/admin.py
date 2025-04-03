from django.contrib import admin
from .models import ChatHistory
from .models import Company

@admin.register(ChatHistory)
class ChatHistoryAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'user_message', 'bot_response', 'timestamp')
    search_fields = ('session_id', 'user_message', 'bot_response')
    list_filter = ('timestamp',)
    ordering = ('-timestamp',)

@admin.register(Company)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ('name', 'system_instruction')  # Display these fields in the admin list view
    search_fields = ('name',)  # Add a search bar for the company name
