from django.db import models

class Company(models.Model):
    name = models.CharField(max_length=255, unique=True)  # Company name
    system_instruction = models.TextField()  # System instruction for the chatbot

    def __str__(self):
        return self.name


class ChatHistory(models.Model):
    session_id = models.CharField(max_length=255)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    user_message = models.TextField()
    bot_response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=[('Open', 'Open'), ('Completed', 'Completed')], default='Open')

    def __str__(self):
        return f"Session {self.session_id} at {self.timestamp}"


class SellMateAgentChatHistory(models.Model):
    session_id = models.CharField(max_length=255)
    user_message = models.TextField()
    bot_response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"SellMate Session {self.session_id} at {self.timestamp}"