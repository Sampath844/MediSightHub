import json
import os
import uuid
from datetime import datetime
from PIL import Image
import google.generativeai as genai

class ReportQAChat:
    def get_qa_chat_store(self):
        if os.path.exists("qa_chat_store.json") and os.path.getsize("qa_chat_store.json") > 0:
            with open("qa_chat_store.json", "r") as f:
                try: return json.load(f)
                except json.JSONDecodeError: return {"qa_rooms": {}}
        return {"qa_rooms": {}}

    def save_qa_chat_store(self, store):
        with open("qa_chat_store.json", "w") as f:
            json.dump(store, f, indent=2)

    def add_message(self, room_id, user_name, content, image=False):
        store = self.get_qa_chat_store()
        if room_id in store["qa_rooms"]:
            message = {
                "id": str(uuid.uuid4()), "user": user_name, "content": content,
                "image_attached": image, "timestamp": datetime.now().isoformat()
            }
            store["qa_rooms"][room_id]["messages"].append(message)
            self.save_qa_chat_store(store)

    def create_qa_room(self, room_id, creator_name, report_summary=""):
        store = self.get_qa_chat_store()
        if room_id not in store["qa_rooms"]:
            room = {
                "id": room_id, "created_at": datetime.now().isoformat(), "creator": creator_name,
                "report_summary": report_summary, "participants": [creator_name, "Medical AI Assistant"],
                "messages": [], "status": "active"
            }
            welcome = {
                "id": str(uuid.uuid4()), "user": "Medical AI Assistant",
                "content": f"Hello {creator_name}! I'm ready to discuss your report. Ask me anything, or attach another image if you have a related question.",
                "timestamp": datetime.now().isoformat()
            }
            room["messages"].append(welcome)
            store["qa_rooms"][room_id] = room
            self.save_qa_chat_store(store)

    def get_messages(self, room_id):
        return self.get_qa_chat_store().get("qa_rooms", {}).get(room_id, {}).get("messages", [])

    def get_qa_rooms(self):
        rooms = list(self.get_qa_chat_store().get("qa_rooms", {}).values())
        rooms.sort(key=lambda x: x["created_at"], reverse=True)
        return rooms

    def delete_qa_room(self, room_id):
        store = self.get_qa_chat_store()
        if room_id in store.get("qa_rooms", {}):
            del store["qa_rooms"][room_id]
            self.save_qa_chat_store(store)
            return True
        return False

class ReportQASystem:
    def __init__(self, api_key):
        self.api_key = api_key
        self.qa_chat = ReportQAChat()
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')


    def generate_gemini_response(self, question, context="", image: Image = None):
        try:
            system_prompt = "You are a compassionate medical AI assistant helping patients understand their reports. Explain terms simply and always encourage consultation with their doctor."
            user_prompt = f"Based on this report: {context}\n\nPatient question: {question}"
            
            content = [system_prompt, user_prompt]
            if image:
                content.append("Please also consider this attached image in your answer:")
                content.append(image)
            
            response = self.model.generate_content(content)
            return response.text
        except Exception as e:
            return f"I'm sorry, an error occurred: {str(e)}"

    def create_qa_room(self, room_id, creator_name, report_summary=""):
        return self.qa_chat.create_qa_room(room_id, creator_name, report_summary)

    def add_message(self, room_id, user_name, content, image=False):
        return self.qa_chat.add_message(room_id, user_name, content, image)

    def get_messages(self, room_id):
        return self.qa_chat.get_messages(room_id)

    def get_qa_rooms(self):
        return self.qa_chat.get_qa_rooms()

    def delete_qa_room(self, room_id):
        return self.qa_chat.delete_qa_room(room_id)

    def answer_question(self, room_id, question, report_context="", findings=None, image: Image = None):
        try:
            enhanced_context = report_context
            if findings:
                enhanced_context += "\n\nKey findings mentioned: " + ", ".join(findings)
            
            answer = self.generate_gemini_response(question, context=enhanced_context, image=image)
            
            self.add_message(room_id, "Medical AI Assistant", answer)
            return answer
        except Exception as e:
            error_msg = f"Apologies, an issue occurred: {str(e)}"
            self.add_message(room_id, "Medical AI Assistant", error_msg)
            return error_msg