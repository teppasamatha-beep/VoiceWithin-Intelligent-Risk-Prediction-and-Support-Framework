VoiceWithin is a mental health support platform that combines machine learning, AI, and journaling tools to promote emotional well-being.
1. Data Collection & Input Layer:Users answer simple Yes/No questions related to mental health symptoms such as stress, anxiety, depression, and panic attacks through a secure web interface. The data is sent to the backend using JavaScript’s Fetch API.
2. Suicide Risk Prediction (ML Model):A trained machine learning model (Logistic Regression or Random Forest) analyzes user responses and predicts suicide risk as a percentage using probability scores. Based on the risk level, the system either recommends chatbot support and journaling or gently suggests seeking professional help.
3. AI Chatbot Integration:An empathetic AI chatbot provides emotional support and performs real-time sentiment analysis on user messages. Responses are tailored based on detected emotional tone, offering empathy, encouragement, or coping suggestions when needed.
4. Mood Journaling & Tracking:Users can log daily moods using predefined icons and write diary-style entries. This feature supports both quick mood tracking and deeper emotional reflection, with data securely stored.
5. Notifications & Reminders:The system sends daily check-ins, motivational messages, and gentle reminders to encourage self-awareness and consistent mental well-being engagement.

Development Tools:
1. Frontend:HTML, CSS, JavaScript.
2. Backend: Python (Flask Framework)
3. Database: MongoDB Atlas 
4. AI/ML Libraries: scikit-learn, pandas, NumPy, Random Forest.  

MongoDB Collections:
1. users– stores user credentials and basic profile information.
2. survey\_responses – records answers to Yes/No mental health questions.
3. journal\_entries – saves date, mood emoji, and text reflections.
4. chat\_logs – stores chatbot interactions and sentiment of user chat.
