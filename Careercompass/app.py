from flask import Flask, request, jsonify, render_template, redirect, url_for ,session
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import openai
from google.generativeai import GenerativeModel, GenerationConfig

from google.auth import jwt

from flask_sqlalchemy import SQLAlchemy
import requests
import bcrypt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


#tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
#model1 = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")   

# Load the machine learning model
data = pd.read_csv(r'C:\Users\personal\Downloads\Careercompass\careercampass.csv')

# Select relevant columns
data_selected = data[['Englishmarks', 'Socialmarks', 'Sciencemarks', 'Mathsmarks', 'Biologymarks',
                      'Interests', 'FinancialIncome', 'ExtracurricularActivities', 'Attendance',
                      'AptitudeTestScore', 'CareerPrediction']]

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Interests', 'FinancialIncome', 'ExtracurricularActivities']
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data_selected[col] = label_encoders[col].fit_transform(data_selected[col])

# Separate features and target variable
X = data_selected.drop('CareerPrediction', axis=1)
y = data_selected['CareerPrediction']

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Initialize Flask application
app = Flask(__name__)

app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

with app.app_context():
    db.create_all()

# Define route for home page
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get user input from the form
        user_input = {}
        for col in X.columns:
            user_input[col] = float(request.form.get(col, 0))  # Convert input to float
        
        # Call your machine learning function to predict career based on user input
        predicted_career = predict_career(user_input)
        return redirect(url_for('result', predicted_career=predicted_career))
    else:
        # Render the input form template
        return render_template('index.html')

# Define function to predict career based on user input
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the request
    user_input = {}
    for col in X.columns:
        user_input[col] = float(request.form.get(col, 0))  # Convert input to float

    # Call your machine learning function to predict career based on user input
    predicted_career = predict_career(user_input)
    return jsonify({'predicted_career': predicted_career})


def predict_career(user_input):
    input_data = pd.DataFrame([user_input])
    predicted_career = model.predict(input_data)
    return predicted_career[0]  # Return the first prediction


# Define route for displaying result
@app.route('/result')
def result():
    predicted_career = request.args.get('predicted_career')
    return render_template('result.html', predicted_career=predicted_career)

'''@app.route("/Chatbot")
def index():
    return render_template('chat.html')'''

'''--------------------------------------------------------------------------------------------------------'''


def get_response(user_message):
    responses = {
        "What career should I pursue?": "Think about your interests, strengths, and values. You can also consider taking career assessments or speaking with a career counselor for guidance.",
        "How do I write a resume?": "A good resume highlights your skills, experiences, and achievements. Use a clear format, tailor it to the job you're applying for, and proofread carefully.",
        "What skills are in demand?": "Some in-demand skills include communication, problem-solving, adaptability, and digital literacy. It's also important to stay updated on industry-specific skills.",
        "How do I prepare for a job interview?": "Research the company, practice common interview questions, and prepare examples to showcase your skills and experiences. Don't forget to dress professionally and arrive on time!",
        "Should I pursue higher education?": "Consider your career goals and whether further education aligns with them. Higher education can lead to greater opportunities, but it's important to weigh the costs and benefits.",
        "How do I find internships or job opportunities?": "Utilize online job boards, networking events, career fairs, and professional organizations. You can also reach out to your university's career services office for assistance.",
        "How do I advance in my career?": "Continuously develop your skills, seek mentorship, and pursue opportunities for growth. Set clear goals and be proactive in seeking out new challenges.",
        "How do I balance work and personal life?": "Prioritize your tasks, set boundaries, and make time for activities you enjoy outside of work. Effective time management and communication are key.",
        "What are some common career paths?": "Common career paths include fields like healthcare, technology, finance, education, marketing, and engineering. Explore different industries to find what interests you.",
        "How do I deal with career burnout?": "Take breaks, practice self-care, and seek support from friends, family, or a mental health professional. It's important to address burnout early to prevent further exhaustion.",



        "What career path aligns with my interests, strengths, and values?":"Reflect on your passions, skills, and what matters most to you. Consider how these factors can guide you towards a fulfilling career. Additionally, explore career assessments or seek guidance from a career counselor for further insight.",

        "How can I maintain a healthy work-life balance while pursuing my career goals?":" Prioritize self-care, set boundaries between work and personal life, and allocate time for activities that rejuvenate you. Strive for harmony between your professional and personal responsibilities to avoid burnout.",
        "What strategies can I use to overcome challenges and setbacks in my career?":"Cultivate resilience by viewing challenges as opportunities for growth. Seek support from mentors, develop problem-solving skills, and maintain a positive mindset to navigate obstacles effectively.",
        "How do I identify and leverage my strengths in the workplace?":"Take stock of your unique talents and abilities, and find ways to apply them in your current role or future career endeavors. Capitalize on your strengths to excel in your work and contribute meaningfully to your organization.",
        "What steps can I take to advance my career and achieve my long-term goals?":"Set clear objectives, create a roadmap for professional development, and actively pursue opportunities for growth. Network with industry professionals, seek feedback, and continuously strive for improvement.",
        "How do I handle career transitions or changes effectively?":"Embrace change as a natural part of career growth and remain adaptable to new opportunities. Develop a proactive mindset, hone transferable skills, and seek guidance from mentors or career coaches as needed.",
        "What are effective strategies for networking and building professional relationships?":"Engage in networking events, cultivate genuine connections with colleagues and industry peers, and leverage online platforms to expand your professional network. Practice active listening, offer support to others, and nurture relationships over time.",
        "How can I develop a personal brand to stand out in my field?":'Define your unique value proposition and communicate it consistently through your online presence, professional demeanor, and contributions to your field. Cultivate a reputation for excellence and authenticity to distinguish yourself from others.',
        "What resources or opportunities can I explore to further my career development?":"Seek out mentorship programs, attend workshops and conferences, and pursue continuous learning through courses or certifications. Stay informed about industry trends and advancements to remain competitive in your field."
    }
    return responses.get(user_message, "I'm sorry, I don't understand that question.")

@app.route("/Chatbot")
def index():
    return render_template("chat.html")

@app.route("/get_response", methods=["POST"])
def get_bot_response():
    user_message = request.form["user_message"]
    bot_response = get_response(user_message)
    return jsonify({"bot_response": bot_response})



'''-------------------------------------------------------------------------------------------------------------------------------------------------'''

#login function
@app.route('/',methods=['GET','POST'])
def login():
    if request.method=='POST':
        email=request.form['email']
        password=request.form['password']

        user=User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['name']=user.name
            session['email']=user.email
            session['password']=user.password
            return redirect('/tests')
        else:
            return render_template('login.html',error='Invalid User')


    return render_template('login.html')

@app.route('/register',methods=['GET','POST'])
def register():
    if request.method=='POST':
        #handle request
        name=request.form['name']
        email=request.form['email']
        password=request.form['password']

        new_user=User(name=name,email=email,password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/')

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('email',None)
    return redirect('/')

@app.route('/tests')
def tests():
    return render_template('tests.html')

@app.route('/aptitude')
def aptitude():
    return render_template('aptitude.html')

@app.route('/english')
def english():
    return render_template('english.html')

@app.route('/reasoning')
def reasoning():
    return render_template('reasoning.html')

@app.route('/gk')
def gk():
    return render_template('gk.html')

if __name__ == '__main__':
    app.run(debug=True)
