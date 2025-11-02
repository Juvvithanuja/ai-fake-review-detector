from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User 
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import ReviewRecord
from django.contrib.auth import logout
from django.shortcuts import redirect
from django.contrib.auth.models import User
from django.shortcuts import render


import joblib
import re
import string
import json

# -----------------------------
# Load model and vectorizer
# -----------------------------
model = joblib.load("fake_review_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -----------------------------
# Text cleaning function
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------
# Predict single review
# -----------------------------
def predict_single(review_text):
    clean_review = clean_text(review_text)
    vect_review = vectorizer.transform([clean_review])
    pred = model.predict(vect_review)[0]
    pred_proba = model.predict_proba(vect_review)[0]

    label = "Fake Review ❌" if pred == 1 else "Real Review ✅"
    fake_conf = round(float(pred_proba[1]) * 100, 2)
    real_conf = round(float(pred_proba[0]) * 100, 2)

    return {
        "review": review_text,
        "prediction": label,
        "real_confidence": real_conf,
        "fake_confidence": fake_conf
    }

# -----------------------------
# Main page
# -----------------------------
def index(request):
    return render(request, "reviewdetectorapp/index.html")

# -----------------------------
# API: Handle review prediction
# -----------------------------
@login_required  # user must log in before scanning
def predict(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            reviews = data.get("reviews", "").split("\n")

            results = []
            for r in reviews:
                r = r.strip()
                if not r:
                    continue

                result = predict_single(r)

                # ✅ Save to database
                ReviewRecord.objects.create(
                    user=request.user,
                    review_text=r,
                    prediction=result["prediction"],
                    real_confidence=result["real_confidence"],
                    fake_confidence=result["fake_confidence"],
                )

                results.append(result)

            return JsonResponse({"results": results})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request method"}, status=400)
# -----------------------------
# Login Page
# -----------------------------
def user_login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('index')  # redirect to your detector home page
        else:
            messages.error(request, "Invalid username or password")

    return render(request, "login.html")
def register(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
            return redirect("register")
        User.objects.create_user(username=username, password=password)
        messages.success(request, "Account created successfully. Please log in.")
        return redirect("login")
    return render(request, "register.html")

@login_required
def history(request):
    records = ReviewRecord.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'history.html', {'records': records})
def user_logout(request):
    logout(request)
    messages.success(request, "You have been logged out successfully.")
    return redirect("login")
def show_users(request):
    users = User.objects.all()
    return render(request, 'show_users.html', {'users': users})

@login_required
def index(request):
    return render(request, "reviewdetectorapp/index.html")
