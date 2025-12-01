# 5-Day-AI-Agents-Intensive-Course-with-Google-2025-Capstone-Project

Doctor Visit Companion

An agentic AI system that records, summarizes, and explains your doctor visits using Google Gemini, helping you understand medications, track follow-ups, and stay organized with a clean Streamlit interface.

# Overview

Doctor Visit Companion is an AI-powered assistant designed to help patients capture, understand, and organize the essential details from their medical appointments.
It uses multiple specialized AI agents to summarize visit notes, explain medications in simple language, and suggest appropriate follow-up actions, all while storing your visit history securely on your device.

Built with Python, Streamlit, and Google Gemini, this application acts as your personal health companion by making medical information clearer, accessible, and easier to follow.

Features
✔ AI Summaries of Doctor Visits

Turns raw notes from your appointment into clear, friendly explanations and key bullet-point highlights.

✔ Medication Explanations

Each medication is analyzed and described in simple, patient-friendly language.

✔ Follow-Up Recommendations

AI suggests reasonable follow-up timelines based on your visit details.

✔ Visit History

All saved visits are organized and searchable, allowing you to review past summaries anytime.

✔ Appointment Manager

Create and view follow-up appointments linked to previous visits.

✔ Agent Trace

See exactly how AI agents processed your data for transparency and debugging.

# Architecture

The app uses three primary AI agents:

Visit Summarizer Agent
Generates structured summaries and highlights.

Medication Explainer Agent
Explains what medications do and how they help your condition.

Follow-Up Planner Agent
Suggests when and with whom you should book your next visit.

All agents interact with the Gemini model through a unified helper function and store their results in local JSON-based memory.

# Technologies Used

Python 3

Streamlit (UI)

Google Gemini via google-genai

Dataclasses for data modeling

JSON storage for persistent memory

# Set Your Gemini API Key

You must set your Gemini API key before running the app.

macOS / Linux:
export GEMINI_API_KEY="your_api_key_here"

Windows (PowerShell):
setx GEMINI_API_KEY "your_api_key_here"

Run these commands in the terminal

# Running the App:
streamlit run app.py

# How to Use the App
1. Record a New Visit

Go to the “New Visit” tab:

Enter the doctor’s name, visit date, specialty, location, and reason.

Paste your visit notes or transcript.

Optionally list medications (one per line, format: Name | dosage instructions).

Click Save & Generate AI Summary.

The app will:

Run the Summarizer Agent

Run the Medication Explainer Agent

Run the Follow-up Planner Agent

Store all outputs locally

2. View Visit History

In the Visit History tab:

Search past visits

View summaries, highlights, medications, and explanations

3. Manage Appointments

In the Appointments tab:

Schedule follow-ups linked to previous visits

View upcoming appointments

Copy calendar-friendly text snippets

4. Inspect AI Reasoning

The Agent Trace tab shows:

When each agent ran

What inputs does it use

A preview of the outputs

# Author
Joshua Harrell
Qimora Mason
Luz Rivera
