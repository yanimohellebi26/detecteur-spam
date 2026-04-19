# AI-Powered Spam Detector with Telegram Bot

## Description

An intelligent spam detection system that combines machine learning with a live Telegram bot interface. Users can send any message to the bot and receive an instant classification — spam or legitimate — backed by a logistic regression model trained on a labelled SMS dataset.

## What it brings

Spam filtering is one of the most practical and ubiquitous applications of NLP. This project demonstrates the full pipeline from raw data to a deployed, interactive bot: cleaning and vectorising text, training and evaluating a classifier, and wrapping it in a real-time messaging interface that anyone can try without technical knowledge.

## How it works

The dataset is pre-processed with TF-IDF vectorisation to convert messages into numerical features. A logistic regression model is trained and cross-validated, with metrics (precision, recall, F1-score, confusion matrix, ROC curve) saved for transparency. The Telegram bot loads the serialised model at startup and classifies each incoming message in milliseconds.

## Status

✅ Complete — model trained, bot deployed and operational.

## Tech Stack

`Python` · `scikit-learn` · `TF-IDF` · `Telegram Bot API` · `Pandas` · `Matplotlib`
