# AI-RAG-Enabled Digital Human

## Work in Progress (WIP)

This repository contains experimental work on developing an AI-powered digital human with Retrieval-Augmented Generation (RAG) capabilities. The project leverages NVIDIA's Avatar Cloud Engine (ACE) for digital human animation and voice synthesis, combined with advanced RAG techniques for enhanced conversational abilities.

## Overview

This project is part of a digital human conversational interface. It uses smart context management along with a conversational interface to enable fluid conversation while minimizing token usage or compute required.

## Features

- Speech recognition and natural language understanding
- Retrieval-Augmented Generation (RAG) for knowledge-based responses
- Digital human animation and rendering
- Audio-to-face synchronization
- Conversational AI capabilities

## Technologies

- Python
- NVIDIA Avatar Cloud Engine (ACE)
- Vosk speech recognition models
- RAG (Retrieval-Augmented Generation)
- NLP processing with Spacy

## Setup Instructions

### Language Models and Dependencies for Spacy

```bash
pip install -U pip setuptools wheel
pip install -U 'spacy[apple]'
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

### Running Audio-to-Face Examples

For Claire avatar:
```bash
python ./nim_a2f_client.py ./audio/sample.wav ./config/config_claire.yml --apikey $NVIDIA_NIM_API_KEY --function-id 462f7853-60e8-474a-9728-7b598e58472c
```

For Mark avatar:
```bash
python ./nim_a2f_client.py ./audio/sample.wav ./config/config_mark.yml --apikey $NVIDIA_NIM_API_KEY --function-id 945ed566-a023-4677-9a49-61ede107fd5a
```

## Status

This project is in active development. The current implementation is a work in progress, focusing on integrating RAG capabilities with digital human interfaces to create more intelligent and responsive conversational agents.
