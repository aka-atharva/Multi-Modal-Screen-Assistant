# Multi-Modal-Screen-Assistant
A comprehensive AI-powered desktop assistant that combines multiple AI models to create an intelligent programming companion. This tool integrates visual processing, text analysis, and voice interaction to enhance programmer productivity through an intuitive interface.

## Overview

The Multi-Modal Screen Assistant is designed to serve as an AI-powered companion for developers, combining the capabilities of Llama, OpenAI-Whisper, and Google Gemini models. It processes multiple types of input including screen captures, clipboard content, and voice commands to provide contextual assistance during programming tasks.

## Key Features

**Visual Processing**
- Real-time screen capture analysis
- Live image processing using Google Gemini
- Contextual understanding of visual content

**Text Processing**
- Clipboard content analysis
- LLM-powered text interpretation
- Multi-language text processing

**Voice Interaction**
- Speech recognition using OpenAI Whisper
- Voice command processing

**AI Integration**
- RAG-based architecture for knowledge retrieval
- Context-aware conversations
- Customizable knowledge base
- Interactive AI chatbot functionality

## Technical Architecture

**Core Components**
- Base Model: Llama for conversational AI
- Vision Model: Google Gemini for image processing
- Speech Model: OpenAI Whisper for voice recognition
- RAG Framework: Custom implementation for knowledge retrieval

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Multi-Modal-Screen-Assistant.git
cd Multi-Modal-Screen-Assistant
```

2. Set up virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage Instructions

**Starting the Application**
```bash
python fox.py
```

**Basic Operations**
1. Voice Control
   - Speak commands naturally
   - System processes speech using Whisper
   - Receives AI-generated responses

2. Screen Analysis
   - Capture screen content
   - System analyzes visual elements
   - Provides contextual information

3. Text Processing
   - Copy text to clipboard
   - System automatically analyzes content
   - Receives relevant suggestions

4. Language Options
   - Select preferred language
   - System adapts responses accordingly
   - Maintains consistent interaction

## Configuration

**System Settings**
- Knowledge Base: Customize information sources
- Model Parameters: Adjust response characteristics
- Voice Settings: Configure speech recognition
- Language Preferences: Set interaction language

## Development and Contribution

1. Fork the repository
2. Create a feature branch:
```bash
git checkout -b feature/NewFeature
```
3. Implement changes
4. Submit pull request

## Technical Requirements

- Python 3.9 or higher
- Sufficient RAM for model operations
- Graphics capability for screen capture
- Microphone for voice input

## Support and Documentation

For assistance:
- Create an issue in the repository
- Review existing documentation
- Contact project maintainers

## Future Development Plans

**Planned Enhancements**
- Advanced modal integration
- Extended language support
- RAG system improvements
- Performance optimization
- Additional model integration

## License

This project is released under the MIT License. See LICENSE file for details.
