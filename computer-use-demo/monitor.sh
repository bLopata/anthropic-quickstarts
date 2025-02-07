#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Monitor the log file in real-time
tail -f logs/agent.log 