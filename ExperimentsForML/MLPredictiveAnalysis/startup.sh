#!/bin/bash
# Simple startup script for the funding analysis schedulers

echo "==============================================="
echo "Starting Funding Analysis Scheduler"
echo "==============================================="

# Change to the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Define the Python executable to use
PYTHON="python"

# Activate virtual environment if it exists
if [ -f "../.venv/bin/activate" ]; then
    source "../.venv/bin/activate"
    echo "Activated virtual environment"
elif [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
    echo "Activated virtual environment"
fi

# Function to start an analysis in a separate terminal
start_analysis() {
    local script="$1"
    local interval="$2"
    
    echo "Starting $script with $interval hour interval..."
    
    # Use terminal if available, otherwise run in background
    if command -v osascript &> /dev/null; then
        # macOS
        osascript -e "tell application \"Terminal\" to do script \"cd '$SCRIPT_DIR' && $PYTHON $script --interval $interval\""
    elif command -v gnome-terminal &> /dev/null; then
        # Linux with GNOME
        gnome-terminal -- bash -c "cd '$SCRIPT_DIR' && $PYTHON $script --interval $interval; exec bash"
    elif command -v xterm &> /dev/null; then
        # Linux with xterm
        xterm -e "cd '$SCRIPT_DIR' && $PYTHON $script --interval $interval; exec bash" &
    else
        # Fallback: run in background
        echo "No terminal emulator found, running in background..."
        nohup $PYTHON "$script" --interval "$interval" > "${script%.py}_scheduler.log" 2>&1 &
        echo "Process started with PID $! - logs in ${script%.py}_scheduler.log"
    fi
}

# Ask the user what to start
echo "What would you like to start?"
echo "1. Funding Anomaly Detection (24 hour schedule)"
echo "2. Funding Continuation Analysis (24 hour schedule)"
echo "3. Both (24 hour schedule)"
echo "4. Custom intervals"
echo "5. Exit"
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        start_analysis "funding_anomaly_detection.py" 24
        ;;
    2)
        start_analysis "funding_continuation.py" 24
        ;;
    3)
        start_analysis "funding_anomaly_detection.py" 24
        start_analysis "funding_continuation.py" 24
        ;;
    4)
        read -p "Enter interval for Anomaly Detection (hours): " anomaly_interval
        read -p "Enter interval for Funding Continuation (hours): " continuation_interval
        read -p "Start Anomaly Detection? (y/n): " start_anomaly
        read -p "Start Funding Continuation? (y/n): " start_continuation
        
        if [[ "$start_anomaly" == "y" ]]; then
            start_analysis "funding_anomaly_detection.py" "$anomaly_interval"
        fi
        
        if [[ "$start_continuation" == "y" ]]; then
            start_analysis "funding_continuation.py" "$continuation_interval"
        fi
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "Scheduler(s) started. Check terminal windows for output."
echo "Press Ctrl+C in those windows to stop the schedulers." 