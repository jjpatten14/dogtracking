#!/usr/bin/env python3
"""
Test script for the pre-compact memory update hook
"""

import json
import subprocess
import tempfile
from pathlib import Path

def create_mock_transcript():
    """Create a mock transcript for testing"""
    mock_transcript = {
        "messages": [
            {
                "role": "user",
                "content": "I'm having an issue with the video streaming being very slow"
            },
            {
                "role": "assistant", 
                "content": "I'll help fix the slow video streaming issue. The problem is the blocking video capture in the main thread.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "Edit",
                            "arguments": json.dumps({
                                "file_path": "/mnt/c/yard/app.py",
                                "old_string": "# old code",
                                "new_string": "# new threading code"
                            })
                        }
                    }
                ]
            },
            {
                "role": "assistant",
                "content": "I've implemented a threading solution to fix the video streaming performance issue. The camera now runs in a separate thread at 30fps."
            },
            {
                "role": "user",
                "content": "That fixed it! The interface is much more responsive now."
            }
        ]
    }
    return mock_transcript

def test_hook():
    """Test the pre-compact hook functionality"""
    print("Testing pre-compact memory update hook...")
    
    # Create mock transcript file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        transcript = create_mock_transcript()
        json.dump(transcript, f, indent=2)
        transcript_path = f.name
    
    # Create mock hook input data
    hook_data = {
        "session_id": "test_session_12345",
        "transcript_path": transcript_path,
        "hook_event_name": "PreToolUse",
        "tool_name": "compact"
    }
    
    try:
        # Run the hook script
        process = subprocess.run(
            ["python3", "/mnt/c/yard/scripts/pre_compact_memory_update.py"],
            input=json.dumps(hook_data),
            text=True,
            capture_output=True,
            timeout=30
        )
        
        print(f"Hook exit code: {process.returncode}")
        print(f"Hook stdout: {process.stdout}")
        if process.stderr:
            print(f"Hook stderr: {process.stderr}")
        
        # Check if memory file was updated
        memory_file = Path("/mnt/c/yard/mcp-memory/dist/memory.json")
        if memory_file.exists():
            print(f"Memory file size: {memory_file.stat().st_size} bytes")
            
            # Read last few lines to see if new entries were added
            with memory_file.open('r') as f:
                lines = f.readlines()
                if lines:
                    print("Last few memory entries:")
                    for line in lines[-5:]:
                        try:
                            entry = json.loads(line.strip())
                            print(f"  {entry.get('type', 'unknown')}: {entry.get('name', 'no name')}")
                        except:
                            pass
        
        return process.returncode == 0
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False
    
    finally:
        # Clean up
        Path(transcript_path).unlink(missing_ok=True)

if __name__ == "__main__":
    success = test_hook()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")