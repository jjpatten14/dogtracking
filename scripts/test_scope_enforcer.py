#!/usr/bin/env python3
"""
Test script for the scope enforcer hook
"""

import json
import subprocess
import tempfile
from pathlib import Path

def create_mock_transcript(user_message, assistant_response="I'll help with that."):
    """Create a mock transcript for testing"""
    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ]
    }

def test_scenario(name, transcript, tool_name, tool_input, expected_decision):
    """Test a specific scenario"""
    print(f"\n🧪 Testing: {name}")
    
    # Create mock transcript file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for msg in transcript['messages']:
            f.write(json.dumps(msg) + '\n')
        transcript_path = f.name
    
    # Create hook input data
    hook_data = {
        "session_id": "test_session",
        "transcript_path": transcript_path,
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": tool_input
    }
    
    try:
        # Run the scope enforcer
        process = subprocess.run(
            ["python3", "/mnt/c/yard/scripts/scope_enforcer.py"],
            input=json.dumps(hook_data),
            text=True,
            capture_output=True,
            timeout=10
        )
        
        if process.stdout:
            response = json.loads(process.stdout.strip())
            decision = response.get('decision', 'unknown')
            reason = response.get('reason', 'no reason')
            
            # Check if result matches expectation
            success = decision == expected_decision
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   {status} - Decision: {decision}")
            print(f"   Reason: {reason}")
            
            return success
        else:
            print(f"   ❌ FAIL - No output from hook")
            if process.stderr:
                print(f"   Error: {process.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ FAIL - Error: {e}")
        return False
    finally:
        # Clean up
        Path(transcript_path).unlink(missing_ok=True)

def run_tests():
    """Run all test scenarios"""
    print("🔍 Testing Scope Enforcer Hook")
    print("=" * 50)
    
    tests = []
    
    # Test 1: Normal file edit should be approved
    tests.append(test_scenario(
        "Normal file edit",
        create_mock_transcript("Fix the video streaming issue in app.py"),
        "Edit",
        {"file_path": "/mnt/c/yard/app.py", "old_string": "old", "new_string": "new"},
        "approve"
    ))
    
    # Test 2: Documentation creation should be blocked
    tests.append(test_scenario(
        "Unwanted documentation creation",
        create_mock_transcript("Fix the video streaming issue"),
        "Write", 
        {"file_path": "/mnt/c/yard/README.md", "content": "# Project Documentation\n..."},
        "block"
    ))
    
    # Test 3: Dangerous command should be blocked
    tests.append(test_scenario(
        "Dangerous bash command",
        create_mock_transcript("Clean up the project"),
        "Bash",
        {"command": "rm -rf /tmp/*"},
        "block"
    ))
    
    # Test 4: Read operations should always be approved
    tests.append(test_scenario(
        "Read operation",
        create_mock_transcript("Help me understand the code"),
        "Read",
        {"file_path": "/mnt/c/yard/app.py"},
        "approve"
    ))
    
    # Test 5: "no rails" keyword should disable enforcement
    tests.append(test_scenario(
        "Disabled with 'no rails' keyword",
        create_mock_transcript("no rails - create comprehensive documentation"),
        "Write",
        {"file_path": "/mnt/c/yard/COMPREHENSIVE_DOCS.md", "content": "Lots of docs..."},
        "approve"
    ))
    
    # Test 6: Scope creep language should be blocked
    tests.append(test_scenario(
        "Scope creep language detection",
        create_mock_transcript("Fix the button color", "I'll fix the button color. While we're at it, let me also refactor the entire UI system."),
        "Write",
        {"file_path": "/mnt/c/yard/ui_refactor.py", "content": "# Complete UI refactor"},
        "block"
    ))
    
    # Calculate results
    passed = sum(tests)
    total = len(tests)
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Scope enforcer is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)