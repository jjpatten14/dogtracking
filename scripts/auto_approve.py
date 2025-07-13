#!/usr/bin/env python3
"""
Claude Code Auto-Approval Hook

Automatically approves permission prompts when "auto approve" is present in the user's prompt.
Default behavior: Manual approval required (safe by default)
Activation: Include "auto approve" in your prompt
No state persistence - each prompt is evaluated independently.
"""

import json
import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
log_file = Path(__file__).parent / "auto_approve.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

class AutoApprover:
    """Handles automatic approval of Claude Code permission prompts"""
    
    def __init__(self, hook_data):
        self.hook_data = hook_data
        self.session_id = hook_data.get('session_id', 'unknown')
        self.prompt_text = hook_data.get('prompt', '')
        self.transcript_path = hook_data.get('transcript_path', '')
        
        # Load session transcript to get recent user messages
        self.transcript = self._load_transcript()
    
    def _load_transcript(self):
        """Load the session transcript"""
        try:
            if self.transcript_path and os.path.exists(self.transcript_path):
                with open(self.transcript_path, 'r') as f:
                    lines = f.readlines()
                    # Parse JSONL format
                    messages = []
                    for line in lines:
                        line = line.strip()
                        if line:
                            try:
                                msg = json.loads(line)
                                messages.append(msg)
                            except:
                                continue
                    return {'messages': messages}
            return {'messages': []}
        except Exception as e:
            logger.warning(f"Could not load transcript: {e}")
            return {'messages': []}
    
    def _get_latest_user_message(self):
        """Get the most recent user message from transcript"""
        messages = self.transcript.get('messages', [])
        
        # Look for the most recent user message
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                return msg.get('content', '')
        
        return ""
    
    def _is_permission_prompt(self):
        """Check if the current prompt is asking for permission"""
        prompt_lower = self.prompt_text.lower()
        
        permission_indicators = [
            'do you want to proceed',
            'continue with this action',
            'approve this tool call',
            'allow this operation',
            'would you like to continue',
            'proceed with',
            'execute this',
            'run this command',
            'make this change',
            'confirm this action'
        ]
        
        return any(indicator in prompt_lower for indicator in permission_indicators)
    
    def _should_auto_approve(self):
        """Check if auto-approval should be enabled for current prompt"""
        # Get the latest user message (the one that triggered this interaction)
        latest_user_message = self._get_latest_user_message()
        
        if not latest_user_message:
            return False, "No user message found"
        
        # Check if "auto approve" is present in the current user prompt
        user_content_lower = latest_user_message.lower()
        
        if "auto approve" in user_content_lower:
            return True, f"Auto-approval activated by user prompt: '{latest_user_message[:100]}...'"
        else:
            return False, "Auto-approval not requested (default: manual approval required)"
    
    def process_prompt(self):
        """Main auto-approval logic"""
        try:
            # Check if this is a permission prompt
            if not self._is_permission_prompt():
                # Not a permission prompt, pass through normally
                response = {"response": None}
                print(json.dumps(response))
                return
            
            logger.info(f"Permission prompt detected: {self.prompt_text[:100]}...")
            
            # Check if auto-approval should be enabled
            should_approve, reason = self._should_auto_approve()
            
            if should_approve:
                # Auto-approve the permission
                response = {
                    "response": "y",
                    "reason": f"Auto-approved: {reason}"
                }
                logger.info(f"AUTO-APPROVED: {reason}")
            else:
                # Pass through for manual approval
                response = {"response": None}
                logger.info(f"MANUAL APPROVAL REQUIRED: {reason}")
            
            print(json.dumps(response))
            
        except Exception as e:
            logger.error(f"Auto-approval processing error: {e}")
            # Default to no interference on error
            response = {"response": None}
            print(json.dumps(response))

def main():
    """Main hook execution function"""
    try:
        # Read hook data from stdin
        input_data = sys.stdin.read().strip()
        if not input_data:
            logger.error("No input data received")
            return
        
        hook_data = json.loads(input_data)
        
        # Create and run auto-approver
        approver = AutoApprover(hook_data)
        approver.process_prompt()
        
    except Exception as e:
        logger.error(f"Hook execution failed: {e}")
        # Default to no interference to avoid breaking normal operation
        response = {"response": None}
        print(json.dumps(response))

if __name__ == "__main__":
    main()