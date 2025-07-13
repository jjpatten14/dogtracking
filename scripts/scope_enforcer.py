#!/usr/bin/env python3
"""
Claude Code Scope Enforcement Hook

Prevents Claude from going off-rails and adding unnecessary features/complexity.
Includes disable mechanism via "no rails" keywords.
"""

import json
import sys
import os
import re
from pathlib import Path
import logging
from datetime import datetime

# Add parent directory to path for project_paths import
sys.path.append(str(Path(__file__).parent.parent))
from project_paths import get_path

# Set up logging
log_file = get_path('scripts') / "scope_enforcer.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

class ScopeEnforcer:
    """Analyzes tool calls to prevent scope creep and off-rail behavior"""
    
    def __init__(self, hook_data):
        self.hook_data = hook_data
        self.session_id = hook_data.get('session_id', 'unknown')
        self.tool_name = hook_data.get('tool_name', '')
        self.tool_input = hook_data.get('tool_input', {})
        self.transcript_path = hook_data.get('transcript_path', '')
        
        # Load session transcript
        self.transcript = self._load_transcript()
        
        # Disable keywords that bypass scope checking
        self.disable_keywords = [
            'no rails', 'off rails', 'free form', 'no scope',
            'norails', 'offrails', 'freeform', 'noscope'
        ]
        
        # High-risk scope creep indicators
        self.scope_creep_patterns = [
            'while we\'re at it', 'also add', 'let\'s also', 'might as well',
            'we could also', 'additionally', 'furthermore', 'moreover',
            'i\'ll also', 'let me also', 'i\'ll add', 'let me add',
            'improve', 'enhance', 'optimize', 'refactor'
        ]
        
        # File patterns that are often unnecessary
        self.unnecessary_file_patterns = [
            r'.*\.md$', r'.*README.*', r'.*CHANGELOG.*', r'.*LICENSE.*',
            r'.*docs?/.*', r'.*documentation.*', r'.*guide.*'
        ]
    
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
    
    def _check_disable_keywords(self):
        """Check if user has disabled scope enforcement"""
        recent_messages = self.transcript.get('messages', [])[-10:]  # Last 10 messages
        
        for msg in recent_messages:
            if msg.get('role') == 'user':
                content = msg.get('content', '').lower()
                for keyword in self.disable_keywords:
                    if keyword in content:
                        logger.info(f"Scope enforcement disabled by keyword: '{keyword}'")
                        return True, keyword
        
        return False, None
    
    def _is_plan_mode(self):
        """Check if current session is in plan mode"""
        recent_messages = self.transcript.get('messages', [])[-10:]  # Last 10 messages
        
        plan_mode_indicators = [
            'plan mode', 'planning mode', 'we are in plan mode',
            'pokease dont ask to exit', 'dont exit plan', 'in polan mode',
            'we are in polan mode', 'plan mode is active'
        ]
        
        for msg in recent_messages:
            if msg.get('role') == 'user':
                content = msg.get('content', '').lower()
                for indicator in plan_mode_indicators:
                    if indicator in content:
                        logger.info(f"Plan mode detected: '{indicator}'")
                        return True
        
        return False
    
    def _is_readonly_tool(self):
        """Check if tool is read-only (information gathering)"""
        readonly_tools = [
            'Read', 'LS', 'Glob', 'Grep', 'WebSearch', 'WebFetch',
            'TodoRead', 'NotebookRead', 'mcp__memory__read_graph',
            'mcp__memory__search_nodes', 'mcp__memory__open_nodes'
        ]
        
        return self.tool_name in readonly_tools
    
    def _extract_original_task(self):
        """Extract the original user request/task from transcript"""
        messages = self.transcript.get('messages', [])
        
        # Look for the first substantial user message
        for msg in messages:
            if msg.get('role') == 'user':
                content = msg.get('content', '').strip()
                if len(content) > 20:  # Substantial message
                    return content
        
        return "No clear task identified"
    
    def _is_documentation_file(self, file_path):
        """Check if file is documentation-related"""
        if not file_path:
            return False
            
        file_path = str(file_path).lower()
        
        for pattern in self.unnecessary_file_patterns:
            if re.match(pattern, file_path):
                return True
        
        return False
    
    def _analyze_action_relevance(self):
        """Analyze if the current action is relevant to the original task"""
        original_task = self._extract_original_task()
        tool_name = self.tool_name
        tool_input = self.tool_input
        
        # Check for documentation file creation
        if tool_name in ['Write', 'Edit', 'MultiEdit']:
            file_path = tool_input.get('file_path', '')
            if self._is_documentation_file(file_path):
                return False, f"Creating documentation file '{file_path}' not requested in original task"
        
        # Check for excessive new file creation
        if tool_name == 'Write':
            file_path = tool_input.get('file_path', '')
            content = tool_input.get('content', '')
            
            # Large new files might be scope creep
            if len(content) > 5000:  # Arbitrary threshold
                return False, f"Creating large new file '{file_path}' may be beyond original scope"
        
        # Check for scope creep language in recent assistant messages
        recent_messages = self.transcript.get('messages', [])[-5:]
        for msg in recent_messages:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '').lower()
                for pattern in self.scope_creep_patterns:
                    if pattern in content:
                        return False, f"Detected scope creep language: '{pattern}'"
        
        # Check for unrelated file modifications
        if tool_name in ['Edit', 'MultiEdit']:
            file_path = tool_input.get('file_path', '')
            original_task_lower = original_task.lower()
            
            # If original task mentions specific files/areas, check relevance
            if 'web' in original_task_lower and 'app.py' in original_task_lower:
                if file_path and 'app.py' not in file_path and 'web' not in file_path:
                    return False, f"Modifying '{file_path}' not related to web/app.py task"
        
        return True, "Action appears relevant to original task"
    
    def _should_block_action(self):
        """Determine if the action should be blocked"""
        tool_name = self.tool_name
        tool_input = self.tool_input
        
        # Always allow Read operations (gathering information is OK)
        if tool_name in ['Read', 'LS', 'Glob', 'Grep']:
            return False, "Information gathering allowed"
        
        # Check action relevance
        is_relevant, reason = self._analyze_action_relevance()
        if not is_relevant:
            return True, reason
        
        # Check for potentially destructive operations
        if tool_name == 'Bash':
            command = tool_input.get('command', '')
            if any(dangerous in command for dangerous in ['rm -rf', 'rm -r', 'sudo', 'format']):
                return True, f"Potentially dangerous command: {command}"
        
        return False, "Action approved"
    
    def enforce_scope(self):
        """Main scope enforcement logic"""
        try:
            # Quick check for plan mode or read-only tools - skip processing
            if self._is_plan_mode():
                response = {
                    "decision": "approve",
                    "reason": "Plan mode active - scope enforcement skipped for performance"
                }
                print(json.dumps(response))
                return
            
            if self._is_readonly_tool():
                response = {
                    "decision": "approve", 
                    "reason": "Read-only tool - information gathering allowed"
                }
                print(json.dumps(response))
                return
            
            logger.info(f"Checking scope for {self.tool_name} tool")
            
            # Check if user disabled scope enforcement
            is_disabled, keyword = self._check_disable_keywords()
            if is_disabled:
                response = {
                    "decision": "approve",
                    "reason": f"Scope enforcement disabled by user keyword: '{keyword}'"
                }
                print(json.dumps(response))
                return
            
            # Analyze if action should be blocked
            should_block, reason = self._should_block_action()
            
            if should_block:
                response = {
                    "decision": "block",
                    "reason": f"Scope enforcement: {reason}. Add 'no rails' to your message to disable scope checking."
                }
                logger.info(f"BLOCKED: {self.tool_name} - {reason}")
            else:
                response = {
                    "decision": "approve",
                    "reason": f"Action approved: {reason}"
                }
                logger.info(f"APPROVED: {self.tool_name} - {reason}")
            
            print(json.dumps(response))
            
        except Exception as e:
            logger.error(f"Scope enforcement error: {e}")
            # Default to approve on error to avoid blocking legitimate actions
            response = {
                "decision": "approve", 
                "reason": f"Scope enforcement error, defaulting to approve: {e}"
            }
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
        
        # Create and run scope enforcer
        enforcer = ScopeEnforcer(hook_data)
        enforcer.enforce_scope()
        
    except Exception as e:
        logger.error(f"Hook execution failed: {e}")
        # Default to approve to avoid breaking legitimate operations
        response = {"decision": "approve", "reason": f"Hook error: {e}"}
        print(json.dumps(response))

if __name__ == "__main__":
    main()