#!/usr/bin/env python3
"""
Pre-Compact Memory Update Hook

Automatically extracts meaningful context from Claude Code session before compacting
and updates the MCP memory database to preserve problem-solution context.
"""

import json
import sys
import os
import subprocess
import re
from datetime import datetime
from pathlib import Path
import logging

# Set up logging
log_file = Path(__file__).parent / "hook.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

class ContextExtractor:
    """Extracts meaningful context from Claude Code session transcript"""
    
    def __init__(self, transcript_data):
        self.transcript = transcript_data
        self.extracted_data = {
            'problems': [],
            'solutions': [],
            'decisions': [],
            'code_changes': [],
            'performance_improvements': []
        }
    
    def extract_problems(self, messages):
        """Extract problems mentioned in the session"""
        problem_patterns = [
            r'(?i)(error|issue|problem|bug|failing|broken).*?(?:\.|$)',
            r'(?i)(not working|doesn\'t work|fails to).*?(?:\.|$)',
            r'(?i)(slow|performance|timeout).*?(?:\.|$)'
        ]
        
        problems = []
        for msg in messages:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                for pattern in problem_patterns:
                    matches = re.findall(pattern, content)
                    problems.extend(matches)
        
        return problems[:5]  # Limit to 5 most recent problems
    
    def extract_solutions(self, messages):
        """Extract solutions and fixes from the session"""
        solution_indicators = [
            'fixed', 'solved', 'resolved', 'implemented', 'updated',
            'changed', 'modified', 'added', 'created', 'improved'
        ]
        
        solutions = []
        for msg in messages:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                for indicator in solution_indicators:
                    if indicator in content.lower():
                        # Extract sentence containing the solution
                        sentences = content.split('.')
                        for sentence in sentences:
                            if indicator in sentence.lower():
                                solutions.append(sentence.strip())
                                break
        
        return solutions[:5]  # Limit to 5 most recent solutions
    
    def extract_code_changes(self, messages):
        """Extract significant code changes from file edits"""
        code_changes = []
        
        for msg in messages:
            if msg.get('role') == 'assistant':
                # Look for tool calls that modify files
                if 'tool_calls' in msg:
                    for tool_call in msg['tool_calls']:
                        if tool_call.get('function', {}).get('name') in ['Edit', 'Write', 'MultiEdit']:
                            args = tool_call.get('function', {}).get('arguments', {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except:
                                    continue
                            
                            file_path = args.get('file_path', '')
                            if file_path:
                                code_changes.append(f"Modified {file_path}")
        
        return code_changes[:10]  # Limit to 10 most recent changes
    
    def extract_decisions(self, messages):
        """Extract technical decisions and rationale"""
        decision_patterns = [
            r'(?i)(decided to|chose to|using|implementing).*?(?:\.|$)',
            r'(?i)(because|since|due to|reason).*?(?:\.|$)',
            r'(?i)(approach|strategy|method|technique).*?(?:\.|$)'
        ]
        
        decisions = []
        for msg in messages:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                for pattern in decision_patterns:
                    matches = re.findall(pattern, content)
                    decisions.extend(matches)
        
        return decisions[:5]  # Limit to 5 most recent decisions
    
    def extract_all(self):
        """Extract all meaningful context from transcript"""
        try:
            messages = self.transcript.get('messages', [])
            
            self.extracted_data['problems'] = self.extract_problems(messages)
            self.extracted_data['solutions'] = self.extract_solutions(messages)
            self.extracted_data['code_changes'] = self.extract_code_changes(messages)
            self.extracted_data['decisions'] = self.extract_decisions(messages)
            
            logger.info(f"Extracted context: {len(self.extracted_data['problems'])} problems, "
                       f"{len(self.extracted_data['solutions'])} solutions, "
                       f"{len(self.extracted_data['code_changes'])} code changes")
            
            return self.extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting context: {e}")
            return self.extracted_data

class MCPClient:
    """Client for interacting with MCP memory server"""
    
    def __init__(self, memory_server_path=None):
        if memory_server_path is None:
            memory_server_path = str(Path(__file__).parent.parent / "mcp-memory/dist/index.js")
        self.server_path = memory_server_path
    
    def create_session_entities(self, extracted_data, session_id):
        """Create entities in MCP memory for the session context"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_entity_name = f"Session_{session_id[:8]}_{timestamp}"
            
            entities_to_create = []
            relations_to_create = []
            
            # Create session entity
            entities_to_create.append({
                "name": session_entity_name,
                "entityType": "CodeSession",
                "observations": [
                    f"Session ID: {session_id}",
                    f"Timestamp: {timestamp}",
                    f"Auto-saved before compact operation",
                    f"Problems addressed: {len(extracted_data['problems'])}",
                    f"Solutions implemented: {len(extracted_data['solutions'])}",
                    f"Code changes made: {len(extracted_data['code_changes'])}"
                ]
            })
            
            # Create problem entities
            for i, problem in enumerate(extracted_data['problems']):
                if problem.strip():
                    problem_name = f"Problem_{session_id[:8]}_{i+1}"
                    entities_to_create.append({
                        "name": problem_name,
                        "entityType": "Problem",
                        "observations": [problem.strip(), f"From session {session_id[:8]}"]
                    })
                    relations_to_create.append({
                        "from": session_entity_name,
                        "to": problem_name,
                        "relationType": "encountered"
                    })
            
            # Create solution entities
            for i, solution in enumerate(extracted_data['solutions']):
                if solution.strip():
                    solution_name = f"Solution_{session_id[:8]}_{i+1}"
                    entities_to_create.append({
                        "name": solution_name,
                        "entityType": "Solution",
                        "observations": [solution.strip(), f"From session {session_id[:8]}"]
                    })
                    relations_to_create.append({
                        "from": session_entity_name,
                        "to": solution_name,
                        "relationType": "implemented"
                    })
            
            # Create code change entities
            for i, change in enumerate(extracted_data['code_changes']):
                if change.strip():
                    change_name = f"CodeChange_{session_id[:8]}_{i+1}"
                    entities_to_create.append({
                        "name": change_name,
                        "entityType": "CodeChange",
                        "observations": [change.strip(), f"From session {session_id[:8]}"]
                    })
                    relations_to_create.append({
                        "from": session_entity_name,
                        "to": change_name,
                        "relationType": "made_change"
                    })
            
            # Write to MCP memory using direct file append (simpler than subprocess)
            memory_file = Path(__file__).parent.parent / "mcp-memory/dist/memory.json"
            
            if entities_to_create:
                with memory_file.open("a") as f:
                    for entity in entities_to_create:
                        entity_record = {"type": "entity", **entity}
                        f.write(json.dumps(entity_record) + "\n")
                    
                    for relation in relations_to_create:
                        relation_record = {"type": "relation", **relation}
                        f.write(json.dumps(relation_record) + "\n")
                
                logger.info(f"Successfully saved {len(entities_to_create)} entities and "
                           f"{len(relations_to_create)} relations to MCP memory")
                return True
            
        except Exception as e:
            logger.error(f"Error updating MCP memory: {e}")
            return False
    
def main():
    """Main hook execution function"""
    try:
        # Read hook data from stdin
        hook_data = json.loads(sys.stdin.read())
        
        session_id = hook_data.get('session_id', 'unknown')
        transcript_path = hook_data.get('transcript_path', '')
        
        logger.info(f"Pre-compact hook triggered for session {session_id}")
        
        # Load transcript data
        if transcript_path and os.path.exists(transcript_path):
            with open(transcript_path, 'r') as f:
                transcript_data = json.load(f)
        else:
            logger.warning("No transcript path provided or file not found")
            return
        
        # Extract meaningful context
        extractor = ContextExtractor(transcript_data)
        extracted_data = extractor.extract_all()
        
        # Check if we have anything meaningful to save
        total_items = (len(extracted_data['problems']) + 
                      len(extracted_data['solutions']) + 
                      len(extracted_data['code_changes']))
        
        if total_items == 0:
            logger.info("No meaningful context found to preserve")
            return
        
        # Update MCP memory
        mcp_client = MCPClient()
        success = mcp_client.create_session_entities(extracted_data, session_id)
        
        if success:
            logger.info("Successfully preserved session context before compact")
        else:
            logger.error("Failed to preserve session context")
            
    except Exception as e:
        logger.error(f"Hook execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()