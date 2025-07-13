# Auto-Approval Hook Documentation

## Overview
The auto-approval hook automatically responds "y" to Claude Code permission prompts when explicitly requested by the user.

## Safety Design
- **Default**: Manual approval required for all prompts
- **Activation**: Only when user includes "auto approve" in their prompt
- **No persistence**: Each prompt is evaluated independently
- **No session state**: Cannot be "left on" accidentally

## Usage

### Enable Auto-Approval (Per-Prompt)
Include "auto approve" anywhere in your prompt:
```bash
claude
> "auto approve - implement the dog detection system"
# All permissions auto-approved for this task

> "now add email alerts"
# Back to manual approval (auto approve not mentioned)
```

### Manual Approval (Default)
Simply don't include "auto approve":
```bash
claude  
> "implement boundary detection"
# Normal permission prompts appear
```

## Examples

### Long Running Tasks
```bash
> "auto approve - install YOLOv8, integrate with Flask, add detection logic, test everything, and commit when working"
```

### Mixed Workflow
```bash
> "auto approve - add basic dog detection"     # Auto-approved
> "let me review the detection accuracy"       # Manual approval
> "auto approve - optimize and deploy"         # Auto-approved again
```

## Safety Features

1. **Explicit Activation**: Must include "auto approve" in each prompt
2. **Permission Detection**: Only responds to actual permission prompts
3. **Logging**: All decisions logged to `/mnt/c/yard/scripts/auto_approve.log`
4. **Error Handling**: Defaults to manual approval on any errors
5. **Scope Enforcement**: Still protected by scope enforcer hook

## Technical Details

- **Hook Type**: PrePrompt
- **Trigger**: All prompts (`*` matcher)
- **Script**: `/mnt/c/yard/scripts/auto_approve.py`
- **Timeout**: 5 seconds
- **Log File**: `/mnt/c/yard/scripts/auto_approve.log`

## Integration with Other Hooks

Works alongside:
- **Scope Enforcer**: Still blocks off-rails actions
- **Memory Hook**: Still preserves context before compacts

## Troubleshooting

### Auto-Approval Not Working
1. Check if "auto approve" is in your prompt
2. Verify hook is configured: Check `/root/.claude/settings.json`
3. Check logs: `tail -f /mnt/c/yard/scripts/auto_approve.log`

### Unexpected Auto-Approvals
- Hook only activates when "auto approve" is explicitly in prompt
- Check recent user messages in logs

### Disable Auto-Approval
- Simply don't include "auto approve" in your prompt
- No persistent state to disable

## Example Log Output
```
2025-07-08 10:30:15 - INFO - Permission prompt detected: Do you want to proceed with this action...
2025-07-08 10:30:15 - INFO - AUTO-APPROVED: Auto-approval activated by user prompt: 'auto approve - implement detection...'
```