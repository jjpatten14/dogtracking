# Boundary System Issues Documentation

## Overview
This document tracks the boundary coordinate mismatch issue discovered on 2025-07-09 and the debugging/fixes implemented.

## Issue Description
When loading saved boundaries, they appeared with visual distortion and coordinate mismatches. The boundaries didn't align properly between client canvas overlay and server video stream rendering.

## Root Cause Analysis

### The Problem: Memory vs File Boundary Conflict
The coordinate mismatch was caused by boundaries existing in **BOTH** memory and file simultaneously:

- **Memory boundaries** (`boundary_list`): Used directly by server with original pixel coordinates
- **File boundaries** (`boundary_config.json`): Loaded by client with coordinate conversion
- **Result**: Double-drawing with mismatched coordinate systems

### Technical Flow That Caused Issues:

1. **During Save**:
   - User draws boundary → Client has normalized coordinates (0-1)
   - Client gets video dimensions → Converts to pixel coordinates  
   - Client sends pixel coordinates to server → Server saves to `boundary_list` (memory)
   - User clicks "Save All to File" → Server saves `boundary_list` to `boundary_config.json`
   - **Problem**: Boundaries now exist in BOTH memory and file

2. **During Load**:
   - Server renders boundaries from memory (`boundary_list`) using original pixel coordinates
   - Client loads boundaries from file → Converts pixel coordinates back to normalized
   - Client renders boundaries on canvas using converted coordinates
   - **Result**: Two different coordinate sets = visual mismatch

### Why Normal Drawing Worked:
- During normal boundary creation, coordinates flow: Client → Server memory → Server rendering
- No file conversion involved, so no coordinate conversion errors
- Single rendering path = consistent coordinates

### Why Server Restart "Fixed" It:
- Server restart cleared in-memory `boundary_list`
- Only file boundaries remained (single coordinate system)
- No double-drawing = no visual mismatch

## Debug Logging Added

### Server-Side (`app.py`):
- **Removed**: Per-frame boundary drawing spam (30 FPS logging)
- **Added**: Detailed save boundary coordinate logging in `/save_boundary` endpoint
- **Logs**: Video dimensions, pixel coordinates, frame bounds validation

### Client-Side (`boundary-drawing.js`):
- **Added**: Detailed load boundary coordinate logging in `loadSavedBoundaries()`
- **Logs**: Video dimensions, canvas dimensions, coordinate conversion steps
- **Traces**: Pixel → normalized → canvas coordinate flow

## Current Status

### Fixed:
✅ **Logging spam removed** - No more per-frame debug output  
✅ **Coordinate debugging added** - Detailed logging for save/load operations  
✅ **Issue temporarily resolved** - Server restart cleared memory conflict  

### Still Needs Fixing:
⚠️ **Root cause remains** - Memory vs file boundary conflict not permanently fixed  
⚠️ **Will reoccur** - Issue will happen again when boundaries exist in both memory and file  

## Proposed Permanent Fix

### Strategy: Clear Memory After Save-to-File
1. **Modify "Save All to File" button**:
   - After successful file save, clear in-memory `boundary_list`
   - Make file the single source of truth for persistent boundaries
   - Prevent memory vs file conflicts

2. **Update coordinate flow**:
   - Memory: Only for temporary/active drawing
   - File: Only for persistent/saved boundaries
   - No simultaneous existence in both

### Files to Modify:
- `boundary-drawing.js`: Update `saveBoundary()` to clear server memory after file save
- `app.py`: Add endpoint to clear memory boundaries after file save

## Test Cases for Tomorrow

1. **Test coordinate consistency**:
   - Draw boundary → Save to file → Load from file
   - Verify client and server boundaries align perfectly

2. **Test save/load cycle**:
   - Create multiple boundaries → Save all → Restart server → Load
   - Ensure no coordinate drift or visual mismatches

3. **Test memory clearing**:
   - Save boundaries → Verify memory cleared → Load → Verify single rendering

## Technical Notes

### Video Dimensions:
- Current camera: 2304×1296 (actual frame size)
- Canvas: Variable size based on display
- Coordinate conversion: Normalized (0-1) ↔ Pixel coordinates

### Coordinate Conversion Chain:
```
User Click → Canvas Pixels → Normalized (0-1) → Server Pixels → File Storage
File Storage → Server Pixels → Normalized (0-1) → Canvas Pixels → Display
```

### Key Learning:
The double-drawing was **intentional for debugging** - user wanted to see both client and server boundaries to verify alignment. The issue wasn't the double-drawing itself, but the coordinate mismatch between them.

## Log Examples

### Successful Save (2025-07-09):
```
💾 ===== SAVE BOUNDARY COORDINATE DEBUG =====
📺 Server video dimensions during save: 2304x1296
📥 Received 4 boundary points from client:
   Point 1: pixel(1685, 308)
   Point 2: pixel(955, 315)
   Point 3: pixel(1031, 770)
   Point 4: pixel(2097, 704)
✅ Boundary saved to server (total: 1)
```

### System Status:
- Camera resolution: 2304×1296 (no forced resize for better performance)
- Dynamic coordinate system: ✅ Working
- Boundary persistence: ⚠️ Needs memory/file conflict fix