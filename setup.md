# Dog Tracking System - Web Interface Setup Documentation

## Project Overview
This is a camera-based dog tracking system that uses AI visual tracking with configurable boundaries. The system allows users to draw boundaries on camera feeds and will eventually trigger actions when dogs cross those boundaries.

## What Was Built (Pre-MCP Setup)

### 1. Project Structure Created
```
/mnt/c/yard/
├── setup.bat                 # Windows installation script (verbose mode)
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html           # Main web interface
├── static/
│   ├── css/
│   │   └── style.css        # All styling (IMPORTANT: Layout defined here)
│   ├── js/
│   │   └── boundary-drawing.js  # Interactive drawing functionality
│   └── images/              # Static images directory
└── mcp-memory/              # MCP memory server (added later)
```

### 2. Installation Script (setup.bat)
**Purpose**: Automated Python environment setup with verbose output for monitoring
**Key Features**:
- Creates Python virtual environment (`venv`)
- Installs all dependencies with `--verbose` flag
- Creates project directory structure
- Shows detailed progress for each step

**Dependencies Installed**:
- Flask 2.3.3 (web framework)
- OpenCV 4.8.1.78 (camera processing)
- NumPy 1.24.3 (array operations)
- Pillow 10.0.0 (image processing)
- Flask-SocketIO 5.3.4 (real-time communication)
- Eventlet 0.33.3 (async support)

### 3. Flask Application (app.py)
**Core Functionality**:
- **Camera Streaming**: Real-time video feed via `/video_feed` endpoint
- **Boundary Management**: Save/load/clear boundary configurations
- **API Endpoints**: RESTful API for boundary operations
- **Threading**: Separate thread for camera capture (~30 fps)
- **Error Handling**: Graceful fallback when no camera detected

**Important Classes**:
- `CameraStream`: Handles video capture and threading
- **Global Variables**: `boundaries[]`, `current_boundary[]`, `drawing` state

**API Endpoints**:
- `GET /` - Main interface
- `GET /video_feed` - Live camera stream
- `POST /save_boundary` - Save drawn boundary
- `POST /clear_boundaries` - Clear all boundaries
- `POST /save_config` - Export to JSON file
- `POST /load_config` - Import from JSON file

### 4. Web Interface (templates/index.html)
**Layout Structure** (CRITICAL FOR EDITING):
```html
<div class="container">
  <header>                    <!-- Title and description -->
  <div class="main-content">  <!-- Grid: video + controls -->
    <div class="video-section">
      <div class="video-container">
        <img id="videoFeed">    <!-- Live camera feed -->
        <canvas id="drawingCanvas">  <!-- Overlay for drawing -->
      </div>
      <div class="video-controls">  <!-- Camera start/stop -->
    </div>
    <div class="controls-section">  <!-- All control panels -->
      <div class="drawing-controls">  <!-- Draw/finish/cancel buttons -->
      <div class="boundary-info">     <!-- Live stats display -->
      <div class="config-controls">   <!-- Save/load config -->
      <div class="instructions">      <!-- User guide -->
    </div>
  </div>
  <div class="status-section">    <!-- Fixed position notifications -->
</div>
```

**Key Elements**:
- `#videoFeed`: Displays live camera stream
- `#drawingCanvas`: Transparent overlay for boundary drawing
- `#statusMessages`: Fixed-position notification area

### 5. CSS Styling (static/css/style.css)
**⚠️ CRITICAL: ALL LAYOUT IS DEFINED IN CSS, NOT HTML**

**Layout System**:
- **Grid Layout**: `.main-content` uses CSS Grid (2fr 1fr)
- **Video Positioning**: `.video-wrapper` uses relative positioning
- **Canvas Overlay**: `#drawingCanvas` positioned absolutely over video
- **Responsive**: Grid collapses to single column on mobile

**Key CSS Classes**:
```css
.main-content {
  display: grid;
  grid-template-columns: 2fr 1fr;  /* Video takes 2/3, controls 1/3 */
}

#drawingCanvas {
  position: absolute;  /* CRITICAL: Overlays video */
  top: 0;
  left: 0;
  cursor: crosshair;
}

.video-wrapper {
  position: relative;  /* CRITICAL: Contains canvas overlay */
}

.status-section {
  position: fixed;     /* CRITICAL: Fixed notifications */
  top: 20px;
  right: 20px;
}
```

**Button Styling**:
- `.btn`: Base button class with hover effects
- `.btn-primary/.btn-success/.btn-danger`: Color variants
- `:disabled` state properly handled

### 6. JavaScript Functionality (static/js/boundary-drawing.js)
**Core Class**: `BoundaryDrawer`

**Key Methods**:
- `setupCanvas()`: Manages canvas sizing and video overlay
- `startDrawing()`: Initiates boundary creation mode
- `addPoint()`: Records click coordinates (normalized 0-1)
- `finishBoundary()`: Converts to pixels and saves via API
- `redrawBoundaries()`: Renders all boundaries on canvas

**Coordinate System**:
- **Normalized Coordinates**: Stored as 0-1 scale for resolution independence
- **Pixel Coordinates**: Converted for server storage (640x480 assumed)
- **Canvas Coordinates**: Real-time drawing uses actual canvas dimensions

**Event Handling**:
- **Mouse Click**: Adds boundary points
- **Mouse Move**: Shows preview line when drawing
- **Window Resize**: Automatically adjusts canvas overlay
- **Button Events**: All UI interactions

**State Management**:
- `isDrawing`: Boolean flag for drawing mode
- `currentBoundary[]`: Points being drawn
- `boundaries[]`: All completed boundaries

### 7. Configuration System
**File Storage**: `boundary_config.json`
**Format**:
```json
{
  "boundaries": [
    [[x1,y1], [x2,y2], [x3,y3], ...]
  ],
  "timestamp": 1704067200.123
}
```

## Development Notes for Future Context

### 1. CSS-Driven Layout
**⚠️ WARNING**: The layout is entirely CSS-driven. Changing HTML structure without updating CSS will break the interface. Key dependencies:
- Video overlay requires `position: relative` parent
- Canvas positioning is absolute within video wrapper
- Grid layout controls responsive behavior

### 2. Coordinate System
- **Drawing**: Uses normalized 0-1 coordinates for resolution independence
- **Storage**: Converts to pixels (640x480) for server
- **Display**: Real-time canvas uses actual element dimensions

### 3. Camera Threading
- Camera runs in separate thread to prevent UI blocking
- Frame updates at ~30fps independent of web requests
- Graceful degradation when no camera available

### 4. API Design
- RESTful endpoints for all operations
- JSON responses with consistent `{status, message}` format
- Error handling returns proper HTTP status codes

### 5. Browser Compatibility
- Uses modern JavaScript (ES6+ classes)
- Canvas API for drawing
- Fetch API for AJAX requests
- CSS Grid for layout

## Starting the Application
1. Run `setup.bat` (first time only)
2. Activate environment: `call venv\Scripts\activate.bat`
3. Start server: `python app.py`
4. Open browser: `http://localhost:5000`

## Future Development Areas
- Reference point tracking for camera stability
- AI/ML integration for dog detection
- Multiple camera support
- Action trigger system when boundaries are crossed