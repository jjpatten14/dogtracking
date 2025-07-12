"""
Alert and Notification System

Manages alerts for boundary violations and system events.
Supports email, sound, logging, and webhook notifications.

Key Features:
- Multiple alert channels (email, sound, logs, webhooks)
- Configurable alert rules and thresholds
- Alert escalation and de-duplication
- System health monitoring alerts
- Alert history and analytics
"""

import logging
import smtplib
import json
import time
import threading
import queue
import subprocess
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from enum import Enum
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertType(Enum):
    """Alert type enumeration"""
    BOUNDARY_VIOLATION = "boundary_violation"
    SYSTEM_ERROR = "system_error"
    CAMERA_OFFLINE = "camera_offline"
    MODEL_ERROR = "model_error"
    TRACKING_LOST = "tracking_lost"
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertChannel(Enum):
    """Alert delivery channels"""
    EMAIL = "email"
    SOUND = "sound"
    LOG = "log"
    WEBHOOK = "webhook"
    CONSOLE = "console"

@dataclass
class AlertConfig:
    """Alert configuration settings"""
    enabled: bool = True
    channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.LOG])
    min_severity: AlertSeverity = AlertSeverity.WARNING
    cooldown_seconds: int = 300  # 5 minutes default cooldown
    max_alerts_per_hour: int = 10
    email_settings: Dict[str, Any] = field(default_factory=dict)
    webhook_settings: Dict[str, Any] = field(default_factory=dict)
    sound_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Individual alert message"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    source: str = "dog_tracking_system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    channels_sent: List[AlertChannel] = field(default_factory=list)
    delivery_status: Dict[str, str] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'metadata': self.metadata,
            'channels_sent': [c.value for c in self.channels_sent],
            'delivery_status': self.delivery_status,
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_time': self.acknowledged_time.isoformat() if self.acknowledged_time else None
        }

class EmailNotifier:
    """Email notification handler"""
    
    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get('enabled', False)
        self.smtp_server = config.get('smtp_server', '')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username', '')
        self.password = config.get('password', '')
        self.from_email = config.get('from_email', '')
        self.to_emails = config.get('to_emails', [])
        self.use_tls = config.get('use_tls', True)
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via email"""
        if not self.enabled or not self.to_emails:
            return False
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()
            
            if self.username and self.password:
                server.login(self.username, self.password)
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body"""
        severity_color = {
            AlertSeverity.INFO: "#2196F3",
            AlertSeverity.WARNING: "#FF9800",
            AlertSeverity.ERROR: "#F44336",
            AlertSeverity.CRITICAL: "#9C27B0"
        }.get(alert.severity, "#666666")
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <div style="border-left: 4px solid {severity_color}; padding-left: 20px;">
                <h2 style="color: {severity_color}; margin-top: 0;">
                    {alert.title}
                </h2>
                <p><strong>Severity:</strong> {alert.severity.value.title()}</p>
                <p><strong>Type:</strong> {alert.alert_type.value.replace('_', ' ').title()}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Source:</strong> {alert.source}</p>
                
                <h3>Message:</h3>
                <p>{alert.message}</p>
                
                {self._format_metadata(alert.metadata)}
            </div>
            
            <hr style="margin: 30px 0;">
            <p style="color: #666; font-size: 12px;">
                Dog Tracking System Alert - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </body>
        </html>
        """
        return html
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for email display"""
        if not metadata:
            return ""
        
        html = "<h3>Additional Information:</h3><ul>"
        for key, value in metadata.items():
            html += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value}</li>"
        html += "</ul>"
        return html

class SoundNotifier:
    """Sound notification handler"""
    
    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get('enabled', False)
        self.sound_file = config.get('sound_file', '/usr/share/sounds/alsa/Front_Left.wav')
        self.volume = config.get('volume', 80)  # Percentage
        self.repeat_count = config.get('repeat_count', 1)
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via sound"""
        if not self.enabled:
            return False
        
        try:
            # Use different sounds for different severities
            sound_file = self._get_sound_file(alert.severity)
            
            # Play sound using system command
            for _ in range(self.repeat_count):
                if Path(sound_file).exists():
                    subprocess.run(['aplay', sound_file], check=False, capture_output=True)
                else:
                    # Fallback to system beep
                    subprocess.run(['beep', '-f', '800', '-l', '200'], check=False, capture_output=True)
                
                if self.repeat_count > 1:
                    time.sleep(0.5)
            
            logger.info(f"Sound alert played: {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to play sound alert: {e}")
            return False
    
    def _get_sound_file(self, severity: AlertSeverity) -> str:
        """Get sound file based on severity"""
        severity_sounds = {
            AlertSeverity.INFO: '/usr/share/sounds/alsa/Front_Left.wav',
            AlertSeverity.WARNING: '/usr/share/sounds/alsa/Front_Right.wav',
            AlertSeverity.ERROR: '/usr/share/sounds/alsa/Rear_Left.wav',
            AlertSeverity.CRITICAL: '/usr/share/sounds/alsa/Rear_Right.wav'
        }
        return severity_sounds.get(severity, self.sound_file)

class WebhookNotifier:
    """Webhook notification handler"""
    
    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get('enabled', False)
        self.webhook_url = config.get('webhook_url', '')
        self.headers = config.get('headers', {'Content-Type': 'application/json'})
        self.timeout = config.get('timeout', 10)
        self.retry_count = config.get('retry_count', 3)
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook"""
        if not self.enabled or not self.webhook_url:
            return False
        
        payload = {
            'alert': alert.to_dict(),
            'system': 'dog_tracking_system',
            'timestamp': datetime.now().isoformat()
        }
        
        for attempt in range(self.retry_count + 1):
            try:
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    logger.info(f"Webhook alert sent: {alert.alert_id}")
                    return True
                else:
                    logger.warning(f"Webhook returned status {response.status_code}: {response.text}")
                
            except Exception as e:
                logger.error(f"Webhook attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_count:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return False

class AlertSystem:
    """Main alert management system"""
    
    def __init__(self, config_path: str = '/mnt/c/yard/alert_config.json'):
        self.config_path = config_path
        self.config = AlertConfig()
        self.alerts_history: List[Alert] = []
        self.alert_counts: Dict[str, int] = {}  # For rate limiting
        self.last_alert_times: Dict[str, datetime] = {}  # For cooldown
        self.alert_queue = queue.Queue()
        
        # Notification handlers
        self.email_notifier: Optional[EmailNotifier] = None
        self.sound_notifier: Optional[SoundNotifier] = None
        self.webhook_notifier: Optional[WebhookNotifier] = None
        
        # Statistics
        self.stats = {
            'total_alerts_generated': 0,
            'alerts_sent': 0,
            'alerts_failed': 0,
            'alerts_suppressed': 0,
            'alerts_by_type': {t.value: 0 for t in AlertType},
            'alerts_by_severity': {s.value: 0 for s in AlertSeverity},
            'alerts_by_channel': {c.value: 0 for c in AlertChannel}
        }
        
        self.lock = threading.Lock()
        
        # Load configuration
        self.load_config()
        
        # Start alert processing thread
        self.processing_thread = threading.Thread(target=self._process_alerts_loop, daemon=True)
        self.processing_thread.start()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("Alert system initialized")
    
    def load_config(self):
        """Load alert configuration from file"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update config
                self.config.enabled = config_data.get('enabled', True)
                self.config.channels = [AlertChannel(c) for c in config_data.get('channels', ['log'])]
                self.config.min_severity = AlertSeverity(config_data.get('min_severity', 'warning'))
                self.config.cooldown_seconds = config_data.get('cooldown_seconds', 300)
                self.config.max_alerts_per_hour = config_data.get('max_alerts_per_hour', 10)
                self.config.email_settings = config_data.get('email_settings', {})
                self.config.webhook_settings = config_data.get('webhook_settings', {})
                self.config.sound_settings = config_data.get('sound_settings', {})
                
                # Initialize notifiers
                if AlertChannel.EMAIL in self.config.channels:
                    self.email_notifier = EmailNotifier(self.config.email_settings)
                
                if AlertChannel.SOUND in self.config.channels:
                    self.sound_notifier = SoundNotifier(self.config.sound_settings)
                
                if AlertChannel.WEBHOOK in self.config.channels:
                    self.webhook_notifier = WebhookNotifier(self.config.webhook_settings)
                
                logger.info("Alert configuration loaded")
            else:
                self.save_config()  # Create default config
                
        except Exception as e:
            logger.error(f"Failed to load alert configuration: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config_data = {
                'enabled': self.config.enabled,
                'channels': [c.value for c in self.config.channels],
                'min_severity': self.config.min_severity.value,
                'cooldown_seconds': self.config.cooldown_seconds,
                'max_alerts_per_hour': self.config.max_alerts_per_hour,
                'email_settings': self.config.email_settings,
                'webhook_settings': self.config.webhook_settings,
                'sound_settings': self.config.sound_settings
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info("Alert configuration saved")
            
        except Exception as e:
            logger.error(f"Failed to save alert configuration: {e}")
    
    def send_alert(self, alert_type: AlertType, severity: AlertSeverity,
                  title: str, message: str, metadata: Dict[str, Any] = None) -> Optional[str]:
        """
        Send alert through configured channels
        
        Args:
            alert_type: Type of alert
            severity: Alert severity
            title: Alert title
            message: Alert message
            metadata: Additional metadata
            
        Returns:
            Alert ID if sent, None if suppressed
        """
        if not self.config.enabled:
            return None
        
        # Check severity filter
        severity_levels = [AlertSeverity.INFO, AlertSeverity.WARNING, AlertSeverity.ERROR, AlertSeverity.CRITICAL]
        if severity_levels.index(severity) < severity_levels.index(self.config.min_severity):
            return None
        
        # Create alert
        alert_id = f"alert_{int(datetime.now().timestamp())}_{alert_type.value}"
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # Check rate limiting and cooldown
        if self._should_suppress_alert(alert):
            with self.lock:
                self.stats['alerts_suppressed'] += 1
            return None
        
        # Queue alert for processing
        self.alert_queue.put(alert)
        
        with self.lock:
            self.stats['total_alerts_generated'] += 1
            self.stats['alerts_by_type'][alert_type.value] += 1
            self.stats['alerts_by_severity'][severity.value] += 1
        
        return alert_id
    
    def _should_suppress_alert(self, alert: Alert) -> bool:
        """Check if alert should be suppressed due to rate limiting or cooldown"""
        alert_key = f"{alert.alert_type.value}_{alert.severity.value}"
        current_time = datetime.now()
        
        # Check cooldown
        if alert_key in self.last_alert_times:
            time_since_last = (current_time - self.last_alert_times[alert_key]).total_seconds()
            if time_since_last < self.config.cooldown_seconds:
                return True
        
        # Check hourly rate limit
        hour_key = f"{alert_key}_{current_time.hour}"
        if hour_key not in self.alert_counts:
            self.alert_counts[hour_key] = 0
        
        if self.alert_counts[hour_key] >= self.config.max_alerts_per_hour:
            return True
        
        # Update counters
        self.last_alert_times[alert_key] = current_time
        self.alert_counts[hour_key] += 1
        
        return False
    
    def _process_alerts_loop(self):
        """Background thread to process alert queue"""
        while True:
            try:
                alert = self.alert_queue.get(timeout=1)
                self._deliver_alert(alert)
                self.alert_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
    
    def _deliver_alert(self, alert: Alert):
        """Deliver alert through configured channels"""
        success_count = 0
        
        for channel in self.config.channels:
            try:
                success = False
                
                if channel == AlertChannel.LOG:
                    self._send_log_alert(alert)
                    success = True
                    
                elif channel == AlertChannel.CONSOLE:
                    self._send_console_alert(alert)
                    success = True
                    
                elif channel == AlertChannel.EMAIL and self.email_notifier:
                    success = self.email_notifier.send_alert(alert)
                    
                elif channel == AlertChannel.SOUND and self.sound_notifier:
                    success = self.sound_notifier.send_alert(alert)
                    
                elif channel == AlertChannel.WEBHOOK and self.webhook_notifier:
                    success = self.webhook_notifier.send_alert(alert)
                
                if success:
                    alert.channels_sent.append(channel)
                    alert.delivery_status[channel.value] = "success"
                    success_count += 1
                else:
                    alert.delivery_status[channel.value] = "failed"
                
                with self.lock:
                    self.stats['alerts_by_channel'][channel.value] += 1
                
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")
                alert.delivery_status[channel.value] = f"error: {str(e)}"
        
        # Store alert in history
        with self.lock:
            self.alerts_history.append(alert)
            if success_count > 0:
                self.stats['alerts_sent'] += 1
            else:
                self.stats['alerts_failed'] += 1
    
    def _send_log_alert(self, alert: Alert):
        """Send alert to system log"""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(alert.severity, logging.INFO)
        
        log_message = f"ALERT [{alert.alert_type.value}] {alert.title}: {alert.message}"
        logger.log(log_level, log_message)
    
    def _send_console_alert(self, alert: Alert):
        """Send alert to console"""
        severity_colors = {
            AlertSeverity.INFO: '\033[36m',      # Cyan
            AlertSeverity.WARNING: '\033[33m',   # Yellow
            AlertSeverity.ERROR: '\033[31m',     # Red
            AlertSeverity.CRITICAL: '\033[35m'   # Magenta
        }
        
        color = severity_colors.get(alert.severity, '')
        reset_color = '\033[0m'
        
        print(f"{color}[{alert.severity.value.upper()}] {alert.title}{reset_color}")
        print(f"  {alert.message}")
        print(f"  Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def _cleanup_loop(self):
        """Background cleanup of old alerts and counters"""
        while True:
            try:
                time.sleep(3600)  # Run every hour
                current_time = datetime.now()
                
                # Clean up old alert history (keep last 1000)
                with self.lock:
                    if len(self.alerts_history) > 1000:
                        self.alerts_history = self.alerts_history[-500:]
                
                # Clean up old rate limiting counters
                current_hour = current_time.hour
                keys_to_remove = [
                    key for key in self.alert_counts.keys()
                    if key.endswith(f"_{current_hour - 2}") or key.endswith(f"_{current_hour - 1}")
                ]
                
                for key in keys_to_remove:
                    del self.alert_counts[key]
                
            except Exception as e:
                logger.error(f"Alert cleanup error: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        with self.lock:
            for alert in self.alerts_history:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_by = acknowledged_by
                    alert.acknowledged_time = datetime.now()
                    logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                    return True
        return False
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_alerts = [
                alert.to_dict() for alert in self.alerts_history
                if alert.timestamp >= cutoff_time
            ]
        
        return recent_alerts
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alert system statistics"""
        with self.lock:
            return self.stats.copy()
    
    # Convenience methods for common alerts
    def boundary_violation_alert(self, dog_name: str, zone_name: str, camera_id: int):
        """Send boundary violation alert"""
        self.send_alert(
            AlertType.BOUNDARY_VIOLATION,
            AlertSeverity.WARNING,
            f"Boundary Violation: {dog_name}",
            f"{dog_name} has violated the boundary in {zone_name}",
            {'dog_name': dog_name, 'zone_name': zone_name, 'camera_id': camera_id}
        )
    
    def camera_offline_alert(self, camera_id: int, camera_name: str):
        """Send camera offline alert"""
        self.send_alert(
            AlertType.CAMERA_OFFLINE,
            AlertSeverity.ERROR,
            f"Camera Offline: {camera_name}",
            f"Camera {camera_id} ({camera_name}) has gone offline",
            {'camera_id': camera_id, 'camera_name': camera_name}
        )
    
    def system_error_alert(self, error_message: str, component: str):
        """Send system error alert"""
        self.send_alert(
            AlertType.SYSTEM_ERROR,
            AlertSeverity.ERROR,
            f"System Error in {component}",
            error_message,
            {'component': component}
        )

# Global alert system instance
alert_system = AlertSystem()

def get_alert_system() -> AlertSystem:
    """Get global alert system instance"""
    return alert_system