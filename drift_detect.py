import os
import time
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
from collections import Counter
import prometheus_client 
from prometheus_client import start_http_server, Gauge, Histogram
from prometheus_client.core import CollectorRegistry
import signal
import sys
from PIL import Image # Pillow for image loading
import numpy as np # For brightness calculation
import subprocess

# --- Configuration ---
MONGO_URI = os.getenv('MONGO_URI', "mongodb://localhost:27017/")
DB_NAME = "image_organizer"
COLLECTION_NAME = "image_metadata"
PROMETHEUS_PORT = int(os.getenv('PROMETHEUS_PORT', 8001))
SCRAPE_INTERVAL_SECONDS = int(os.getenv('SCRAPE_INTERVAL_SECONDS', 60))
TIME_WINDOW_MINUTES = int(os.getenv('TIME_WINDOW_MINUTES', 60))
METRIC_PREFIX = "ai_img_org_"

# Add drift detection configuration
DRIFT_CONFIG = {
    'brightness_threshold_high': 200,
    'brightness_threshold_low': 50,
    'class_distribution_threshold': 0.3,  # 30% change in distribution
    'min_drift_samples': 500,  # Minimum samples before considering drift
    'drift_window_hours': 1,  # Time window to accumulate drift samples
    'retrain_threshold': 0.75,  # 75% of samples showing drift triggers retraining
    'min_new_samples': 200,  # Minimum new samples needed in last hour
    'retrain_cooldown_minutes': 60  # Minimum time between retraining
}

# Track last retraining time
last_retraining_time = None

# Create a new registry for our metrics
registry = CollectorRegistry()

# --- Prometheus Metrics Definition ---
PREDICTED_CLASS_DISTRIBUTION = Gauge(
    f'{METRIC_PREFIX}predicted_class_distribution_percent',
    'Distribution of predicted classes in the last time window (%)',
    ['predicted_class'],
    registry=registry
)
RECORDS_PROCESSED_IN_WINDOW = Gauge(
    f'{METRIC_PREFIX}records_processed_in_window_total',
    'Total number of records processed in the current time window',
    registry=registry
)
BRIGHTNESS_BUCKETS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255]
AVERAGE_BRIGHTNESS_DISTRIBUTION = Histogram(
    f'{METRIC_PREFIX}average_brightness_distribution',
    'Distribution of average brightness calculated from files in the last time window',
    buckets=BRIGHTNESS_BUCKETS,
    registry=registry
)
FILES_READ_ERRORS = prometheus_client.Counter(
    f'{METRIC_PREFIX}files_read_errors_total',
    'Total number of errors encountered reading image files for drift detection',
    registry=registry
)

# --- Helper Function (can reuse from FastAPI or define here) ---
def calculate_brightness(image_path):
    """Calculate normalized brightness value between 0-255"""
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img_array = np.array(img)
        brightness = np.mean(img_array)
        # Ensure brightness is in 0-255 range
        brightness = np.clip(brightness, 0, 255)
        return brightness
    except Exception as e:
        print(f"Error calculating brightness for {image_path}: {e}")
        return None

# --- MongoDB Connection ---
try:
    print(f"Connecting to MongoDB at {MONGO_URI}...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    client.admin.command('ping')
    print("MongoDB connection successful.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    sys.exit(1)

# --- Graceful Shutdown Handling ---
shutdown_flag = False
def handle_shutdown(sig, frame):
    global shutdown_flag
    print(f"Received signal {sig}, shutting down...")
    shutdown_flag = True
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

def update_drift_status(collection, start_time, end_time, drift_detected):
    """Update drift status for records in the time window"""
    collection.update_many(
        {"timestamp": {"$gte": start_time, "$lt": end_time}},
        {"$set": {"drift_detected": drift_detected}}
    )

def check_drift_conditions(collection):
    """Check if enough drift has been detected to trigger retraining"""
    global last_retraining_time
    
    current_time = datetime.now(timezone.utc)
    
    # Check cooldown period
    if last_retraining_time and (current_time - last_retraining_time) < timedelta(minutes=DRIFT_CONFIG['retrain_cooldown_minutes']):
        print("Still in cooldown period after last retraining")
        return False

    # Check for new samples in the last hour that haven't been used in training
    one_hour_ago = current_time - timedelta(hours=1)
    new_samples = list(collection.find({
        "timestamp": {"$gte": one_hour_ago},
        "used_in_training": {"$ne": True}  # Only get samples not used in training
    }))
    
    if len(new_samples) < DRIFT_CONFIG['min_new_samples']:
        print(f"Not enough new unused samples in the last hour: {len(new_samples)} < {DRIFT_CONFIG['min_new_samples']}")
        return False

    # Get samples since last retraining that haven't been used in training
    start_time = last_retraining_time if last_retraining_time else one_hour_ago
    
    # Check drift in unused samples since last retraining
    drift_samples = list(collection.find({
        "timestamp": {"$gte": start_time},
        "drift_detected": True,
        "used_in_training": {"$ne": True}
    }))
    
    total_samples = list(collection.find({
        "timestamp": {"$gte": start_time},
        "used_in_training": {"$ne": True}
    }))
    
    if len(total_samples) < DRIFT_CONFIG['min_drift_samples']:
        return False
    
    drift_ratio = len(drift_samples) / len(total_samples)
    print(f"Drift ratio in unused samples: {drift_ratio:.2f} ({len(drift_samples)}/{len(total_samples)} samples)")
    
    return drift_ratio >= DRIFT_CONFIG['retrain_threshold']

def trigger_retraining():
    """Trigger the retraining pipeline"""
    global last_retraining_time
    
    try:
        subprocess.run(["python", "ml_pipeline/retrain_pipeline.py"], check=True)
        last_retraining_time = datetime.now(timezone.utc)
        print("Retraining triggered successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error triggering retraining: {e}")

def reset_metrics():
    """Reset all metrics for the new monitoring window"""
    # Reset Gauges
    for cls in ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]:
        PREDICTED_CLASS_DISTRIBUTION.labels(predicted_class=cls).set(0)
    RECORDS_PROCESSED_IN_WINDOW.set(0)
    
    # Note: Histogram and Counter metrics cannot be reset in Prometheus
    # They are cumulative by design

# --- Main Monitoring Loop ---
def run_monitoring():
    """Run continuous monitoring of image metadata"""
    global last_retraining_time
    
    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Start Prometheus metrics server with our registry
    start_http_server(PROMETHEUS_PORT, registry=registry)
    print(f"Starting Prometheus metrics server on port {PROMETHEUS_PORT}")

    # Track processed images to avoid duplicates
    processed_images = set()

    print("Monitoring started...")
    while not shutdown_flag:
        try:
            # Reset metrics at the start of each monitoring window
            reset_metrics()
            
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=TIME_WINDOW_MINUTES)
            
            # Get records in the current time window that haven't been processed
            records = list(collection.find({
                "timestamp": {"$gte": start_time, "$lt": end_time},
                "used_in_training": {"$ne": True},  # Only consider unused samples
                "random_name": {"$nin": list(processed_images)}  # Only get unprocessed images
            }))
            
            total_records = len(records)
            if total_records == 0:
                time.sleep(SCRAPE_INTERVAL_SECONDS)
                continue

            # Calculate metrics
            brightness_values = []
            class_counts = Counter()

            known_classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
            
            for record in records:
                # Get image brightness
                try:
                    brightness = calculate_brightness(record['path'])
                    if brightness is not None:
                        brightness_values.append(brightness)
                        # Add to processed images set
                        processed_images.add(record['random_name'])
                except Exception as e:
                    print(f"Error processing image {record['path']}: {e}")
                    continue

                # Count predicted classes
                predicted_class = record.get('predicted_class')
                if predicted_class:
                    class_counts[predicted_class] += 1

            # Update class distribution metrics
            distribution = {}
            for cls in known_classes:
                count = class_counts.get(cls, 0)
                percentage = (count / total_records * 100) if total_records > 0 else 0
                distribution[cls] = percentage
                PREDICTED_CLASS_DISTRIBUTION.labels(predicted_class=cls).set(percentage)

            # Update other metrics only for new images
            RECORDS_PROCESSED_IN_WINDOW.set(total_records)
            
            if brightness_values:
                for brightness in brightness_values:
                    AVERAGE_BRIGHTNESS_DISTRIBUTION.observe(brightness)

            # Detect drift based on multiple conditions
            drift_detected = False
            
            # Check brightness drift
            if brightness_values:
                mean_brightness = np.mean(brightness_values)
                if (mean_brightness > DRIFT_CONFIG['brightness_threshold_high'] or 
                    mean_brightness < DRIFT_CONFIG['brightness_threshold_low']):
                    drift_detected = True
                    print(f"Brightness drift detected: {mean_brightness}")
            
            # Check class distribution drift
            expected_percentage = 100 / len(known_classes)
            for cls, percentage in distribution.items():
                if abs(percentage - expected_percentage) > (DRIFT_CONFIG['class_distribution_threshold'] * 100):
                    drift_detected = True
                    print(f"Class distribution drift detected for {cls}: {percentage:.2f}% (expected: {expected_percentage:.2f}%)")

            # Update drift status for current window's records
            if drift_detected:
                collection.update_many(
                    {
                        "timestamp": {"$gte": start_time, "$lt": end_time},
                        "used_in_training": {"$ne": True},
                        "random_name": {"$nin": list(processed_images)}
                    },
                    {"$set": {"drift_detected": True}}
                )
                
                # Check if retraining should be triggered
                if check_drift_conditions(collection):
                    print("Drift threshold exceeded and enough new samples collected. Triggering retraining...")
                    trigger_retraining()
                    # Clear processed images after retraining
                    processed_images.clear()

            # Periodically clear old entries from processed_images set
            # (optional, to prevent memory growth over very long runs)
            if len(processed_images) > 10000:  # Arbitrary threshold
                processed_images.clear()
                print("Cleared processed images cache")

            time.sleep(SCRAPE_INTERVAL_SECONDS)

        except Exception as e:
            print(f"Error during monitoring loop: {e}")
            time.sleep(SCRAPE_INTERVAL_SECONDS)

    print("Monitoring loop stopped.")
    client.close()
    print("MongoDB connection closed.")


if __name__ == "__main__":
    run_monitoring()