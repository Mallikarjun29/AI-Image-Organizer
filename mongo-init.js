db = db.getSiblingDB('image_organizer');

// Create collections
db.createCollection('image_metadata');

// Create indexes
db.image_metadata.createIndex({ "random_name": 1 }, { unique: true });
db.image_metadata.createIndex({ "timestamp": 1 });
db.image_metadata.createIndex({ "drift_detected": 1 });
db.image_metadata.createIndex({ "used_in_training": 1 });

// Create compound indexes for efficient querying
db.image_metadata.createIndex({ 
    "timestamp": 1, 
    "drift_detected": 1, 
    "used_in_training": 1 
});